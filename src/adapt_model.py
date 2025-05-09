import os
from pathlib import Path
from typing import Dict, List, Tuple
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import shutil
import numpy as np
import random

from rome import ROMEHyperParams, apply_rome_to_model, execute_rome
from dama import DAMAHyperParams, apply_dama_to_model, execute_dama
from AlphaEdit.AlphaEdit_hparams import AlphaEditHyperParams
from AlphaEdit2.alphaedit2_hparams import AlphaEdit2HyperParams
from AlphaEdit.AlphaEdit_main import apply_AlphaEdit_to_model
from AlphaEdit2.alphaedit2_main import apply_alphaedit2_to_model
from dama_l import DAMALeaceHyperParams
from dama_l.dama_l_main import apply_dama_l_to_model, execute_dama_l
from memit import MEMITHyperParams, apply_memit_to_model, execute_memit
from ft import FTHyperParams, apply_ft_to_model, execute_ft

from utils.generate import generate_interactive, generate_fast
from utils import nethook
from utils.globals import *
from utils.model_utils import *

from AlphaEdit.AlphaEdit_main import get_cov

import argparse
import sys


def load_projection_matrices(projections_path: str, hparams) -> Dict[int, torch.Tensor]:
    """
    Loads projection matrices saved as P_{i}.npy for each layer in hparams.
    Returns: Dictionary mapping layer index (int) → torch.Tensor of shape [d, d]
    """
    P = {}
    for i, layer in enumerate(hparams.layers):
        path = os.path.join(projections_path, f"P_{i}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing projection matrix at: {path}")
        P[layer] = torch.from_numpy(np.load(path)).float()
        print(f"Loaded P[{layer}] from {path} with shape {P[layer].shape}")
    return P

def apply_method(apply_function,
                    model: AutoModelForCausalLM,
                    tok: AutoTokenizer,
                    requests: List[Dict],
                    hparams: ROMEHyperParams | DAMAHyperParams | AlphaEditHyperParams | AlphaEdit2HyperParams,
                    saveto=None, loadrom=None):
    orig_weights = None
    if loadrom:
        print("Loading model from", loadrom)
        model_new = AutoModelForCausalLM.from_pretrained(loadrom, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                                      low_cpu_mem_usage=True, device_map='auto')
        orig_weights = None
    else:
        if apply_function == apply_AlphaEdit_to_model:
            # Load or create projection matrices here
            W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
            P, cache_c = None, None

            if 'gpt' in args.model_name:
                hidden_size = W_out.shape[0]          # 1600
                cache_c = torch.zeros(len(hparams.layers), hidden_size, hidden_size, device="cpu")
                P       = torch.zeros_like(cache_c)
            else:
                cache_c = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
                P = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")

            for i, layer in enumerate(hparams.layers):
                P[i, :, :] = get_project(model,tok,layer,hparams)
            torch.save(P, "null_space_project.pt")

            model_new, orig_weights = apply_AlphaEdit_to_model(
                model, tok, requests, hparams, cache_template=None, cache_c=cache_c, P=P
            )
        elif apply_function == apply_alphaedit2_to_model:
            # Load or create projection matrices here
            P = load_projection_matrices("projections",hparams)  # You’ll need to define this
            model_new, orig_weights = apply_alphaedit2_to_model(
                model, tok, requests, hparams, P, return_orig_weights=True
            )
        else:
            model_new, orig_weights = apply_function(model, tok, requests, hparams, return_orig_weights=True)
    if saveto:
        print("Saving model to", saveto)
        model.generation_config.pad_token_id = tok.eos_token_id
        model_new.save_pretrained(saveto)
    return model_new, orig_weights

def get_project(model, tok, layer, hparams):
    force_recompute = False
    cov = get_cov(
        model,
        tok,
        hparams.rewrite_module_tmp.format(layer),
        hparams.mom2_dataset,
        hparams.mom2_n_samples
        if not force_recompute
        else hparams.mom2_n_samples // 1000,
        hparams.mom2_dtype,
        force_recompute=force_recompute,
    ).cpu()
    U, S, _ = torch.linalg.svd(cov, full_matrices=False)
    threshold = hparams.nullspace_threshold
    small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]
    print(len(small_singular_indices))
    return U[:, small_singular_indices] @ U[:, small_singular_indices].T

def model_editing(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    generation_prompts: List[str],
    hparams: ROMEHyperParams | DAMAHyperParams,
    method: str,
    projections_saveto: Path = None,
    projections_loadfrom: Path = None,
    output_dir: Path = None,
    ncv: bool = False,
    val: bool = False,
    use_neutral: bool = False
) -> tuple[AutoModelForCausalLM | AutoModelForCausalLM, list[str]]:
    """
    Applies the selected model editing algorithm. Generates text both before and after
    for comparison of model behavior. Returns the updated model and the original values of
    weights that were changed.
    """

    nethook.set_requires_grad(True, model)

    print("Generating pre-update text")
    pre_update_text = generate_fast(model, tok, generation_prompts, max_out_len=100)
    print(pre_update_text)

    print(f"Applying {method} to model")
    if method == 'MEMIT':
        model_new, orig_weights= apply_method(apply_memit_to_model, model, tok, requests, hparams,
                                              projections_saveto, projections_loadfrom)
    elif method == 'FT':
        model_new, orig_weights = apply_method(apply_ft_to_model, model, tok, requests, hparams,
                                               projections_saveto, projections_loadfrom)
    elif method == 'ROME':
        model_new, orig_weights = apply_method(apply_rome_to_model, model, tok, requests, hparams,
                                               projections_saveto, projections_loadfrom)
    elif method == 'DAMA':
        model_new, orig_weights = apply_dama_to_model(
            model, tok, requests, hparams, copy=False, return_orig_module=True,
            projections_saveto=projections_saveto, projections_loadfrom=projections_loadfrom,
            output_dir=output_dir, ncv=ncv, val=val, use_neutral=use_neutral)
    elif method == 'DAMA_L':
        model_new, orig_weights = apply_dama_l_to_model(
            model, tok, requests, hparams, copy=False, return_orig_module=True,
            projections_saveto=projections_saveto, projections_loadfrom=projections_loadfrom,
            output_dir=output_dir)
    elif method == 'ALPHA_EDIT':
        model_new, orig_weights = apply_method(apply_AlphaEdit_to_model, model, tok, requests, hparams,
                                               projections_saveto, projections_loadfrom)
    elif method == 'ALPHA_EDIT2':
        model_new, orig_weights = apply_method(apply_alphaedit2_to_model, model, tok, requests, hparams,
                                               projections_saveto, projections_loadfrom)
    else:
        raise ValueError(f"Unknown method {method}. Choose from: ROME, DAMA")

    print("Generating post-update text")
    post_update_text = generate_fast(
        model_new, tok, generation_prompts, max_out_len=100
    )
    print(post_update_text)

    print("Summarizing differences")
    for i, (prompt, pre, post) in enumerate(
        zip(generation_prompts, pre_update_text, post_update_text)
    ):
        if i > 0:
            print("".join(["-" for _ in range(10)]))

        prompt_str = "[Prompt]:"
        pre_str = f"[Pre-{method}]:"
        post_str = f"[Post-{method}]:"
        pad_to = 1 + max(len(prompt_str), len(pre_str), len(post_str))

        for s, t in zip([prompt_str, post_str, pre_str], [prompt, post, pre]):
            print(s.ljust(pad_to), t)

    return model_new, orig_weights


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llama")
    parser.add_argument("--param_number", type=int, default=None)
    parser.add_argument("--method", type=str, default="ROME")
    parser.add_argument("--request_file", type=str, default=None)
    parser.add_argument("--multilingual_request_files", type=str, default=None, help="Dictionary of language to request file")
    parser.add_argument("--generation_file", type=str, default=None)
    parser.add_argument("--save_projections", type=bool, default=True)
    parser.add_argument("--load_projections", type=bool, default=False)
    parser.add_argument("--compare_against", type=bool, default=False)
    parser.add_argument("--num_layers", type=int, default=9)
    parser.add_argument("--task", type=str, default="gen")
    parser.add_argument("--batch_size", type=int, default=1)
    # legacy DAMA arguments
    parser.add_argument("--iterative_update", type=bool, default=False) # legacy DAMA: False
    parser.add_argument("--mixed_update", type=bool, default=False) # legacy DAMA: True
    parser.add_argument("--post_linear", type=bool, default=False)  # legacy DAMA: True
    parser.add_argument("--orthogonal_constraint", type=float, default=None) # legacy DAMA: 0
    parser.add_argument("--null_dim", type=int, default=1024) # legacy DAMA: self-adapt with Leace
    parser.add_argument("--no_colinear_vs", type=bool, default=False) # legacy DAMA: True
    parser.add_argument("--vs_at_last", type=bool, default=False)  # legacy DAMA: True
    parser.add_argument("--use_neutral", type=bool, default=False) # legacy DAMA: True
    parser.add_argument("--delta_only", type=bool, default=False) # legacy DAMA: True
    parser.add_argument("--no_whitening", type=bool, default=False) # legacy DAMA: False
    parser.add_argument("--random_seed", type=int, default=0)
    args = parser.parse_args()

    print(f"Setting random seeds at: {args.random_seed}")
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    
    print(f"Load original model to compare: {args.compare_against}")
    model_name, model, orig_model, tok = get_model_tokenizer(args.model_name, args.param_number, args.compare_against)

    experiment_name_suffix = parse_experiment_name(
        num_layers=args.num_layers, iterative_update=args.iterative_update, mixed_update=args.mixed_update,
        task=args.task,
        post_linear=args.post_linear, batch_size=args.batch_size, orthogonal_constraint=args.orthogonal_constraint,
        no_colinear_vs=args.no_colinear_vs, vs_at_last=args.vs_at_last, null_dim=args.null_dim, use_neutral=args.use_neutral,
        delta_only=args.delta_only, nw=args.no_whitening, seed=args.random_seed
    )
    experiment_name = f"{experiment_name_suffix}"
    if args.method == "DAMA":
        output_dir = os.path.join(RESULTS_DIR, args.method, model_name, experiment_name)
        print(f"Conducting experiment: {experiment_name}.")
    else:
        output_dir = os.path.join(RESULTS_DIR, args.method, f"{model_name}_{str(args.num_layers)}L")
    if args.multilingual_request_files is not None:
        output_dir += "_multilingual"
    os.makedirs(output_dir, exist_ok=True)

    request = []
    if args.request_file is not None:
        with open(os.path.join(DATA_DIR, args.request_file), "r") as f:
            request = json.load(f)

    if args.multilingual_request_files is not None:
        request = parse_multilingual_request_files(args.multilingual_request_files, model_name)


    if args.generation_file is not None:
        with open(os.path.join(DATA_DIR, args.generation_file), "r") as f:
            generation_prompts = json.load(f)
    else:
        generation_prompts = [
            "A secretary said that",
            "A nurse said that",
            "A nurse slept because",
            "A secretary slept because",
            "A secretary was tired because",
            "A secretary whispered because",
        ]

    projections_saveto = None
    projections_loadfrom = None
    if args.method in ("DAMA", "DAMA_L"):
        if args.load_projections:
            projections_loadfrom = os.path.join(output_dir, "projections.npy")
        if args.save_projections:
            projections_saveto = os.path.join(output_dir, "projections.npy")

    elif args.method in ("MEMIT", "FT" , "ROME", "ALPHA_EDIT"):
        if args.load_projections:
            projections_loadfrom = output_dir
        if args.save_projections:
            projections_saveto = output_dir

    
    print(f"Retrieving {args.method} hyperparameters")

    if args.method == 'DAMA':
        hparams_path = os.path.join(HPARAMS_DIR, args.method, model_name, f"{experiment_name}.json")
        hparams = DAMAHyperParams.from_json(hparams_path)
    elif args.method == 'DAMA_L':
        hparams_path = os.path.join(HPARAMS_DIR, args.method, f"{model_name}_{str(args.num_layers)}L.json")
        hparams = DAMALeaceHyperParams.from_json(hparams_path)
    elif args.method == 'MEMIT':
        hparams_path = os.path.join(HPARAMS_DIR, args.method, f"{model_name}.json")
        hparams = MEMITHyperParams.from_json(hparams_path)
    elif args.method == 'FT':
        hparams_path = os.path.join(HPARAMS_DIR, args.method, f"{model_name}.json")
        hparams = FTHyperParams.from_json(hparams_path)
    elif args.method == 'ROME':
        hparams_path = os.path.join(HPARAMS_DIR, args.method, f"{model_name}.json")
        hparams = ROMEHyperParams.from_json(hparams_path)
    elif args.method == 'ALPHA_EDIT':
        hparams_path = os.path.join(HPARAMS_DIR, args.method, f"{model_name}.json")
        hparams = AlphaEditHyperParams.from_json(hparams_path)
        # hparams = DAMAHyperParams.from_json(hparams_path)
    elif args.method == 'ALPHA_EDIT2':
        hparams_path = os.path.join(HPARAMS_DIR, args.method, f"{model_name}.json")
        hparams = AlphaEdit2HyperParams.from_json(hparams_path)
    else:
        raise ValueError(f"Unknown method {args.method}. Choose from: ROME, DAMA")
    print("Loaded from", hparams_path)
    print(hparams)

    model_new, orig_weights = model_editing(
        model, tok, request, generation_prompts, hparams, args.method,
        projections_saveto=projections_saveto, projections_loadfrom=projections_loadfrom, output_dir=output_dir,
        ncv=args.no_colinear_vs, val=args.vs_at_last, use_neutral=args.use_neutral)

    print(f"Dumping parameters and code to: {output_dir}")
    shutil.copy(hparams_path, os.path.join(output_dir, "hparams.json"))
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f)
    with open(sys.argv[0], 'r') as this_code, open(os.path.join(output_dir, 'adapt_model.py'), 'w') as source_out:
        code_lines = this_code.readlines()
        source_out.writelines(code_lines)

    if 'llama' in args.model_name:
        generate_interactive(model_new, tok, max_out_len=100, use_logit_lens=True,
                            layer_module_tmp= "model.layers.{}",
                            ln_f_module= "model.norm",
                            lm_head_module= "lm_head",
                            compare_against=orig_model)
    else:
        generate_interactive(model_new, tok, max_out_len=100, use_logit_lens=True,
                             layer_module_tmp = hparams.layer_module_tmp,
                             ln_f_module= hparams.ln_f_module,
                             lm_head_module= hparams.lm_head_module,
                             compare_against=orig_model)
