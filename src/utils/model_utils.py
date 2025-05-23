import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tokenizers import processors
import json
import langcodes

from utils import nethook
from utils.globals import *
from utils.constants import MODEL_NAME_MAP, TRANSLATION_PROMPTS
from dama.dama_main import apply_dama_on_module


def parse_experiment_name(num_layers: int = 9,
                          iterative_update: bool = False,
                          mixed_update: bool = False,
                          task: str = "gen",
                          post_linear: bool = False,
                          batch_size: int = 1,
                          orthogonal_constraint: float = 0.,
                          no_colinear_vs: bool = False,
                          vs_at_last: bool = False,
                          null_dim: int = 1024,
                          use_neutral: bool = False,
                          delta_only: bool = False,
                          use_neutral_tensor: bool = False,
                          nw: bool = False,
                          seed: int = 0
                          ) -> str:
    experiment_string = f"l{num_layers}"
    if iterative_update:
        experiment_string += "_iter"
    elif mixed_update:
        experiment_string += ""
    else:
        experiment_string += "_once"

    if post_linear:
        experiment_string += ""
    else:
        experiment_string += "_prel"

    if task == "gen":
        experiment_string += ""
    elif task == "coref":
        experiment_string += "_coref"
        raise NotImplementedError("Coreference resolution task not implemented yet")
    else:
        raise ValueError("Unknown task. Choose from: gen, coref")

    if batch_size == 1:
        experiment_string += ""
    elif batch_size > 1:
        experiment_string += f"_b{batch_size}"
    else:
        raise ValueError("Batch size must be a positive integer")

    if orthogonal_constraint:
        experiment_string += f"_o{orthogonal_constraint}"
    else:
        experiment_string += "_on"

    if null_dim != 1024:
        experiment_string += f"_nd{null_dim}"

    if no_colinear_vs:
        experiment_string += "_ncv"

    if vs_at_last:
        experiment_string += "_val"

    if use_neutral:
        experiment_string += "_neutral"
    if delta_only:
        experiment_string += "_delta_only"

    if use_neutral_tensor:
        experiment_string += "_nt"

    if nw:
        experiment_string += '_nw'

    if seed != 0:
        experiment_string += f"_s{seed}"

    return experiment_string


def parse_multilingual_request_files(multilingual_request_files: str, model_name: str):
    multilingual_requests = json.loads(multilingual_request_files)
    
    all_requests = []
    translation_prompt = TRANSLATION_PROMPTS.get(model_name, "{src_lang}: {src_sentence} {tgt_lang}: ".format)
    
    for lang, req_file in multilingual_requests.items():
        with open(os.path.join(DATA_DIR, req_file), 'r') as f:
            requests = json.load(f)
            for req in requests:
                if lang != "en":
                    req["prompt"] = translation_prompt(src_lang=langcodes.Language("en").language_name(),
                                                       tgt_lang=langcodes.Language(lang).language_name(),
                                                       src_sentence=req["src_sentence"]) + req["prompt"]
                    req["prompt"] = req["prompt"].replace("  ", " ").strip()
                req["targets"] ={polv: token for polv, token in zip(["pos", "neg"], req["completions"])}
        all_requests.extend(requests)
    np.random.shuffle(all_requests)
    
    return all_requests


def load_dama_model(model, hparams, projection_file):
    layers = dict(model.named_modules())
    devices = [layers["model.layers.{}.mlp".format(i)].down_proj.weight.device for i in hparams.layers]
    print(f"Loading projections from {projection_file}")
    loaded_projections = np.load(projection_file, allow_pickle=True).item()
    if torch.cuda.is_available():
        projections = {m_name: (torch.tensor(values['M'], device=dev, dtype=torch.float16),
                                torch.tensor(values['mu_in'], device=dev, dtype=torch.float16),
                                torch.tensor(values['mu_out'], device=dev, dtype=torch.float16))
                       for dev, (m_name, values) in zip(devices, loaded_projections.items())}
    else:
        projections = {m_name: (torch.tensor(values['M'], device='cpu', dtype=torch.float32),
                                torch.tensor(values['mu_in'], device='cpu', dtype=torch.float32),
                                torch.tensor(values['mu_out'], device='cpu', dtype=torch.float32))
                       for m_name, values in loaded_projections.items()}

    with torch.no_grad():
        for m_name, (P, mu_in, mu_out) in projections.items():
            if int(m_name.split('.')[2]) not in hparams.layers:
                continue

            orig_module = nethook.get_module(model, m_name)
            new_module = apply_dama_on_module(orig_module, P, mu_in, mu_out,
                                              hparams.projection_location if hasattr(hparams, "projection_location") else "after")

            nethook.replace_module(model, m_name, new_module)

        print(f"New weights successfully inserted into layers: {hparams.layers}")

    return model


def get_model_tokenizer(model_name, param_number, compare_against=False):
    if model_name.endswith("llama"):
        model_name = "llama"
        model_path = os.path.join(MODEL_DIR, "llama")
        if param_number in {7, 13, 30, 65}:
            model_name += f"_{param_number}B"
            model_path += f"_{param_number}B"
        tokenizer_path = model_path
    elif model_name == "HuggingFaceM4/tiny-random-LlamaForCausalLM":
        model_path = model_name
        model_name = "llama_tiny"
        # For debugging purposes
        # tokenizer_path = os.path.join(MODEL_DIR, "llama_7B")
        tokenizer_path = model_path
    else:
        model_path = model_name
        tokenizer_path = model_name
        model_name = model_name.split("/")[-1].replace('-','_')
        if model_name in MODEL_NAME_MAP:
            model_name = MODEL_NAME_MAP[model_name]

    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                                 low_cpu_mem_usage=True, device_map='auto', offload_folder="offload")
    

    if torch.cuda.is_available() and torch.cuda.device_count() == 1:
        model = model.eval().cuda()
    elif torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = model.eval()

    if compare_against:
        orig_model = AutoModelForCausalLM.from_pretrained(model_path,
                                                          torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                                          low_cpu_mem_usage=True, device_map='auto', offload_folder="offload")
        if torch.cuda.is_available():
            orig_model = orig_model.eval().cuda()
    else:
        orig_model = None

    use_fast = False if "gpt2" in model_name else True
    tok = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=use_fast, return_token_type_ids=False, add_bos_token=False)
    # set llama special tokens
    if "Llama_3" not in model_name:
        tok.bos_token = "<s>"
        tok.eos_token = "</s>"
        tok.unk_token = "<unk>"
        #for gpt2-xl
        tok.add_special_tokens({"pad_token": tok.eos_token})
        tok.padding_side = "right"
        model.resize_token_embeddings(len(tok))
    else:
        # hack to do not make the bos token in llama3 (until tokenizer is fixed)
        tok._tokenizer.post_processor = processors.Sequence([processors.ByteLevel(trim_offsets=False),
                                                             processors.TemplateProcessing(single=f"$A:0",
                                                                                           pair=f"$A:0 {tok.bos_token}:1 $B:1",
                                                                                           special_tokens=[(tok.bos_token, tok.bos_token_id),
                                                                                                           (tok.eos_token, tok.eos_token_id)],
                                                                                           ),
                                                             ])

    return model_name, model, orig_model,  tok
