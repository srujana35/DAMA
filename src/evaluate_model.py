import argparse
import os
import sys

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import nethook
from Baselines.dama.dama_main import apply_dama_on_module
from Baselines.dama.dama_hparams import DAMAHyperParams
# from Baselines.dama_l.dama_l_hparams import DAMALeaceHyperParams
from Baselines.memit.memit_main import MEMITHyperParams
from AlphaEdit.AlphaEdit_hparams import AlphaEditHyperParams
from Baselines.ft.ft_main import FTHyperParams
from utils.globals import *
from utils.model_utils import *
from utils.generate import generate_interactive, generate_fast
from adapt_model import get_model_tokenizer, parse_experiment_name

from evaluation import EvaluateGeneration, EvaluateCoreference, EvaluateCausalLM, EvaluateQA,\
    EvaluateStereoset, EvaluateTranslation, EvaluateARC

def run_evaluation_on_task(model, tokenizer, model_name, task, test_file, output_dir):
    # Check if test file exists
    # if test_file:
    #     test_file_path = os.path.join(DATA_DIR, test_file)
    #     if not os.path.exists(test_file_path):
    #         print(f"Error: Test file not found at {test_file_path}")
    #         sys.exit(1)
    # else:
    #     test_file_path = None

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    try:
        if task == "gen":
            evaluator = EvaluateGeneration(model, tokenizer, test_file_path, task)
        elif task == "coref":
            evaluator = EvaluateCoreference(model, tokenizer, test_file_path, task)
        elif task == "causal_lm":
            evaluator = EvaluateCausalLM(model, tokenizer, test_file, task)
        elif task == "stereoset":
            evaluator = EvaluateStereoset(model, tokenizer, test_file_path, task)
        elif task == "interactive":
            generate_interactive(model, tokenizer, max_out_len=100, use_logit_lens=True,
                            layer_module_tmp= "model.layers.{}",
                            ln_f_module= "model.norm",
                            lm_head_module= "lm_head",
                            compare_against=None)
        elif task == "qa":
            evaluator = EvaluateQA(model, tokenizer, test_file_path, task)
        elif task == "translation":
            evaluator = EvaluateTranslation(model, tokenizer, test_file, task, model_name)
        elif task in ["arc-challenge", "arc-easy"]:
            evaluator = EvaluateARC(model, tokenizer, test_file, task)
        else:
            raise ValueError(f"Unknown task {task}")

        evaluator.evaluate()
        evaluator.save_results(output_dir)
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        sys.exit(1)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llama")
    parser.add_argument("--param_number", type=int, default=None)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--test_file", type=str, default=None)
    parser.add_argument("--test_task", type=str, default="gen")
    parser.add_argument("--num_layers", type=int, default=9)
    parser.add_argument("--task", type=str, default="gen")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--multilingual_training", type=bool, default=False)
    # legacy DAMA arguments
    parser.add_argument("--iterative_update", type=bool, default=False)
    parser.add_argument("--mixed_update", type=bool, default=False)
    parser.add_argument("--post_linear", type=bool, default=False)
    parser.add_argument("--orthogonal_constraint", type=float, default=None)
    parser.add_argument("--null_dim", type=int, default=1024)
    parser.add_argument("--no_colinear_vs", type=bool, default=False)
    parser.add_argument("--vs_at_last", type=bool, default=False)
    parser.add_argument("--use_neutral", type=bool, default=False)
    parser.add_argument("--delta_only", type=bool, default=False)
    parser.add_argument("--no_whitening", type=bool, default=False)
    parser.add_argument("--random_seed", type=int, default=0)

    args = parser.parse_args()

    # Check if DATA_DIR exists
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory not found at {DATA_DIR}")
        sys.exit(1)

    try:
        model_name, model, _, tok = get_model_tokenizer(args.model_name, args.param_number, False)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        sys.exit(1)

    tok.bos_token = "<s>"
    tok.eos_token = "</s>"
    tok.unk_token = "<unk>"
    tok.add_special_tokens({
        "bos_token": tok.bos_token,
        "eos_token": tok.eos_token,
        "unk_token": tok.unk_token,
        "pad_token": tok.eos_token,   # reuse </s> as pad
    })
    tok.padding_side = "right"
    model.resize_token_embeddings(len(tok))

    model.config.bos_token_id = tok.bos_token_id
    model.config.eos_token_id = tok.eos_token_id
    model.config.unk_token_id = tok.unk_token_id
    model.config.pad_token_id = tok.pad_token_id

    experiment_name_suffix = parse_experiment_name(
        num_layers=args.num_layers, iterative_update=args.iterative_update, mixed_update=args.mixed_update,
        task=args.task,
        post_linear=args.post_linear, batch_size=args.batch_size, orthogonal_constraint=args.orthogonal_constraint,
        no_colinear_vs=args.no_colinear_vs, vs_at_last=args.vs_at_last, null_dim=args.null_dim, use_neutral=args.use_neutral,
        delta_only=args.delta_only, nw=args.no_whitening, seed=args.random_seed
    )
    experiment_name = f"{experiment_name_suffix}"
    if args.method == "DAMA":
        print(f"Evaluating DAMA model {experiment_name}")
        output_dir = os.path.join(RESULTS_DIR, args.method, model_name, experiment_name)
        if args.multilingual_training:
            output_dir += "_multilingual"
        hparams = DAMAHyperParams.from_json(os.path.join(output_dir, "hparams.json"))
        projection_file = os.path.join(output_dir, "projections.npy")
        model = load_dama_model(model, hparams, projection_file)
    # elif args.method == "DAMA_L":
    #     print(f"Evaluating DAMA Leace model")
    #     output_dir = os.path.join(RESULTS_DIR, args.method, f"{model_name}_{str(args.num_layers)}L")
    #     if args.multilingual_training:
    #         output_dir += "_multilingual"

    #     hparams = DAMALeaceHyperParams.from_json(os.path.join(output_dir, "hparams.json"))
    #     projection_file = os.path.join(output_dir, "projections.npy")
    #     model = load_dama_model(model, hparams, projection_file)

    elif args.method == "MEMIT":
        print(f"Evaluating MEMIT model")
        output_dir = os.path.join(RESULTS_DIR, args.method, f"{model_name}_{str(args.num_layers)}L")
        hparams = MEMITHyperParams.from_json(os.path.join(output_dir, "hparams.json"))
        model = AutoModelForCausalLM.from_pretrained(output_dir, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                                      low_cpu_mem_usage=True, device_map='auto')
    elif args.method == "ALPHA_EDIT":
        print(f"Evaluating ALPHA_EDIT model")
        output_dir = os.path.join(RESULTS_DIR, args.method, f"{model_name}_{str(args.num_layers)}L")
        hparams =AlphaEditHyperParams.from_json(os.path.join(output_dir, "hparams.json"))
        model = AutoModelForCausalLM.from_pretrained(output_dir, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                                      low_cpu_mem_usage=True, device_map='auto')
    elif args.method == "FT":
        print(f"Evaluating fine-tuned model")
        output_dir = os.path.join(RESULTS_DIR, args.method, model_name)
        hparams = FTHyperParams.from_json(os.path.join(output_dir, "hparams.json"))
        model = AutoModelForCausalLM.from_pretrained(output_dir, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                                      low_cpu_mem_usage=True, device_map='auto')
    elif args.method == "PEFT":
        print(f"Evaluating PEFT model")
        revision = "v6"
        output_dir = os.path.join(RESULTS_DIR, args.method, model_name, revision)
        hparams = None
        # Model saved in HF hub by PaulM2000
        model = AutoModelForCausalLM.from_pretrained("PaulM2000/merged_peft_model_random_42_without_up_proj_llama-7b",
                                                     revision=revision, offload_folder=output_dir,
                                                     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                                     low_cpu_mem_usage=True, device_map='auto')
    elif args.method == "PLAIN_GPT2_MEDIUM":
        print(f"Evaluating plain GPT2-MEDIUM model")
        output_dir = os.path.join(RESULTS_DIR, args.method, model_name)
        hparams = None
        model = AutoModelForCausalLM.from_pretrained("gpt2-medium", torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                                      low_cpu_mem_usage=True, device_map='auto')
    elif args.method == None:
        print(f"Evaluating original model {model_name}")
        output_dir = os.path.join(RESULTS_DIR, "original",model_name)
        os.makedirs(output_dir, exist_ok=True)
    else:
        raise ValueError(f"Unknown method {args.method}")

    run_evaluation_on_task(model, tok, model_name, args.test_task, args.test_file, output_dir)

