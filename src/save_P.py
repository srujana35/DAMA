import json
import torch
import os
from transformers import AutoModelForCausalLM
from AlphaEdit2.alphaedit2_hparams import AlphaEdit2HyperParams
import numpy as np


# === Edit these paths as needed ===
model_name = "HuggingFaceM4/tiny-random-LlamaForCausalLM"
hparams_path = "/Users/jeevana/Documents/GitHub/DAMA/hparams/ALPHA_EDIT2/llama_tiny.json"
save_dir = "projections/"
proj_dim = 48  # Or 1024

# === Load model and hyperparams ===
model = AutoModelForCausalLM.from_pretrained(model_name)
hparams = AlphaEdit2HyperParams.from_json(hparams_path)


def compute_P_projection(weight: torch.Tensor, proj_dim: int) -> torch.Tensor:
    """
    Compute projection matrix P ∈ ℝ^{d×d} (or lower) as V_r @ V_r.T
    where V_r are the top 'proj_dim' right singular vectors from SVD.
    """
    print(f"Running SVD on weight shape: {weight.shape}")
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
    V_r = Vh[:proj_dim, :]  # shape: [proj_dim, d]
    P = V_r.T @ V_r         # shape: [d, d]
    print(f"Projection matrix P shape: {P.shape}")
    return P

def extract_and_save_P(model, hparams, save_dir: str, proj_dim: int = 64):
    os.makedirs(save_dir, exist_ok=True)
    for i, layer in enumerate(hparams.layers):
        layer_name = hparams.rewrite_module_tmp.format(layer)
        weight_name = f"{layer_name}.weight"
        print(f"Extracting from {weight_name}")
        
        weight = dict(model.named_parameters())[weight_name].detach().cpu()
        P = compute_P_projection(weight, proj_dim)

        save_path = os.path.join(save_dir, f"P_{i}.npy")
        np.save(save_path, P.numpy())
        print(f"Saved P[{i}] to {save_path}")

# === Run projection extraction and save ===
extract_and_save_P(model, hparams, save_dir=save_dir, proj_dim=proj_dim)
