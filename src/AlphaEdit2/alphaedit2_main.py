import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import nethook
from utils.repr_tools import get_module_input_output_at_words
from .compute_ks import compute_ks
from .compute_z import compute_z
from .alphaedit2_hparams import AlphaEdit2HyperParams

def apply_alphaedit2_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: AlphaEdit2HyperParams,
    P: Dict[int, torch.Tensor],  # Projection matrices by layer index
    copy: bool = False,
    return_orig_weights: bool = False,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    if copy:
        model = deepcopy(model)

    weights_copy = {}
    device = next(model.parameters()).device
    context_templates = [["{}"]]
    req_contexts = [templ.format(r["prompt"]) for templ in context_templates[0] for r in requests]
    z_layer = hparams.v_loss_layer

    # Compute target representations
    z_list = [compute_z(model, tok, r, hparams, z_layer, context_templates) for r in requests]
    zs = torch.stack(z_list, dim=1)

    for i, layer in enumerate(hparams.layers):
        print(f"\nLAYER {layer}")
        layer_name = hparams.rewrite_module_tmp.format(layer)
        weight_name = f"{layer_name}.weight"
        w = nethook.get_parameter(model, weight_name)

        if return_orig_weights and weight_name not in weights_copy:
            weights_copy[weight_name] = w.detach().clone()

        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T.to(device)

        cur_zs = get_module_input_output_at_words(
            model, tok,
            contexts=req_contexts,
            words=[r["subject"] for r in requests],
            layer=z_layer,
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token
        )[1].T.to(device)

        targets = zs.to(device) - cur_zs
        repeat_factor = layer_ks.size(1) // targets.size(1)
        targets = targets.repeat_interleave(repeat_factor, dim=1)
        resid = targets / (len(hparams.layers) - i)

        # Project the residual
        P_i = P[i].to(device)  # projection matrix ∈ ℝ^{d×d}
        resid_proj = P_i @ resid
        upd_matrix = resid_proj @ layer_ks.T

        if upd_matrix.shape != w.shape:
            if upd_matrix.T.shape == w.shape:
                upd_matrix = upd_matrix.T
            else:
                raise ValueError("Shape mismatch in update matrix")

        with torch.no_grad():
            w[...] = weights_copy[weight_name] + upd_matrix.float()

        print(f"Update norm: {torch.linalg.norm(upd_matrix):.4f}")

    return model, weights_copy if return_orig_weights else {}
