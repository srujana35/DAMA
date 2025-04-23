# This is the AlphaEdit2-converted version of the MEMIT editing stack
# based on your files: memit_main.py, compute_z.py, compute_ks.py, memit_hparams.py

# Key changes made:
# - Renamed MEMIT functions and classes to AlphaEdit2
# - Removed covariance matrix solving logic (AlphaEdit doesn't solve a system)
# - Added projection-based update using precomputed P matrices
# - Converted MEMITHyperParams -> AlphaEdit2HyperParams

# The following Python modules should be created/updated accordingly:

# alphaedit2_hparams.py
from dataclasses import dataclass
from typing import List, Literal
from utils.hparams import HyperParams

@dataclass
class AlphaEdit2HyperParams(HyperParams):
    #Method
    layers: List[int]
    layer_selection: Literal["all", "random"]
    fact_token: Literal[
        "last", "subject_first", "subject_last", "subject_first_after_last"
    ]
    v_num_grad_steps: int
    v_lr: float
    v_loss_layer: int
    v_weight_decay: float
    clamp_norm_factor: float
    kl_factor: float
    mom2_adjustment: bool
    mom2_update_weight: float
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    # Statistics
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str

    # Update strategy
    update_strategy: Literal["random", "neutral", "opposite"]
