# src/utils/device.py

import torch
from typing import Tuple

def get_device_and_attention() -> Tuple[str, str]:
    """
    Detect available device and return appropriate attention implementation.
    
    Returns
    -------
    Tuple[str, str]
        (device, attention_implementation)
    """
    if torch.cuda.is_available():
        device = "cuda"
        attn_implementation = "flash_attention_2"
    elif torch.backends.mps.is_available():
        device = "mps"
        attn_implementation = "sdpa"
    else:
        device = "cpu"
        attn_implementation = "sdpa"
    
    return device, attn_implementation