import torch
import functools
from typing import Callable
from contextlib import nullcontext
import torch.autograd.forward_ad as fw

def autograd_context(func: Callable):
    @functools.wraps(func)    
    def wrapper(*args, **kwargs):
        is_forward_ad   = kwargs.pop("is_forward_ad", False)
        is_grad_enabled = kwargs.pop("is_grad_enabled", False)

        with torch.set_grad_enabled(is_grad_enabled):
            with fw.dual_level() if is_forward_ad else nullcontext():
                return func(*args, **kwargs)
            
    return wrapper