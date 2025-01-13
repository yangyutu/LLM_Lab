import torch
from torch import Tensor

def move_to_device(batch, device):
    
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch