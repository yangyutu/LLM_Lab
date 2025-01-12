from typing import List, Dict, Any
from torch import Tensor
import torch
import collections


def default_data_collator(
    batch: List[Dict[str, Any]], 
    collate_keys={"input_ids","attention_mask","label"}
    )-> Dict[str, Any]:
    result_dict = collections.defaultdict(list)
    first = batch[0]
    for example in batch:
        for k, v in example.items():
            if k in collate_keys:
                result_dict[k].append(v)

    for k, v in first.items():
        if v is not None and not isinstance(v, str):
            #dtype = torch.long if isinstance(v, int) else v.dtype
            if isinstance(v, torch.Tensor):
                result_dict[k] = torch.stack(result_dict[k])
            else:
                result_dict[k] = torch.tensor(result_dict[k])

    return result_dict

def default_data_collator_with_padding(
    batch: List[Dict[str, Any]], 
    pad_token_id: int = None,
    collate_keys = {"input_ids","attention_mask","label"},
    padding_keys = {"input_ids","attention_mask"},
    padding_strategy: str = "longest",
    pad_to_multiple_of: int = None,
    padding_side: str = "right",
    max_length = None,
    )-> Dict[str, Any]:
    
    assert padding_strategy in {'longest', 'max_length'}
    assert padding_side in {"left", "right"}
    if "input_ids" in padding_keys:
        assert pad_token_id is not None
    
    for padding_key in padding_keys:
        assert padding_key in {"attention_mask", "input_ids"}
    
    
    first_example = batch[0]
    first_key = list(padding_keys)[0]
    
    if padding_strategy == 'longest':
        max_length = max([len(example[first_key]) for example in batch])
        
    if max_length is not None and pad_to_multiple_of is not None:
        max_length = (max_length // pad_to_multiple_of + 1) * pad_to_multiple_of
        
    for example in batch:
        
        for k, v in first_example.items():
            if k not in padding_keys:
                continue
            
            if k == "attention_mask":
                padding_token = 0
            elif k == "input_ids":
                padding_token = pad_token_id
            
            difference = max_length - len(example[k])
            
            if padding_side == "left":
                example[k] = [padding_token] * difference + example[k]
            elif padding_side == "right":
                example[k] = example[k] + [padding_token] * difference
    
    result_dict = default_data_collator(batch=batch, collate_keys=collate_keys)
    
    return result_dict