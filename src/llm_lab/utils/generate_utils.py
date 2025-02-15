from typing import List
import torch


# follow implementation in https://github.com/hkproj/pytorch-llama/blob/main/inference.py
def generate(model, 
             tokenizer,
             prompts: List[str],
             device, 
             greedy_decoding: bool=False,
             temperature: float=0.8,
             top_p: float=0.9,
             max_new_tokens: int=100):
    
    
    prompt_tokens = [tokenizer.encode(prompt) for prompt in prompts]
    
    max_prompt_len = max([len(prompt_token) for prompt_token in prompt_tokens])
    
    batch_size = len(prompt_tokens)
    
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_token_id = tokenizer.eos_token_id
    
    total_len = max_prompt_len + max_new_tokens
    
    # every token is defaulted to pad_token_id
    tokens = torch.full((batch_size, total_len), pad_token_id, dtype=torch.long, device=device)
    prompt_pad_mask = tokens == pad_token_id # True if the token is a prompt token, False otherwise
    for k, t in enumerate(prompt_tokens):
        # fill in existing prompt tokens
        tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)
        
    eos_reached = torch.tensor([False] * batch_size, device=device)
    
    for cur_pos in range(total_len - 1):
        
        with torch.no_grad():
            logits = model(tokens[:,cur_pos:cur_pos + 1], use_cache=True, start_pos=cur_pos)

            if greedy_decoding:
                next_token = torch.argmax(logits[:,-1,:], dim=-1)
            else:
                raise NotImplementedError()
        
        # only replace toekn if it is a padding token
        next_token = torch.where(prompt_pad_mask[:, cur_pos+1], next_token, tokens[:,cur_pos + 1])
        
        tokens[:, cur_pos + 1] = next_token
        
        # EOS is reachehed only if we found an EOS token for a padding position
        
        eos_reached |= (prompt_pad_mask[:, cur_pos + 1]) & (next_token == eos_token_id)
        
        if all(eos_reached):
            break
        
    out_tokens = []
    out_text = []
    
    for current_prompt_tokens in tokens.tolist():
        # cut to the EOS token if present
        if eos_token_id in current_prompt_tokens:
            eos_idx = current_prompt_tokens.index(eos_token_id)
            current_prompt_tokens = current_prompt_tokens[:eos_idx]
        
        out_tokens.append(current_prompt_tokens)
        out_text.append(tokenizer.decode(current_prompt_tokens))
    
    return (out_tokens, out_text)
    