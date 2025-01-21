import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union

def preference_loss(
    chosen_log_probs: torch.FloatTensor, 
    rejected_log_probs: torch.FloatTensor,
    reference_chosen_log_probs: torch.FloatTensor,
    reference_rejected_log_probs: torch.FloatTensor,
    beta: float = 0.1, # suggested value in the DPO paper
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """
    Args:
        chosen_log_probs: log probabilities of the policy model for the chosen responses, shape: (batch_size,)
        
    """
    
    pi_logratios = chosen_log_probs - rejected_log_probs
    ref_logratios = reference_chosen_log_probs - reference_rejected_log_probs
    
    logratios_difference = pi_logratios - ref_logratios
    
    losses = - F.logsigmoid(beta * logratios_difference)
        
    chosen_rewards = beta * (chosen_log_probs - reference_chosen_log_probs).detach()
    rejected_rewards = beta * (rejected_log_probs - reference_rejected_log_probs).detach()
    
    return losses, chosen_rewards, rejected_rewards

def _get_sequence_log_probs(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    loss_mask: torch.LongTensor,
    average_log_prob: bool = False,
    ) -> torch.FloatTensor:
    
    """
    Compute the log probability of a text sequence given logits
    Args:
        logits: logits of the model output. Shape: (batch_size, seq_length, vocab_size)
        labels: labels for which token's log probability; label = -100 indicates ignore. Shape (batch_size, seq_length)

    """

    assert logits.shape[:-1] == labels.shape
    #assert labels.shape == loss_mask.shape
    # let the sequence be A, B, C, D
    # labels[:,1：] are B, C, D
    # logits corresponds to B, C, D, X
    # logits[:,:-1,:] corresponds to B, C, D
    labels = labels[:,1:].clone() # labels 
    logits = logits[:,:-1,:]
    loss_mask = loss_mask[:, 1:]
    
    
    # log_probs shape (batch_size, seq_len - 1, vocab_size)
    # label shape before unsqueeze - (batch_size, seq_len - 1), after - (batch_size, seq_len - 1, vocab_size)
    log_probs = logits.log_softmax(-1)
    # per_token_logps shape (batch_size, seq_len - 1)
    per_token_logps = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(2) # squeeze on the last dim
    
    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)
        