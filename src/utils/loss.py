import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss


def fast_lm_loss(
    hidden_states: torch.FloatTensor,
    lm_head: torch.nn.Linear,
    labels: torch.LongTensor,
    ignore_index: int = -100,
    shift: bool = True
):
    
    # shift if needed
    if shift:
        hidden_states, labels = shift_tokens(hidden_states, labels)

    # reshape to remove batch dimension
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    labels = labels.view(-1)

    # calculate the logits and mask
    logits: torch.FloatTensor = lm_head(hidden_states)
    mask = labels != ignore_index
    mask_sum = mask.float().sum()

    # calculate the loss
    loss = F.cross_entropy(
        logits, labels,
        ignore_index=ignore_index,
    )

    # calculate the accuracy
    correct = (logits.argmax(dim=-1) == labels).float()
    acc = correct.masked_fill(~mask, 0.0).sum() / (mask_sum + 1)

    # calculate the pcorr
    logp = -F.cross_entropy(
        logits, labels,
        reduction='none',
    )
    p = logp.exp()
    pcorr = p.masked_fill(~mask, 0.0).sum() / (mask_sum + 1)

    return {
        "loss": loss,
        "acc": acc,
        "pcorr": pcorr
    }



def shift_tokens(logits, labels):
    return logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()


def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100, shifted=False) -> torch.Tensor:
    """
    Computes cross entropy loss of `logits` against the ground truth `labels` during
    next token prediction.

    Useful as the loss function of a LLM in pretraining or supervised finetuning.
    """
    # Shift so that tokens < n predict n
    if shifted:
        shift_logits, shift_labels = logits, labels
    else:
        shift_logits, shift_labels = shift_tokens(logits, labels)
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(ignore_index=ignore_index)
    shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
    shift_labels = shift_labels.view(-1)
    shift_labels = shift_labels.to(shift_logits.device)
    return loss_fct(shift_logits, shift_labels)


def accuracy(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100, shifted=False) -> float:
    """
    Computes the accuracy of the model's predictions against the ground truth labels.
    
    Args:
        logits (torch.Tensor): The model's output logits.
        labels (torch.Tensor): The ground truth labels.
        ignore_index (int): The index to ignore in the accuracy calculation.
    
    Returns:
        float: The accuracy as a percentage.
    """
    if not shifted:
        logits, labels = shift_tokens(logits, labels)
    mask = labels != ignore_index

    correct = (logits.argmax(dim=-1) == labels).float()
    correct = correct.masked_fill(~mask, 0.0).sum()

    total = mask.float().sum() + 1

    return correct / total    


def pcorr(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100, shifted=False) -> float:
    if not shifted:
        logits, labels = shift_tokens(logits, labels)
    mask = labels != ignore_index

    logp = -F.cross_entropy(
        logits.view(-1, logits.shape[-1]),
        labels.view(-1),
        reduction='none',
    )
    
    p = logp.exp()
    p = p.masked_fill(~mask.view(-1), 0.0).sum()

    total = mask.float().sum() + 1

    return p / total   
