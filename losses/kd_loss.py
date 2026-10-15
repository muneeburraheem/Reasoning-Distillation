import torch
import torch.nn.functional as F


def kl_div_loss(
    student_logits,
    teacher_logits,
    attention_mask,
    temperature=4.0,
    chunk_size=0,
):
    """
    Token-wise KL divergence for knowledge distillation.

    Args:
        student_logits: [B, T, V]
        teacher_logits: [B, T, V]
        attention_mask: [B, T] (1 for valid tokens, 0 for padding)
        temperature: Distillation temperature.
        chunk_size: If > 0, compute the sum over sequence positions in slices
            of this many tokens instead of materialising the full [B, T, V]
            intermediates at once. Mathematically identical - the loss is
            already a sum over positions - but it frees each slice's softmax
            and elementwise-KL tensors before the next slice is built. With a
            248,320-token vocabulary each [1, 2048, V] tensor is 0.95 GiB, and
            the unchunked path holds five of them simultaneously (~4.7 GiB).
            0 keeps the original single-shot path.

    Returns:
        Mean KL divergence over valid tokens.
    """
    mask = attention_mask.float()
    denom = mask.sum().clamp(min=1.0)

    if chunk_size and chunk_size > 0:
        total = student_logits.new_zeros(())
        for start in range(0, student_logits.shape[1], chunk_size):
            stop = start + chunk_size
            s_chunk = F.log_softmax(student_logits[:, start:stop] / temperature, dim=-1)
            t_chunk = F.softmax(teacher_logits[:, start:stop] / temperature, dim=-1)
            kl_chunk = F.kl_div(s_chunk, t_chunk, reduction="none").sum(dim=-1)
            total = total + (kl_chunk * mask[:, start:stop]).sum()
            del s_chunk, t_chunk, kl_chunk
        return (total / denom) * (temperature ** 2)

    # Temperature-scaled distributions
    s_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    t_probs = F.softmax(teacher_logits / temperature, dim=-1)

    # Token-wise KL divergence
    # Shape: [B, T, V] -> [B, T]
    kl = F.kl_div(
        s_log_probs,
        t_probs,
        reduction="none",
    ).sum(dim=-1)

    # Mask out padding tokens
    kl = (kl * mask).sum() / denom

    # Standard temperature correction
    return kl * (temperature ** 2)