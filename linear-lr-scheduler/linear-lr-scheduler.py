def linear_lr(step,
              total_steps,
              initial_lr,
              final_lr=0.0,
              warmup_steps=0) -> float:
    """
    Linear warmup followed by linear decay.

    Parameters:
        step          : current step (0-based)
        total_steps   : total training steps
        initial_lr    : peak learning rate
        final_lr      : ending learning rate
        warmup_steps  : number of warmup steps

    Returns:
        float learning rate
    """

    # After training ends
    if step > total_steps:
        return float(final_lr)

    # Warmup phase
    if warmup_steps > 0 and step < warmup_steps:
        lr = (step * initial_lr) / warmup_steps
        return float(lr)

    # No decay region (edge case)
    if total_steps == warmup_steps:
        return float(final_lr)

    # Linear decay phase
    decay_progress = (total_steps - step) / (total_steps - warmup_steps)

    lr = final_lr + (initial_lr - final_lr) * decay_progress

    return float(lr)