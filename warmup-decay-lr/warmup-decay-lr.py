def warmup_decay_schedule(base_lr, warmup_steps, total_steps, current_step):
    """
    Compute the learning rate at a given step using warmup + linear decay.
    
    Args:
        base_lr (float): Base learning rate.
        warmup_steps (int): Number of warmup steps.
        total_steps (int): Total number of steps.
        current_step (int): Current step number.
    
    Returns:
        float: Learning rate at current_step.
    """
    if current_step < warmup_steps and warmup_steps > 0:
        # Warmup phase: linear increase from 0 to base_lr
        lr = base_lr * (current_step / warmup_steps)
    else:
        # Decay phase: linear decrease from base_lr to 0
        decay_steps = total_steps - warmup_steps
        # Avoid division by zero if decay_steps == 0 (should not happen per constraints)
        if decay_steps == 0:
            lr = 0.0
        else:
            lr = base_lr * (total_steps - current_step) / decay_steps
        # Clamp lr to zero if current_step > total_steps (optional safety)
        if lr < 0:
            lr = 0.0
    return lr
