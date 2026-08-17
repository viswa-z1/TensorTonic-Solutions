import math

def cosine_annealing_schedule(base_lr, min_lr, total_steps, current_step):
    """
    Compute the learning rate using cosine annealing.
    
    Args:
        base_lr (float): Initial (maximum) learning rate.
        min_lr (float): Minimum learning rate.
        total_steps (int): Total number of steps.
        current_step (int): Current step number.
    
    Returns:
        float: Learning rate at current_step.
    """
    cosine_term = math.cos(math.pi * current_step / total_steps)
    lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cosine_term)
    return lr
