import numpy as np

def wasserstein_critic_loss(real_scores, fake_scores):
    """
    Compute Wasserstein Critic Loss for WGAN.

    Parameters:
        real_scores : array-like
            Critic outputs for real samples
        fake_scores : array-like
            Critic outputs for fake samples

    Returns:
        float
    """

    real_scores = np.asarray(real_scores, dtype=float)
    fake_scores = np.asarray(fake_scores, dtype=float)

    mean_real = np.mean(real_scores)
    mean_fake = np.mean(fake_scores)

    return float(mean_fake - mean_real)