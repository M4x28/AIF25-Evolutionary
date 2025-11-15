"""Feed-forward neural policies used by CMA-ES."""
from __future__ import annotations

import numpy as np


def two_layer_parameter_count(obs_dim: int, hidden_size: int, act_dim: int) -> int:
    """Return the flattened parameter count for a 2-hidden-layer MLP."""
    h = hidden_size
    # weights + biases for input->h1, h1->h2, h2->out
    return (obs_dim * h) + (h * h) + (h * act_dim) + (h + h + act_dim)


def two_layer_action(
    params: np.ndarray,
    observation: np.ndarray,
    obs_dim: int,
    hidden_size: int,
    act_dim: int,
) -> np.ndarray:
    """Compute the policy action given flattened parameters."""
    expected = two_layer_parameter_count(obs_dim, hidden_size, act_dim)
    if params.shape[0] != expected:
        raise ValueError(f"Model size {params.shape[0]} != expected {expected}")

    h = hidden_size
    w1_end = obs_dim * h
    w2_end = w1_end + (h * h)
    w3_end = w2_end + (h * act_dim)
    b1_end = w3_end + h
    b2_end = b1_end + h

    w1 = params[:w1_end].reshape(obs_dim, h)
    w2 = params[w1_end:w2_end].reshape(h, h)
    w3 = params[w2_end:w3_end].reshape(h, act_dim)
    b1 = params[w3_end:b1_end].reshape(1, h)
    b2 = params[b1_end:b2_end].reshape(1, h)
    b3 = params[b2_end:].reshape(1, act_dim)

    obs = observation if observation.ndim > 1 else observation.reshape(1, -1)
    h1 = np.maximum(0, obs @ w1 + b1)
    h2 = np.maximum(0, h1 @ w2 + b2)
    out = np.tanh(h2 @ w3 + b3)
    return out.flatten() if out.shape[0] == 1 else out
