import numpy as np

def total_parameters(obs_dim: int, hidden_size: int, act_dim: int) -> int:
    """Calcola i parametri per una rete a 2 strati nascosti (entrambi di 'hidden_size')."""
    h1_size = hidden_size
    h2_size = hidden_size

    # Pesi
    w1 = obs_dim * h1_size
    w2 = h1_size * h2_size
    w3 = h2_size * act_dim

    # Bias
    b1 = h1_size
    b2 = h2_size
    b3 = act_dim

    return w1 + w2 + w3 + b1 + b2 + b3


def get_action(
    model: np.ndarray,
    observation: np.ndarray,
    obs_dim: int,
    hidden_size: int,
    act_dim: int,
) -> np.ndarray:
    """
    Calcola l'azione a partire dai parametri flat della rete:
     - input -> hidden1 (ReLU)
     - hidden1 -> hidden2 (ReLU)
     - hidden2 -> output (Tanh per restare in [-1, 1])
    Entrambi gli strati nascosti hanno 'hidden_size' neuroni.
    """
    expected = total_parameters(obs_dim, hidden_size, act_dim)
    if model.shape[0] != expected:
        raise ValueError(f"Model size {model.shape[0]} != expected {expected}")

    h1_size = hidden_size
    h2_size = hidden_size

    # Calcolo degli indici di slicing
    w1_end = obs_dim * h1_size
    w2_end = w1_end + (h1_size * h2_size)
    w3_end = w2_end + (h2_size * act_dim)
    
    b1_end = w3_end + h1_size
    b2_end = b1_end + h2_size
    # b3 va fino alla fine

    # Estrazione Pesi
    w1 = model[:w1_end].reshape(obs_dim, h1_size)
    w2 = model[w1_end:w2_end].reshape(h1_size, h2_size)
    w3 = model[w2_end:w3_end].reshape(h2_size, act_dim)
    
    # Estrazione Bias
    b1 = model[w3_end:b1_end].reshape(1, h1_size)
    b2 = model[b1_end:b2_end].reshape(1, h2_size)
    b3 = model[b2_end:].reshape(1, act_dim)


    if observation.ndim == 1:
        observation = observation.reshape(1, -1)

    # === Forward Pass ===
    
    # Strato 1 con attivazione ReLU
    h1 = np.maximum(0, observation @ w1 + b1)
    
    # Strato 2 con attivazione ReLU
    h2 = np.maximum(0, h1 @ w2 + b2)
    
    # Strato di output con attivazione Tanh
    outputs = np.tanh(h2 @ w3 + b3)
    
    return outputs if outputs.shape[0] > 1 else outputs.flatten()