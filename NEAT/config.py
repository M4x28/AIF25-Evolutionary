# config.py
from dataclasses import dataclass

@dataclass # @dataclass è un "decoratore". Python scriverà automaticamente __init__ (il costruttore),  __repr__ (per stamparla), ecc.
class TrainConfig: # Contenitore centralizzato per tutti i parametri di configurazione dell'esperimento. 
    # --- Parametri dell'Ambiente ---
    # L'identificatore dell'ambiente Gymnasium
    env_id: str = "Walker2d-v5" 
    
    # Se 'True', l'osservazione non includerà la posizione x (e z) assoluta del root (tronco).
    exclude_current_positions_from_observation: bool = True
    
    # Il numero massimo di passi (timesteps) che un singolo episodio può durare prima di essere interrotto forzatamente (truncated).
    max_episode_steps: int = 10_000
    
    # Il numero di frame per secondo (FPS) da usare durante la creazione dei video di rendering.
    render_fps: int = 30
    
    # Un moltiplicatore nell'ambiente Walker2d che scala la ricompensa ottenuta per il movimento in avanti.
    forward_reward_weight: float = 2.0

    # --- Parametri di NEAT (Evoluzione) ---
    # La dimensione della popolazione NEAT (quanti genomi per generazione).
    pop_size: int = 150
    
    # Quante generazioni di NEAT vengono eseguite *per ogni fase* di rumore.
    max_generations_per_phase: int = 25
    
    # Il numero totale di fasi di training. Ad ogni fase, il rumore applicato alle osservazioni viene cambiato.
    phases: int = 12

    # --- Parametri del Rumore (Robustezza) ---
    # La deviazione standard del rumore Gaussiano (casuale) che verrà aggiunto alle osservazioni.
    noise_std: float = 0.00

    # --- Parametri di Output e Logging ---
    
    # Il percorso della cartella principale dove salvare tutti i risultati (checkpoint, video, grafici). 
    out_dir: str = "runs/runs_neat_2_walker2d" 
    
    # La durata massima (in secondi) dei video registrati per il miglior genoma.
    video_seconds: int = 15

    # --- Parametri di Riproducibilità ---
    
    # Il "seed" (seme) globale per la generazione di numeri casuali.
    seed: int = 0

    # --- Percorsi File ---
    
    # Il percorso del file di configurazione specifico per 
    neat_config_path: str = "configs/neat_walker2d.cfg"
