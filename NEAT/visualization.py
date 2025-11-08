# visualization.py
import gymnasium as gym
import neat
import numpy as np
from typing import Optional, List

try:
    import cv2
except Exception:
    cv2 = None # Video opzionale

# Importa la definizione della configurazione
from config import TrainConfig 

def record_video(
    cfg: TrainConfig,
    genome,
    neat_config: neat.Config,
    obs_dim: int,
    act_dim: int,
    noise_vec: Optional[np.ndarray],
    out_path: str,
) -> None:
    '''
    Spiegazione funzione:
    Esegue un episodio completo nell'ambiente Gymnasium utilizzando
    uno specifico genoma (rete neurale) e registra i fotogrammi (frames)
    risultanti in un file video .mp4.
    
    Questa funzione è progettata per visualizzare la performance
    del "miglior" genoma trovato durante l'addestramento.

    Spiegazione input:
        cfg (TrainConfig): L'oggetto di configurazione globale che contiene
                           parametri come 'env_id', 'render_fps',
                           'video_seconds' e 'max_episode_steps'.
        genome (neat.DefaultGenome): Il genoma NEAT (il "cervello")
                                     che si vuole valutare e registrare.
        neat_config (neat.Config): La configurazione NEAT necessaria per
                                   ricostruire la rete neurale dal genoma.
        obs_dim (int): La dimensione (numero) degli input della rete.
                       (Nota: non usata attivamente nel corpo di questa funzione).
        act_dim (int): La dimensione (numero) degli output della rete.
                       (Nota: non usata attivamente nel corpo di questa funzione).
        noise_vec (Optional[np.ndarray]): Un vettore di rumore costante
                                           da sommare alle osservazioni.
                                           Se 'None', non viene aggiunto rumore.
        out_path (str): Il percorso completo del file (es. "run/best.mp4")
                        dove il video verrà salvato.

    Spiegazione output:
        None: La funzione non restituisce alcun valore. Il suo effetto
              è la creazione di un file video sul disco.
    '''
    
    # Se la libreria cv2 (OpenCV) non è stata importata con successo
    # all'avvio dello script, la registrazione video è disabilitata.
    # La funzione termina immediatamente.
    if cv2 is None:
        print("Warning: cv2 (OpenCV) non trovato, impossibile registrare video.")
        return

    # 1. Creazione dell'ambiente
    # Crea un'istanza dell'ambiente Gymnasium.
    # Cruciale: 'render_mode="rgb_array"' dice a Gym di non aprire
    # una finestra, ma di restituire i frame come array NumPy.
    env = gym.make(
        cfg.env_id,
        render_mode="rgb_array", 
        exclude_current_positions_from_observation=cfg.exclude_current_positions_from_observation,
        forward_reward_weight=cfg.forward_reward_weight,
    )
    
    # Resetta l'ambiente usando il seed globale per la riproducibilità
    obs, _ = env.reset(seed=cfg.seed)
    
    # Ricostruisce la rete neurale (il "cervello") a partire dal genoma
    net = neat.nn.FeedForwardNetwork.create(genome, neat_config)

    # Lista per accumulare tutti i frame (immagini) catturati
    frames: List[np.ndarray] = []
    
    # Calcola il numero massimo di frame da registrare
    max_frames = cfg.video_seconds * cfg.render_fps
    steps = 0 # Contatore dei passi della simulazione

    # 2. Loop di Simulazione e Cattura
    # Continua finché l'episodio non supera i passi massimi
    # O finché non abbiamo raccolto abbastanza frame per il video
    while steps < cfg.max_episode_steps and len(frames) < max_frames:
        
        # Applica il rumore (se presente) all'osservazione
        noisy_obs = obs if noise_vec is None else (obs + noise_vec)
        
        # Chiede alla rete di decidere un'azione
        raw_action = net.activate(noisy_obs.tolist())
        
        # Applica 'tanh' per mappare l'output nell'intervallo [-1, 1]
        action = np.tanh(np.array(raw_action, dtype=np.float32))
        
        # Esegui l'azione nell'ambiente
        obs, _, terminated, truncated, _ = env.step(action)

        # Richiedi il rendering del frame corrente come array RGB
        rgb = env.render()
        
        # Aggiungi il frame alla lista (se il rendering è andato a buon fine)
        if rgb is not None:
            frames.append(rgb)
            
        # Se l'episodio è terminato (es. l'agente è caduto) o
        # è stato troncato (tempo scaduto), interrompi il loop
        if terminated or truncated:
            break
        steps += 1

    # Chiudi l'ambiente per liberare le risorse
    env.close()

    # Se per qualche motivo non sono stati catturati frame, esci
    if not frames:
        return

    # 3. Scrittura del File Video
    
    # Ottieni altezza (h) e larghezza (w) dal primo frame
    # (servono per inizializzare il video writer)
    h, w, _ = frames[0].shape
    
    # Definisce il codec video (in questo caso, 'mp4v' per un file .mp4)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    
    # Inizializza l'oggetto VideoWriter di OpenCV
    # Specifica: percorso file, codec, FPS, e dimensioni (w, h)
    vw = cv2.VideoWriter(out_path, fourcc, cfg.render_fps, (w, h))
    
    # Itera su ogni frame che abbiamo salvato nella lista
    for frame in frames:
        # Converte il frame da RGB (formato di Gym) a BGR
        # (formato che OpenCV si aspetta per la scrittura)
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Scrivi il frame convertito nel file video
        vw.write(bgr)
        
    # Finalizza e chiude il file video, salvandolo su disco
    vw.release()