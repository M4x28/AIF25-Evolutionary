from __future__ import annotations

import json
import os
import random
import pathlib
import pickle
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import multiprocessing

import gymnasium as gym
import matplotlib.pyplot as plt
import neat
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None  # Video opzionale


# ======================================================================================
# Utility
# ======================================================================================

def ensure_dir(path: str) -> None:
    '''Assicura che una directory (cartella) esista.
    
    Se la directory specificata non esiste, questa funzione la crea.
    Crea anche tutte le directory "genitore" necessarie (es. se si chiede
    di creare "A/B/C" e "A" non esiste, crea sia "A" che "A/B").
    Non fa nulla se la directory esiste già.

    Input:
        path (str): Il percorso della directory da controllare/creare.

    Output:
        None: La funzione non restituisce alcun valore.
    '''
    # Chiama .mkdir() su questo oggetto.
    #   parents=True: dice a mkdir di creare anche le cartelle "genitore"
    #                 necessarie, se mancano.
    #   exist_ok=True: dice a mkdir di non sollevare un errore (crash)
    #                  se la cartella (o le cartelle) esistono già.
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> None:
    '''Imposta il "seed" (seme) global.

    Input:
        seed (int): Il numero intero da usare come seme.

    Output:
        None: La funzione non restituisce alcun valore.
    '''
    # Imposta il seed per NumPy
    np.random.seed(seed)
    
    # Imposta il seed per il modulo 'random' (usato da 'neat-python')
    random.seed(seed)

# Template di configurazione NEAT con commenti in italiano
"""fitness_criterion     = max           # Criterio per la fitness: 'max' (massimizzare), 'min' (minimizzare), o 'mean' (media).
fitness_threshold     = 9999999       # Soglia di fitness: l'evoluzione si ferma se un genoma raggiunge questo valore.
pop_size              = {pop_size}    # Dimensione della popolazione (numero di genomi). (Template)
reset_on_extinction   = False         # Se 'True', crea una nuova popolazione casuale se tutte le specie si estinguono.

[DefaultGenome]
# --- Impostazioni Funzioni di Attivazione ---
activation_default      = tanh          # Funzione di attivazione predefinita per i nuovi nodi (es. tanh, sigmoid, relu).
activation_mutate_rate  = 0.1           # Probabilità (0.0-1.0) che un nodo esistente muti la sua funzione di attivazione.
activation_options      = tanh          # Lista di funzioni di attivazione tra cui scegliere durante la mutazione (separate da spazio).

# --- Impostazioni Funzioni di Aggregazione (come un nodo somma i suoi input) ---
aggregation_default     = sum           # Come un nodo aggrega i suoi input (es. sum, mean, max).
aggregation_mutate_rate = 0.0           # Probabilità (0.0-1.0) che un nodo muti la sua funzione di aggregazione.
aggregation_options     = sum           # Lista di funzioni di aggregazione tra cui scegliere (separate da spazio).

# --- Impostazioni Bias (Offset) dei Nodi ---
bias_init_mean          = 0.0           # Media per l'inizializzazione dei bias (distribuzione normale).
bias_init_stdev         = 1.0           # Deviazione standard per l'inizializzazione dei bias.
bias_max_value          = 30.0          # Valore massimo assoluto che un bias può assumere.
bias_min_value          = -30.0         # Valore minimo assoluto che un bias può assumere.
bias_mutate_power       = 0.5           # "Potenza" della mutazione del bias (dev. std. del rumore aggiunto).
bias_mutate_rate        = 0.4           # Probabilità (0.0-1.0) che il bias di un nodo venga mutato (perturbato).
bias_replace_rate       = 0.1           # Probabilità (0.0-1.0) che il bias di un nodo venga sostituito con un valore casuale.

# --- Impostazioni di Compatibilità (per la Speciazione) ---
compatibility_disjoint_coefficient = 1.0 # Coefficiente (C1) per i geni "disgiunti" nel calcolo della distanza di compatibilità.
compatibility_weight_coefficient   = 0.5 # Coefficiente (C3) per la differenza media dei pesi nel calcolo della distanza.

# --- Impostazioni Mutazioni Connessioni ---
conn_add_prob           = 0.1           # Probabilità (0.0-1.0) di aggiungere una nuova connessione (mutazione strutturale).
conn_delete_prob        = 0.1           # Probabilità (0.0-1.0) di eliminare una connessione esistente (mutazione strutturale).
enabled_default         = True          # Se 'True', le nuove connessioni sono abilitate di default.
enabled_mutate_rate     = 0.01          # Probabilità (0.0-1.0) di abilitare/disabilitare una connessione esistente.

# --- Impostazioni Generali Genoma ---
feed_forward            = True          # Se 'True', la rete è solo feed-forward (no connessioni ricorsive).
initial_connection      = full_direct   # Connessioni iniziali: 'full_direct' (tutti input -> tutti output), 'full_nodirect', 'partial_direct', etc.
node_add_prob           = 0.1           # Probabilità (0.0-1.0) di aggiungere un nuovo nodo (mutazione strutturale).
node_delete_prob        = 0.1           # Probabilità (0.0-1.0) di eliminare un nodo (mutazione strutturale).
num_hidden              = 0             # Numero di nodi nascosti *iniziali* (NEAT li aggiungerà evolutivamente).
num_inputs              = {num_inputs}  # Numero di neuroni di input (dimensione osservazione). (Template)
num_outputs             = {num_outputs} # Numero di neuroni di output (dimensione azione). (Template)

# --- Impostazioni "Response" (moltiplicatore del nodo, spesso non usato) ---
response_init_mean      = 1.0           # Media per inizializzazione "response" (raramente usato).
response_init_stdev     = 0.0           # Dev. std. per inizializzazione "response".
response_max_value      = 30.0          # Valore max "response".
response_min_value      = -30.0         # Valore min "response".
response_mutate_power   = 0.0           # Potenza mutazione "response".
response_mutate_rate    = 0.0           # Probabilità mutazione "response".
response_replace_rate   = 0.0           # Probabilità sostituzione "response".

# --- Impostazioni Pesi (Weights) delle Connessioni ---
weight_init_mean        = 0.0           # Media per l'inizializzazione dei pesi delle connessioni.
weight_init_stdev       = 1.0           # Deviazione standard per l'inizializzazione dei pesi.
weight_max_value        = 30.0          # Valore massimo assoluto che un peso può assumere.
weight_min_value        = -30.0         # Valore minimo assoluto che un peso può assumere.
weight_mutate_power     = 0.5           # "Potenza" della mutazione del peso (dev. std. del rumore aggiunto).
weight_mutate_rate      = 0.4           # Probabilità (0.0-1.0) che il peso di una connessione venga mutato (perturbato).
weight_replace_rate     = 0.1           # Probabilità (0.0-1.0) che il peso di una connessione venga sostituito con un valore casuale.

[DefaultSpeciesSet]
compatibility_threshold = 3.5           # Soglia di distanza per la speciazione: genomi con distanza < soglia sono della stessa specie.

[DefaultStagnation]
species_fitness_func = max           # Funzione per calcolare la fitness di un'intera specie (es. max, mean).
max_stagnation       = 10          # Numero di generazioni senza miglioramenti prima che una specie sia considerata "stagnante".
species_elitism      = 1           # Numero delle N specie migliori che sono protette dalla stagnazione (non vengono eliminate).

[DefaultReproduction]
elitism              = 2           # Numero di genomi "elite" (migliori) di ogni specie che passano alla generazione successiva senza mutazioni.
survival_threshold   = 0.3         # Frazione (0.0-1.0) dei genomi peggiori di ogni specie che vengono eliminati (non si riproducono).
"""

DEFAULT_NEAT_CONFIG_TEMPLATE = """[NEAT]
fitness_criterion     = max
fitness_threshold     = 9999999
pop_size              = {pop_size}
reset_on_extinction   = False

[DefaultGenome]
activation_default      = tanh
activation_mutate_rate  = 0.0
activation_options      = tanh 
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.4
bias_replace_rate       = 0.1
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient = 0.5
conn_add_prob           = 0.1
conn_delete_prob        = 0.1
enabled_default         = True
enabled_mutate_rate     = 0.01
feed_forward            = True
initial_connection      = full_direct
node_add_prob           = 0.1
node_delete_prob        = 0.1
num_hidden              = 0
num_inputs              = {num_inputs}
num_outputs             = {num_outputs}
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30.0
weight_min_value        = -30.0
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.4
weight_replace_rate     = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.5

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 10
species_elitism      = 1

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.3
"""

def ensure_neat_config(path: pathlib.Path, pop_size: int, num_inputs: int, num_outputs: int) -> None:
    '''Crea un file di configurazione NEAT se non esiste già.

    Controlla se il file specificato in 'path' esiste. Se esiste,
    non fa nulla. Se non esiste, usa il template 
    DEFAULT_NEAT_CONFIG_TEMPLATE per creare un nuovo file di configurazione,
    riempiendo i segnaposto con i parametri forniti.

    Input:
        path (pathlib.Path): L'oggetto Path che punta al file di 
                             configurazione da creare (es. "configs/neat.cfg").
        pop_size (int): La dimensione della popolazione (numero di agenti).
        num_inputs (int): Il numero di neuroni di input (dimensione 
                          dello spazio di osservazione).
        num_outputs (int): Il numero di neuroni di output (dimensione 
                           dello spazio di azione).

    Output:
        None: La funzione non restituisce nulla, ma crea un file su disco.
    '''
    # Controlla se il file specificato da 'path' esiste già sul disco.
    if path.exists():
        # Se esiste, esci immediatamente dalla funzione.
        return
    
    # Se il file non esiste:
    # 1. Assicurati che la cartella che dovrebbe contenere il file
    #    esista (es. "configs/"). 'path.parent' si riferisce alla
    #    directory contenitore. Usiamo la funzione definita prima.
    ensure_dir(str(path.parent))
    
    # 2. Prendi la stringa template e usa il metodo .format()
    #    per sostituire i segnaposto (es. {pop_size}) con i valori
    #    passati come argomenti alla funzione.
    config_text = DEFAULT_NEAT_CONFIG_TEMPLATE.format(
        pop_size=pop_size, num_inputs=num_inputs, num_outputs=num_outputs
    )

    # 3. Scrivi la stringa formattata (il testo completo della 
    #    configurazione) nel file specificato da 'path'.
    path.write_text(config_text)

# ======================================================================================
# Configurazione
# ======================================================================================
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

def eval_single_genome(params: Tuple) -> float:
    """
    Funzione eseguita da un processo worker.
    Valuta un singolo genoma in un ambiente dedicato.
    """
    # 1. Spacchetta i parametri (necessario per pool.map)
    genome, neat_config, cfg, noise_vec, eval_seed = params

    # 2. Crea un ambiente *in questo processo*
    #    Ogni worker DEVE avere il proprio ambiente.
    env = gym.make(
        cfg.env_id,
        exclude_current_positions_from_observation=cfg.exclude_current_positions_from_observation,
        forward_reward_weight=cfg.forward_reward_weight,
    )
    
    # 3. Crea la rete neurale
    net = neat.nn.FeedForwardNetwork.create(genome, neat_config)

    # 4. Resetta l'ambiente (tutti i worker usano lo stesso seed)
    obs, _ = env.reset(seed=eval_seed)
    
    cum_reward = 0.0
    steps = 0

    # 5. Esegui la simulazione (come nel tuo vecchio codice)
    while steps < cfg.max_episode_steps:
        noisy_obs = obs if noise_vec is None else (obs + noise_vec)
        action = np.tanh(np.array(net.activate(noisy_obs.tolist()), dtype=np.float32))
        obs, reward, terminated, truncated, _ = env.step(action)
        
        cum_reward += float(reward)
        steps += 1
        
        if terminated or truncated:
            break
    
    # 6. Chiudi l'ambiente del worker
    env.close()
    
    # 7. Restituisci la fitness calcolata
    return cum_reward
# ======================================================================================
# NEAT Evaluation Helpers
# ======================================================================================

class WalkerEvaluator:
    """Valuta un genoma NEAT su Walker2d con rumore costante per fase."""

    def __init__(self, cfg: TrainConfig, obs_dim: int, act_dim: int) -> None:
        #
        # Spiegazione funzione (Costruttore):
        #   Inizializza il valutatore. Crea e memorizza un'istanza 
        #   condivisa dell'ambiente Gymnasium che verrà usata 
        #   per testare tutti i genomi.
        #
        # Input:
        #   cfg (TrainConfig): La configurazione globale dell'esperimento.
        #   obs_dim (int): Dimensione dello spazio di osservazione.
        #   act_dim (int): Dimensione dello spazio di azione.
        #
        # Output: None
        #
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.seed_addition = 0

        # Crea l'ambiente (verrà riusato da tutti i genomi)
        self.env = gym.make(
            cfg.env_id,
            exclude_current_positions_from_observation=cfg.exclude_current_positions_from_observation,
            forward_reward_weight=cfg.forward_reward_weight,
        )

    def close(self) -> None:
        #
        # Spiegazione funzione:
        #   Chiude correttamente l'ambiente Gymnasium e rilascia le risorse.
        #
        # Input: self
        # Output: None
        #
        try:
            # Tenta di chiudere l'ambiente
            self.env.close()
        except Exception:
            # Ignora errori (es. se già chiuso)
            print("Warning: failed to close the environment cleanly.")
            pass

    def evaluate_genomes(
        self,
        genomes: List[Tuple[int, neat.DefaultGenome]],
        neat_config: neat.Config,
        noise_vec: Optional[np.ndarray],
    ) -> None:
        #
        # Spiegazione funzione:
        #   Calcola la fitness per un'intera popolazione (lista di genomi).
        #   Esegue un episodio completo per ciascun genoma e assegna 
        #   la ricompensa totale accumulata come `genome.fitness`.
        #
        # Input:
        #   genomes: Lista di (genome_id, genome_object) da valutare.
        #   neat_config: Configurazione NEAT per creare le reti neurali.
        #   noise_vec: (Opcionale) Vettore di rumore costante da 
        #              sommare alle osservazioni.
        #
        # Output:
        #   None. Modifica i genomi "in-place", aggiornando il loro 
        #   attributo .fitness.
        #
        
        # Itera su ogni genoma della popolazione
        self.seed_addition += 1
        for _, genome in genomes:
            
            # 1. Crea la rete neurale (FeedForwardNetwork) da questo genoma
            net = neat.nn.FeedForwardNetwork.create(genome, neat_config)
            
            # 2. Resetta l'ambiente
            #    Usiamo lo stesso seed per assicurare che *tutti* i genomi
            #    partano dalla stessa identica condizione (valutazione equa).
            obs, _ = self.env.reset(seed=self.cfg.seed + self.seed_addition)

            cum_reward = 0.0
            steps = 0

            # 3. Esegui la simulazione (un episodio)
            while steps < self.cfg.max_episode_steps:
                
                # Applica il rumore all'osservazione, se fornito
                noisy_obs = obs if noise_vec is None else (obs + noise_vec)
                # In evaluate_genomes:
                # Chiedi alla rete di decidere un'azione
                raw_action = np.array(net.activate(noisy_obs.tolist()), dtype=np.float32)
                # Applica tanh per mappare l'output in [-1, 1]
                action = np.tanh(np.array(raw_action, dtype=np.float32))
                # Esegui l'azione nell'ambiente
                obs, reward, terminated, truncated, _ = self.env.step(action)

                cum_reward += float(reward)
                steps += 1
                
                # Interrompi l'episodio se l'agente cade (terminated) 
                # o finisce il tempo massimo (truncated)
                if terminated or truncated:
                    break

            # 4. Assegna la ricompensa totale come fitness del genoma
            genome.fitness = cum_reward

# ======================================================================================
# Video helper
# ======================================================================================

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

# ======================================================================================
# Trainer NEAT
# ======================================================================================


class NEATTrainer:
    '''
    Spiegazione classe:
    Orchestra l'intero processo di addestramento NEAT.
    
    Questa classe è il "cervello" dell'esperimento. È responsabile di:
    1. Inizializzare l'ambiente, la configurazione NEAT e la popolazione.
    2. Creare un pool di processi paralleli per la valutazione della fitness.
    3. Gestire il loop di addestramento principale, diviso in "fasi" di rumore.
    4. Generare e applicare rumore alle osservazioni in ogni fase.
    5. Eseguire la valutazione di ogni generazione in parallelo.
    6. Tracciare le statistiche (miglior fitness, media) e la storia.
    7. Salvare i checkpoint (stato, grafici, video del migliore) alla fine 
       di ogni fase.
    '''
    
    def __init__(self, cfg: TrainConfig):
        '''
        Spiegazione funzione (Costruttore):
        Inizializza l'intera infrastruttura di addestramento.
        
        Questa funzione viene chiamata quando si crea un oggetto `NEATTrainer`.
        Prepara tutto il necessario per l'addestramento: carica la
        configurazione, crea le cartelle, imposta il seed, "sonda"
        l'ambiente per capirne le dimensioni, inizializza la popolazione
        NEAT e avvia il pool di processi paralleli.

        Spiegazione input:
            cfg (TrainConfig): L'oggetto di configurazione (dataclass)
                               che contiene tutti gli iperparametri
                               dell'esperimento.

        Spiegazione output:
            None: La funzione non restituisce valori, ma popola l'oggetto
                  `self` con tutti gli attributi necessari (es.
                  `self.population`, `self.pool`, `self.obs_dim`, ecc.).
        '''
        
        # Salva la configurazione di training (TrainConfig) nell'oggetto
        self.cfg = cfg
        
        # Assicura che la cartella di output (es. "runs/runs_neat_2_walker2d") esista
        ensure_dir(cfg.out_dir)
        
        # Imposta il seed globale per 'numpy' e 'random' per la riproducibilità
        set_global_seed(cfg.seed)

        # --- Sonda (probe) l'ambiente per ottenere le dimensioni ---
        # Crea un'istanza temporanea dell'ambiente
        probe = gym.make(
            cfg.env_id,
            exclude_current_positions_from_observation=cfg.exclude_current_positions_from_observation,
            forward_reward_weight=cfg.forward_reward_weight,
        )
        # Calcola la dimensione dello spazio di osservazione (es. 17)
        # np.prod gestisce casi in cui lo spazio è multidimensionale
        self.obs_dim = int(np.prod(probe.observation_space.shape)) 
        # Calcola la dimensione dello spazio di azione (es. 6)
        self.act_dim = int(np.prod(probe.action_space.shape))
        # Chiudi l'ambiente temporaneo, non più necessario
        probe.close()

        # --- Configurazione di NEAT ---
        # Definisce il percorso del file .cfg (es. "configs/neat_walker2d.cfg")
        config_path = pathlib.Path(cfg.neat_config_path)
        
        # Crea il file di configurazione NEAT da un template se non esiste,
        # inserendo le dimensioni di input/output e la dimensione della popolazione.
        ensure_neat_config(config_path, cfg.pop_size, self.obs_dim, self.act_dim)
        
        # Carica la configurazione NEAT dal file usando le classi standard
        self.neat_config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(config_path),
        )
        # Garantisce che la pop_size usata da NEAT sia quella del TrainConfig
        # (sovrascrive quella eventualmente letta dal file per coerenza)
        self.neat_config.pop_size = cfg.pop_size
        self.neat_config.population_size = cfg.pop_size

        # --- Inizializzazione Popolazione NEAT ---
        # Crea l'oggetto popolazione, il cuore dell'algoritmo evolutivo
        self.population = neat.Population(self.neat_config)
        
        # Aggiunge un "reporter" per stampare l'output base (gen, fitness, specie)
        # sulla console a ogni generazione.
        self.population.add_reporter(neat.StdOutReporter(True))
        
        # Crea un "reporter" per tracciare le statistiche (min, max, media, ecc.)
        self.stats = neat.StatisticsReporter()
        # Aggiunge il reporter statistico alla popolazione
        self.population.add_reporter(self.stats)
        
        # Contatore usato per variare il seed di valutazione ad ogni generazione
        self.seed_addition = 0

        # --- Inizializzazione Pool Parallelo ---
        # Rileva il numero di core della CPU sulla macchina
        num_cores = multiprocessing.cpu_count()
        print(f"--- Creazione di un Pool con {num_cores} worker paralleli ---")
        # Avvia il pool di processi worker per la valutazione parallela
        self.pool = multiprocessing.Pool(processes=num_cores)

        # --- Inizializzazione Stato e Storia ---
        # Dizionario per salvare la storia della fitness (massima e media)
        self.history: Dict[str, List[float]] = {"gen_max": [], "gen_avg": []}
        
        # Contenitore per il miglior genoma *assoluto* mai visto
        self.best_genome = None
        
        # Contenitore per la fitness del miglior genoma (inizializzato a -infinito)
        self.best_fitness: float = -np.inf
        
        # (Attributo residuo, non più usato, dato che ora il rumore
        # è un vettore `noise_vec`)
        self.current_phase_noise_value: float = 0.0

    def close(self) -> None:
        '''
        Spiegazione funzione:
        Chiude correttamente il pool di processi paralleli.
        
        È fondamentale chiamare questa funzione alla fine dell'addestramento
        per terminare tutti i processi "worker" ed evitare che rimangano
        "appesi" (zombie processes).

        Spiegazione input:
            self: L'istanza della classe.

        Spiegazione output:
            None
        '''
        # Stampa un messaggio di notifica
        print("--- Chiusura del pool di worker ---")
        
        # Dice al pool di non accettare nuovi task
        self.pool.close()
        
        # Attende che tutti i task correnti e i processi worker terminino
        self.pool.join()

    # (Nota: Il type hint originale `-> Tuple[float, np.ndarray]` 
    #  non corrisponde al codice, che ritorna solo `np.ndarray`. 
    #  La documentazione segue ciò che il codice *fa*.)
    def _phase_noise(self, phase_idx: int) -> np.ndarray:
        '''
        Spiegazione funzione:
        Genera e salva un nuovo vettore di rumore per una nuova "fase".
        
        All'inizio di ogni fase di addestramento, questa funzione crea
        un singolo vettore di rumore (della stessa dimensione 
        dell'osservazione). Questo vettore verrà poi sommato alle
        osservazioni per *tutte* le generazioni all'interno di *questa* fase.
        Salva anche il vettore (`.npy`) e la sua norma (`.json`) su disco.

        Spiegazione input:
            phase_idx (int): L'indice della fase corrente (es. 1, 2, ...).
                             Usato per nominare i file di output.

        Spiegazione output:
            np.ndarray: Un vettore di rumore campionato da una distribuzione
                        Gaussiana (media 0, dev. std. da `cfg.noise_std`).
        '''
        
        # Genera un vettore di rumore Gaussiano (media 0)
        noise_vec = np.random.normal(
            loc=0.0, 
            scale=self.cfg.noise_std, 
            size=(self.obs_dim,)
        ).astype(np.float32)
        
        # Salva il vettore di rumore in un file binario NumPy
        np.save(
            os.path.join(self.cfg.out_dir, f"phase_{phase_idx:02d}_noise.npy"), 
            noise_vec
        )
        
        # Salva la norma L2 (grandezza) del vettore in un file JSON
        # per un rapido controllo
        with open(os.path.join(self.cfg.out_dir, f"phase_{phase_idx:02d}_noise.json"), "w") as fh:
            json.dump(
                {"phase_noise_norm": float(np.linalg.norm(noise_vec))}, 
                fh, 
                indent=2
            )

        # Restituisce il vettore generato
        return noise_vec

    def _save_checkpoint(self, phase_idx: int, gen_global: int) -> None:
        '''
        Spiegazione funzione:
        Salva lo stato completo dell'addestramento (checkpoint).
        
        Questa funzione viene chiamata alla fine di ogni fase. Crea una
        nuova sottocartella (es. "checkpoint_phase01_gen0025") e salva
        tutto il necessario per analizzare o riprendere l'esperimento:
        1. `state.json`: Configurazione, storia, fitness, ecc.
        2. `best_genome.pkl`: Il miglior genoma (serializzato con pickle).
        3. `neat_state-*`: Lo stato interno di NEAT (per riprendere l'evoluzione).
        4. `fitness_curve.png`: Un grafico della fitness massima e media.
        5. `best.mp4`: Un video del miglior genoma in azione (se cv2 è disp.).

        Spiegazione input:
            phase_idx (int): L'indice della fase appena completata.
            gen_global (int): Il numero totale di generazioni
                               trascorse finora.

        Spiegazione output:
            None: La funzione non restituisce nulla, ma scrive molti
                  file su disco.
        '''
        
        # Costruisce il percorso della cartella per questo checkpoint
        ckpt_dir = os.path.join(
            self.cfg.out_dir, 
            f"checkpoint_phase{phase_idx:02d}_gen{gen_global:04d}"
        )
        # Crea la cartella se non esiste
        ensure_dir(ckpt_dir)

        # 1. Salva lo stato "leggibile" (config, storia, ecc.)
        state = {
            "config": asdict(self.cfg), # Converte il dataclass in dizionario
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "history": self.history, # Salva la storia delle fitness
            "best_fitness": float(self.best_fitness),
            "current_phase_noise_value": self.current_phase_noise_value, # Residuo
            "neat_generation": self.population.generation, # A che generazione NEAT è arrivato
        }
        with open(os.path.join(ckpt_dir, "state.json"), "w") as fh:
            json.dump(state, fh, indent=2)

        # 2. Salva il miglior genoma (se esiste)
        if self.best_genome is not None:
            with open(os.path.join(ckpt_dir, "best_genome.pkl"), "wb") as fh:
                # Usa pickle per serializzare l'oggetto genoma
                pickle.dump(self.best_genome, fh)

        # 3. Salva lo stato interno di NEAT (per ripristino)
        checkpointer = neat.Checkpointer(
            generation_interval=None, # Non salvare automaticamente
            time_interval_seconds=None, # Non salvare automaticamente
            filename_prefix=os.path.join(ckpt_dir, "neat_state-"), # Prefisso file
        )
        # Salva manualmente lo stato della popolazione
        checkpointer.save_checkpoint(
            self.neat_config,
            self.population.population,
            self.population.species,
            self.population.generation,
        )

        # 4. Salva il grafico della fitness
        plt.figure(figsize=(7, 4))
        plt.plot(self.history["gen_max"], label="Max fitness")
        plt.plot(self.history["gen_avg"], label="Avg fitness")
        plt.xlabel("Generazione")
        plt.ylabel("Fitness")
        plt.title(f"NEAT Walker2d (seed={self.cfg.seed})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(ckpt_dir, "fitness_curve.png"))
        plt.close() # Chiudi la figura per liberare memoria

        # 5. Salva il video del miglior genoma (se esiste)
        if self.best_genome is not None:
            # Carica il vettore di rumore usato in *questa* fase
            noise_path = os.path.join(
                self.cfg.out_dir, f"phase_{phase_idx:02d}_noise.npy"
            )
            noise_vec = np.load(noise_path) if os.path.exists(noise_path) else None
            
            try:
                # Tenta di registrare il video
                record_video(
                    self.cfg,
                    self.best_genome,
                    self.neat_config,
                    self.obs_dim,
                    self.act_dim,
                    noise_vec, # Usa il rumore della fase corrente
                    os.path.join(ckpt_dir, "best.mp4"),
                )
            except Exception:
                # Ignora errori (es. cv2 non installato, problemi di rendering)
                pass

    def _evaluate_current_generation(self, noise_vec: Optional[np.ndarray]) -> None:
        '''
        Spiegazione funzione:
        Valuta la fitness dell'intera popolazione corrente in parallelo.
        
        Questa è la funzione centrale del loop di training. Prepara
        una funzione (`eval_fn`) che NEAT può chiamare. Questa `eval_fn`
        a sua volta:
        1. Prepara i "pacchetti di lavoro" (task) per ogni genoma.
        2. Distribuisce questi task al pool di processi paralleli.
        3. Raccoglie i risultati (le fitness) da tutti i worker.
        4. Assegna la fitness calcolata a ciascun genoma.

        Spiegazione input:
            noise_vec (Optional[np.ndarray]): Il vettore di rumore
                                               costante da applicare
                                               durante questa valutazione.

        Spiegazione output:
            None: La funzione non restituisce nulla, ma aggiorna
                  l'attributo `genome.fitness` per ogni genoma
                  nella `self.population`.
        '''
        
        # 1. Incrementa il contatore del seed.
        #    Questo assicura che tutti i genomi in *questa* generazione
        #    siano valutati sulla *stessa* condizione iniziale (stesso seed),
        #    ma che la generazione *successiva* usi un seed diverso.
        self.seed_addition += 1
        eval_seed = self.cfg.seed + self.seed_addition

        # Definisce la funzione che NEAT userà per valutare i genomi.
        # Questa funzione interna ("closure") ha accesso a 'self',
        # 'noise_vec' e 'eval_seed' dal suo contesto.
        def eval_fn(genomes: List[Tuple[int, neat.DefaultGenome]], neat_cfg: neat.Config):
            
            # 2. Prepara i "pacchetti di lavoro" (task) per il pool
            #    Ogni task è una tupla che contiene *tutto* ciò che
            #    serve al processo worker (che è isolato).
            tasks = []
            for genome_id, genome in genomes:
                tasks.append(
                    # (genoma, config, cfg_training, rumore, seed_valutazione)
                    (genome, neat_cfg, self.cfg, noise_vec, eval_seed)
                )

            # 3. Esegui in parallelo!
            #    `pool.map` distribuisce la lista 'tasks' tra i
            #    processi worker. Ogni worker chiama `eval_single_genome`
            #    (definita a livello globale) su un task.
            #    Poi raccoglie i risultati (fitness) nello stesso ordine.
            try:
                fitness_list = self.pool.map(eval_single_genome, tasks)
            except Exception as e:
                # Gestione di errori (rara, ma possibile)
                print(f"Errore durante la valutazione parallela: {e}")
                # Assegna fitness 0 a tutti per non crashare l'evoluzione
                fitness_list = [0.0] * len(genomes)

            # 4. Assegna la fitness ai genomi
            #    Scorre i genomi e i risultati di fitness corrispondenti
            for (genome_id, genome), fitness in zip(genomes, fitness_list):
                genome.fitness = fitness

        # Di' a NEAT di eseguire la valutazione.
        # NEAT chiamerà la nostra `eval_fn` interna, passandole
        # la popolazione corrente. Il '1' significa "esegui per 1 generazione".
        self.population.run(eval_fn, 1)

    def _update_history(self) -> Tuple[float, float, float]:
        '''
        Spiegazione funzione:
        Aggiorna la storia delle fitness e traccia il miglior genoma.
        
        Chiamata dopo ogni generazione, questa funzione estrae le
        statistiche (max e media) dal reporter `self.stats` di NEAT,
        le aggiunge a `self.history`, e controlla se il miglior
        genoma di questa generazione è il migliore *assoluto* mai visto.
        Se lo è, lo salva in `self.best_genome`.

        Spiegazione input:
            self: L'istanza della classe.

        Spiegazione output:
            Tuple[float, float, float]: Una tupla contenente:
                (gen_max, gen_avg, best_fitness)
                - gen_max: La fitness massima di questa generazione.
                - gen_avg: La fitness media di questa generazione.
                - best_fitness: La migliore fitness assoluta vista finora.
        '''
        
        # Se le statistiche non sono ancora state popolate, ritorna 0
        if not self.stats.most_fit_genomes:
            return 0.0, 0.0, self.best_fitness

        # Ottieni il genoma migliore di quest'ultima generazione
        current_best = self.stats.most_fit_genomes[-1]
        gen_max = float(current_best.fitness)

        # Ottieni la fitness media di quest'ultima generazione
        fitness_means = self.stats.get_fitness_mean()
        gen_avg = float(fitness_means[-1]) if fitness_means else 0.0

        # Aggiungi i valori alla storia
        self.history["gen_max"].append(gen_max)
        self.history["gen_avg"].append(gen_avg)

        # Controlla se abbiamo un nuovo miglior genoma assoluto
        if gen_max > self.best_fitness or self.best_genome is None:
            self.best_fitness = gen_max
            # Salva una *copia* profonda del genoma.
            # pickle.loads(pickle.dumps(...)) è un trucco per
            # clonare l'oggetto, scollegandolo dalla popolazione
            # di NEAT (che potrebbe modificarlo).
            self.best_genome = pickle.loads(pickle.dumps(current_best))

        # Ritorna le statistiche correnti
        return gen_max, gen_avg, self.best_fitness

    def train(self) -> None:
        '''
        Spiegazione funzione:
        Esegue il loop di addestramento principale.
        
        Questa è la funzione che avvia e gestisce l'intero esperimento.
        Si articola in due loop annidati:
        1. Loop esterno (Fasi): Itera sul numero di "fasi".
           All'inizio di ogni fase, genera un *nuovo* vettore di rumore.
        2. Loop interno (Generazioni): Esegue N generazioni di NEAT,
           usando *sempre lo stesso* rumore per quella fase.
        
        Dopo ogni generazione, aggiorna la storia.
        Dopo ogni fase, salva un checkpoint completo.

        Spiegazione input:
            self: L'istanza della classe.

        Spiegazione output:
            None: La funzione esegue l'addestramento e salva
                  i risultati su disco.
        '''
        
        # Calcola il numero totale di generazioni
        total_generations = self.cfg.max_generations_per_phase * self.cfg.phases
        gen_global = 0 # Contatore globale delle generazioni

        # Loop esterno: itera sulle FASI
        for phase_idx in range(1, self.cfg.phases + 1):
            
            # Genera un nuovo vettore di rumore per questa fase
            noise_vec = self._phase_noise(phase_idx)
            print(
                f"[Phase {phase_idx}/{self.cfg.phases}] Noise vector kept constant this phase."
            )

            # Loop interno: itera sulle GENERAZIONI all'interno della fase
            for gen_idx in range(1, self.cfg.max_generations_per_phase + 1):
                
                # Valuta tutti i genomi della popolazione corrente
                # usando il rumore costante 'noise_vec'
                self._evaluate_current_generation(noise_vec=noise_vec)
                
                # Incrementa il contatore globale
                gen_global += 1

                # Aggiorna la storia e ottieni le statistiche
                gen_max, gen_avg, best = self._update_history()
                
                # Stampa un report di stato
                print(
                    f"[Phase {phase_idx}/{self.cfg.phases}] "
                    f"Gen {gen_idx}/{self.cfg.max_generations_per_phase} "
                    f"(global {gen_global}/{total_generations}) | "
                    f"max={gen_max:.2f} avg={gen_avg:.2f} best={best:.2f}"
                )

            # Alla fine di ogni FASE, salva un checkpoint
            self._save_checkpoint(phase_idx, gen_global)

        # Salva un checkpoint finale dopo l'ultima fase
        self._save_checkpoint(self.cfg.phases, total_generations)
        
        # Chiudi il pool di processi paralleli
        self.close()


# ======================================================================================
# Entry point
# ======================================================================================


def main() -> None:
    cfg = TrainConfig()
    trainer = NEATTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
