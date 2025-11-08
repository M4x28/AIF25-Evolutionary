from __future__ import annotations

import json
import os
import random
import pathlib
import pickle
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

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
    pop_size: int = 20
    
    # Quante generazioni di NEAT vengono eseguite *per ogni fase* di rumore.
    max_generations_per_phase: int = 100
    
    # Il numero totale di fasi di training. Ad ogni fase, il rumore applicato alle osservazioni viene cambiato.
    phases: int = 3

    # --- Parametri del Rumore (Robustezza) ---
    # La deviazione standard del rumore Gaussiano (casuale) che verrà aggiunto alle osservazioni.
    noise_std: float = 0.05

    # --- Parametri di Output e Logging ---
    
    # Il percorso della cartella principale dove salvare tutti i risultati (checkpoint, video, grafici). 
    out_dir: str = "runs/runs_neat_2_walker2d" 
    
    # La durata massima (in secondi) dei video registrati per il miglior genoma.
    video_seconds: int = 20

    # --- Parametri di Riproducibilità ---
    
    # Il "seed" (seme) globale per la generazione di numeri casuali.
    seed: int = 0

    # --- Percorsi File ---
    
    # Il percorso del file di configurazione specifico per 
    neat_config_path: str = "configs/neat_walker2d.cfg"


# ======================================================================================
# NEAT Evaluation Helpers
# ======================================================================================


class WalkerEvaluator:
    """Valuta una genome NEAT su Walker2d con rumore costante per fase."""

    def __init__(self, cfg: TrainConfig, obs_dim: int, act_dim: int):
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.env = gym.make(
            cfg.env_id,
            exclude_current_positions_from_observation=cfg.exclude_current_positions_from_observation,
            forward_reward_weight=cfg.forward_reward_weight,
        )

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass

    def evaluate_genomes(
        self,
        genomes,
        neat_config: neat.Config,
        noise_vec: Optional[np.ndarray],
    ) -> None:
        for _, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, neat_config)
            obs, _ = self.env.reset(seed=self.cfg.seed)

            cum_reward = 0.0
            steps = 0

            while steps < self.cfg.max_episode_steps:
                        # 1. Apply noise to the observation
                        noisy_obs = obs if noise_vec is None else (obs + noise_vec)
                        
                        # 2. Get action from network using the noisy observation
                        action_raw = np.array(net.activate(noisy_obs.tolist()), dtype=np.float32)
                        
                        # 3. Apply activation function (tanh)
                        action = np.tanh(action_raw)

                        # 4. Step the environment
                        obs, reward, terminated, truncated, _ = self.env.step(action)
                        cum_reward += float(reward)
                        steps += 1
                        if terminated or truncated:
                            break

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
    if cv2 is None:
        return

    env = gym.make(
        cfg.env_id,
        render_mode="rgb_array",
        exclude_current_positions_from_observation=cfg.exclude_current_positions_from_observation,
        forward_reward_weight=cfg.forward_reward_weight,
    )
    obs, _ = env.reset(seed=cfg.seed)
    net = neat.nn.FeedForwardNetwork.create(genome, neat_config)

    frames: List[np.ndarray] = []
    max_frames = cfg.video_seconds * cfg.render_fps
    steps = 0

    while steps < cfg.max_episode_steps and len(frames) < max_frames:
        noisy_obs = obs if noise_vec is None else (obs + noise_vec)
        action = np.tanh(np.array(net.activate(noisy_obs.tolist()), dtype=np.float32))
        obs, _, terminated, truncated, _ = env.step(action)
        rgb = env.render()
        if rgb is not None:
            frames.append(rgb)
        if terminated or truncated:
            break
        steps += 1

    env.close()

    if not frames:
        return

    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, cfg.render_fps, (w, h))
    for frame in frames:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        vw.write(bgr)
    vw.release()


# ======================================================================================
# Trainer NEAT
# ======================================================================================


class NEATTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        ensure_dir(cfg.out_dir)
        set_global_seed(cfg.seed)

        probe = gym.make(
            cfg.env_id,
            exclude_current_positions_from_observation=cfg.exclude_current_positions_from_observation,
            forward_reward_weight=cfg.forward_reward_weight,
        )
        self.obs_dim = int(np.prod(probe.observation_space.shape)) 
        self.act_dim = int(np.prod(probe.action_space.shape))
        probe.close()

        config_path = pathlib.Path(cfg.neat_config_path)
        ensure_neat_config(config_path, cfg.pop_size, self.obs_dim, self.act_dim)
        self.neat_config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(config_path),
        )
        # Garantiamo pop_size coerente con TrainConfig
        self.neat_config.pop_size = cfg.pop_size
        self.neat_config.population_size = cfg.pop_size

        self.population = neat.Population(self.neat_config)
        self.population.add_reporter(neat.StdOutReporter(True))
        self.stats = neat.StatisticsReporter()
        self.population.add_reporter(self.stats)

        self.evaluator = WalkerEvaluator(cfg, self.obs_dim, self.act_dim)

        self.history: Dict[str, List[float]] = {"gen_max": [], "gen_avg": []}
        self.best_genome = None
        self.best_fitness: float = -np.inf
        self.current_phase_noise_value: float = 0.0

    def close(self) -> None:
        self.evaluator.close()

    # Sostituzione in _phase_noise
    def _phase_noise(self, phase_idx: int) -> Tuple[float, np.ndarray]:
        # Genera un vettore di rumore, non un bias scalare
        noise_vec = np.random.normal(loc=0.0, scale=self.cfg.noise_std, size=(self.obs_dim,)).astype(np.float32)
        
        # Salva il vettore (l'idea di salvare il "valore" scalare non è più valida)
        np.save(os.path.join(self.cfg.out_dir, f"phase_{phase_idx:02d}_noise.npy"), noise_vec)
        with open(os.path.join(self.cfg.out_dir, f"phase_{phase_idx:02d}_noise.json"), "w") as fh:
            # Salva una statistica, es. la norma del vettore
            json.dump({"phase_noise_norm": float(np.linalg.norm(noise_vec))}, fh, indent=2)

        return noise_vec

    def _save_checkpoint(self, phase_idx: int, gen_global: int) -> None:
        ckpt_dir = os.path.join(self.cfg.out_dir, f"checkpoint_phase{phase_idx:02d}_gen{gen_global:04d}")
        ensure_dir(ckpt_dir)

        state = {
            "config": asdict(self.cfg),
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "history": self.history,
            "best_fitness": float(self.best_fitness),
            "current_phase_noise_value": self.current_phase_noise_value,
            "neat_generation": self.population.generation,
        }
        with open(os.path.join(ckpt_dir, "state.json"), "w") as fh:
            json.dump(state, fh, indent=2)

        if self.best_genome is not None:
            with open(os.path.join(ckpt_dir, "best_genome.pkl"), "wb") as fh:
                pickle.dump(self.best_genome, fh)

        checkpointer = neat.Checkpointer(
            generation_interval=None,
            time_interval_seconds=None,
            filename_prefix=os.path.join(ckpt_dir, "neat_state-"),
        )
        checkpointer.save_checkpoint(
            self.neat_config,
            self.population.population,
            self.population.species,
            self.population.generation,
        )

        plt.figure(figsize=(7, 4))
        plt.plot(self.history["gen_max"], label="Max fitness")
        plt.plot(self.history["gen_avg"], label="Avg fitness")
        plt.xlabel("Generazione")
        plt.ylabel("Fitness")
        plt.title("NEAT Walker2d (seed=42)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(ckpt_dir, "fitness_curve.png"))
        plt.close()

        if self.best_genome is not None:
            noise_path = os.path.join(self.cfg.out_dir, f"phase_{phase_idx:02d}_noise.npy")
            noise_vec = np.load(noise_path) if os.path.exists(noise_path) else None
            try:
                record_video(
                    self.cfg,
                    self.best_genome,
                    self.neat_config,
                    self.obs_dim,
                    self.act_dim,
                    noise_vec,
                    os.path.join(ckpt_dir, "best.mp4"),
                )
            except Exception:
                pass

    def _evaluate_current_generation(self, noise_vec: Optional[np.ndarray]) -> None:
        def eval_fn(genomes, neat_cfg):
            self.evaluator.evaluate_genomes(genomes, neat_cfg, noise_vec)

        self.population.run(eval_fn, 1)

    def _update_history(self) -> Tuple[float, float, float]:
        if not self.stats.most_fit_genomes:
            return 0.0, 0.0, self.best_fitness

        current_best = self.stats.most_fit_genomes[-1]
        gen_max = float(current_best.fitness)

        fitness_means = self.stats.get_fitness_mean()
        gen_avg = float(fitness_means[-1]) if fitness_means else 0.0

        self.history["gen_max"].append(gen_max)
        self.history["gen_avg"].append(gen_avg)

        if gen_max > self.best_fitness or self.best_genome is None:
            self.best_fitness = gen_max
            self.best_genome = pickle.loads(pickle.dumps(current_best))

        return gen_max, gen_avg, self.best_fitness

    def train(self) -> None:
        total_generations = self.cfg.max_generations_per_phase * self.cfg.phases
        gen_global = 0

        for phase_idx in range(1, self.cfg.phases + 1):
            noise_vec = self._phase_noise(phase_idx)
            print(
                f"[Phase {phase_idx}/{self.cfg.phases}] Noise vector kept constant this phase: {noise_vec}"
            )

            for gen_idx in range(1, self.cfg.max_generations_per_phase + 1):
                self._evaluate_current_generation(noise_vec=noise_vec)
                gen_global += 1

                gen_max, gen_avg, best = self._update_history()
                print(
                    f"[Phase {phase_idx}/{self.cfg.phases}] "
                    f"Gen {gen_idx}/{self.cfg.max_generations_per_phase} "
                    f"(global {gen_global}/{total_generations}) | "
                    f"max={gen_max:.2f} avg={gen_avg:.2f} best={best:.2f}"
                )

            self._save_checkpoint(phase_idx, gen_global)

        self._save_checkpoint(self.cfg.phases, total_generations)
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
