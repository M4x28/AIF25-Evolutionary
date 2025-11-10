# evaluation.py
import gymnasium as gym
import neat
import numpy as np
from typing import Tuple, List, Optional

# Importa la definizione della configurazione
from config import TrainConfig

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
    while steps <= cfg.max_episode_steps:
        noisy_obs = obs if noise_vec is None else (obs + noise_vec)
                        
        action = np.array(net.activate(noisy_obs.tolist()), dtype=np.float32)
        obs, reward, terminated, truncated, _ = env.step(action)
        
        cum_reward += float(reward)
        steps += 1
        
        if terminated or truncated:
            break
    
    # 6. Chiudi l'ambiente del worker
    env.close()
    
    # 7. Restituisci la fitness calcolata
    return cum_reward

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