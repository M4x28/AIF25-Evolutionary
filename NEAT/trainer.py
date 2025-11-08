# trainer.py
import json
import os
import pathlib
import pickle
import multiprocessing
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import neat
import numpy as np

# Importa dai tuoi nuovi moduli
from config import TrainConfig
from utils import ensure_dir, set_global_seed
from neat_helpers import ensure_neat_config
from evaluation import eval_single_genome # Importa la funzione per il pool
from visualization import record_video


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
