# record.py
import pickle
import gzip  # Per caricare il file neat_state
import neat
import numpy as np
import pathlib
import os # Importato per gestire l'errore gzip

# Importa dalle tue classi
from config import TrainConfig
from visualization import record_video

# --- 1. Definisci i percorsi ---
# Questi percorsi puntano alla cartella che hai creato
CARTELLA_VID = r"\gen1000"
GENOME_PATH = r"C:\Users\tesia\Desktop\AIF25\AIF25-Evolutionary\NEAT\tovideo" + CARTELLA_VID + r"\best_genome.pkl"
NEAT_STATE_PATH = r"C:\Users\tesia\Desktop\AIF25\AIF25-Evolutionary\NEAT\tovideo"+ CARTELLA_VID + r"\neat_state"

# Nome del file video che verrà creato nella cartella principale
OUTPUT_VIDEO_PATH = r"C:\Users\tesia\Desktop\AIF25\AIF25-Evolutionary\NEAT\tovideo" + CARTELLA_VID + r"\video_output.mp4"

print("Avvio script di registrazione...")

try:
    # --- 2. Carica il genoma (il "cervello") ---
    print(f"Caricamento genoma da: {GENOME_PATH}")
    with open(GENOME_PATH, "rb") as f:
        genome = pickle.load(f)
    print("Genoma caricato.")

    # --- 3. Carica la configurazione di default ---
    # Questo crea l'oggetto 'cfg' con i parametri di default
    cfg = TrainConfig()
    print(f"Configurazione (TrainConfig) caricata. Env: {cfg.env_id}")

    # --- 4. Carica la configurazione NEAT ---
    # Il file 'neat_state' è un pickle compresso (gzip)
    # Contiene: (generation, config, population, species_set, random_state)
    print(f"Caricamento neat_config da: {NEAT_STATE_PATH}")
    neat_config = None
    try:
        # Prova prima come gzip (formato standard dei checkpoint NEAT)
        with gzip.open(NEAT_STATE_PATH, "rb") as f:
            # A noi serve solo il secondo elemento: la configurazione NEAT
            _, neat_config, _, _, _ = pickle.load(f)
        print("Configurazione NEAT (gzipped) caricata.")
    except (gzip.BadGzipFile, OSError) as e:
        # Se fallisce (es. non è gzipped), prova come pickle normale
        print(f"Lettura Gzip fallita ({e}), provo come pickle standard...")
        with open(NEAT_STATE_PATH, "rb") as f:
            # Potrebbe essere uno stato salvato diversamente
            # Questo è un tentativo basato sul formato NEAT
            state_data = pickle.load(f)
            # Controlla se è una tupla (formato checkpoint)
            if isinstance(state_data, tuple) and len(state_data) == 5:
                 _, neat_config, _, _, _ = state_data
                 print("Configurazione NEAT (pickle) caricata.")
            else:
                 # Se non è il formato atteso, non possiamo procedere
                 raise ValueError("Formato file 'neat_state' non riconosciuto.")

    # --- 5. Estrai le dimensioni dalla config NEAT ---
    # Questi sono gli argomenti obs_dim e act_dim richiesti
    obs_dim = neat_config.genome_config.num_inputs
    act_dim = neat_config.genome_config.num_outputs
    print(f"Dimensioni rete: Input={obs_dim}, Output={act_dim}")

    # --- 6. Definisci il rumore ---
    # Imposta a 'None' per vedere il genoma in azione senza rumore
    noise_vec = None 
    print("Nessun rumore applicato per questa registrazione.")

    # --- 7. Chiama la funzione di registrazione ---
    print(f"--- Avvio registrazione... Salvo in {OUTPUT_VIDEO_PATH} ---")
    
    # Chiama la funzione che hai fornito!
    record_video(
        cfg=cfg,
        genome=genome,
        neat_config=neat_config,
        obs_dim=obs_dim,
        act_dim=act_dim,
        noise_vec=noise_vec,
        out_path=OUTPUT_VIDEO_PATH,
    )
    print(f"--- Registrazione completata! Video salvato in {OUTPUT_VIDEO_PATH} ---")

except FileNotFoundError as e:
    print(f"\nERRORE: File non trovato. Controlla i percorsi.")
    print(f"Errore: {e}")
    print(f"Assicurati che '{GENOME_PATH}' e '{NEAT_STATE_PATH}' esistano.")
except ImportError:
    print("\nERRORE: 'opencv-python' non trovato.")
    print("Per salvare i video, esegui: pip install opencv-python")
except Exception as e:
    print(f"\nSi è verificato un errore inaspettato: {e}")
    import traceback
    traceback.print_exc()