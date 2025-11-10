# neat_helpers.py
import pathlib
from utils import ensure_dir

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
fitness_threshold     = 4000
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
bias_mutate_rate        = 0.3
bias_replace_rate       = 0.1
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient = 0.5
conn_add_prob           = 0.3
conn_delete_prob        = 0.25
enabled_default         = True
enabled_mutate_rate     = 0.01
feed_forward            = True
initial_connection      = unconnected 
node_add_prob           = 0.3
node_delete_prob        = 0.25
num_hidden              = 0
num_inputs              = {num_inputs}
num_outputs             = {num_outputs}
response_init_mean      = 1.0
response_init_stdev     = 0.1
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.1
response_mutate_rate    = 0.1
response_replace_rate   = 0.1
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30.0
weight_min_value        = -30.0
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.3
weight_replace_rate     = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 25
species_elitism      = 3

[DefaultReproduction]
elitism            = 5
survival_threshold = 0.2
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
