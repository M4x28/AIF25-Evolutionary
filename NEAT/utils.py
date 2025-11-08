# utils.py
import pathlib
import random
import numpy as np

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
