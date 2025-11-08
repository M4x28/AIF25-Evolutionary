# main.py
from config import TrainConfig
from trainer import NEATTrainer

def main() -> None:
    # 1. Crea la configurazione
    cfg = TrainConfig()
    
    # 2. Crea il trainer
    trainer = NEATTrainer(cfg)
    
    # 3. Avvia il training
    trainer.train()


if __name__ == "__main__":
    main()