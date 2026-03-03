import hydra
from omegaconf import DictConfig
import importlib
import time


@hydra.main(config_path="cfg", config_name="train_config")
def main(cfg: DictConfig):
    try:
        environment = importlib.import_module(f"env.{cfg.env.name}")
    except ImportError:
        print(f"Error: Env '{cfg.env.name}' not found in env/ folder.")
        return
    
    print(f"Executing Environment: {cfg.env.name.upper()}")

    environment.train(cfg)

if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Execution time: {time.time() - start:.2f}s")