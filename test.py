import hydra
import importlib
import time
from omegaconf import DictConfig


@hydra.main(config_path="cfg", config_name="config")
def main(cfg: DictConfig):
    try:
        environment = importlib.import_module(f"env.{cfg.env.name}")
    except ImportError:
        print(f"Error: Env '{cfg.env.name}' not found in env/ folder.")
        return

    print(f"Testing Environment: {cfg.env.name.upper()}")
    environment.test(cfg)


if __name__ == "__main__":
    main()
