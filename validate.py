import hydra
import importlib
import time
from omegaconf import DictConfig


@hydra.main(version_base="1.2", config_path="cfg", config_name="config")
def main(cfg: DictConfig):
    try:
        environment = importlib.import_module(f"val.{cfg.val.name}")
        print(environment)
    except ImportError:
        print(f"Error: Validation '{cfg.val.name}' not found in val/ folder.")
        return

    print(f"Validating: {cfg.val.name.upper()}")
    environment.validate(cfg)


if __name__ == "__main__":
    main()
