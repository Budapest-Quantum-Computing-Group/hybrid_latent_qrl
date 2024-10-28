from dataclasses import dataclass
from pathlib import Path

import tf_function.autoencoder as autoencoder

import tyro
import json
import logging
import tensorflow as tf

@dataclass
class Args:
    dataset_path: Path
    save_path: Path
    config_path: Path

    dataset_size: int = 0
    continued: bool = False
    no_pdf: bool = False


logging.basicConfig(
    format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG
)

logger = logging.getLogger(__name__)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logging.info("GPU Memory allocation set to dynamic.")
        logical_gpus = tf.config.list_logical_devices('GPU')
        logger.info(f"Number of physical GPUs: {len(gpus)}")
        logger.info(f"Number of physical GPUs: {len(logical_gpus)}")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        logger.critical(e)

if __name__ == "__main__":
    args = tyro.cli(Args)
    with open(args.config_path, "r") as f:
        config = autoencoder.Config(**json.loads(f.read()),
                                    dataset_path=args.dataset_path,
                                    save_path=args.save_path,
                                    dataset_size=args.dataset_size,
                                    continued=args.continued,
                                    no_pdf=args.no_pdf)

    ae = autoencoder.init(config)
    autoencoder.train(ae)
