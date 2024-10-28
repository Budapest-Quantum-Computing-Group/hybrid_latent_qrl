class Config:
    def __init__(self, _dict):
        self.__dict__.update(_dict)

from pathlib import Path
from tqdm import tqdm

from matplotlib import pyplot as plt

import os
import json
import argparse
import random
import ast
import shutil

import tensorflow as tf
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.optimizers.schedules as lr_schedulers

import numpy as np

import logging

import ae_encoders
import ae_decoders

import beta_schedulers
import state_preprocessings


class AE:
    def __init__(self, config: Config, save_path: Path, dataset_path: Path,
                 dataset_size: int, continued: bool, no_pdf: bool) -> None:
        self.config = config
        self.save_path = save_path
        self.dataset_path = dataset_path
        self.dataset_size = dataset_size
        self.ckpt_path = self.save_path / "checkpoint"
        self.continued = continued
        self.no_pdf = no_pdf

        self.encoder = getattr(ae_encoders, config.encoder_classname)(**config.encoder_kwargs)
        self.decoder = getattr(ae_decoders, config.decoder_classname)(**config.decoder_kwargs)

        self.state_preprocessing = lambda x: x 
        if config.state_preprocessing:
            self.state_preprocessing = getattr(state_preprocessings, config.state_preprocessing)
        self.dataset = list(self.state_preprocessing(np.load(self.dataset_path)))

        if self.dataset_size: 
            self.dataset = random.sample(self.dataset, self.dataset_size)

        self.train_set = self.dataset[:int(round(config.train_size,1)*len(self.dataset))]
        self.test_set = self.dataset[int(round(config.train_size,1)*len(self.dataset))+1:]
                    
        self.beta_scheduler = getattr(beta_schedulers, config.beta_scheduler_name)(**config.beta_scheduler_kwargs)
        self.lr_scheduler = getattr(lr_schedulers, config.lr_scheduler_name)(**config.lr_scheduler_kwargs)
        self.epoch_start = 0
        self.batch_size = config.batch_size
        self.opt = getattr(optimizers, config.optimizer)(self.lr_scheduler)

        if not self.continued:
            plt.figure()
            plt.plot( [self.lr_scheduler(k) for k in range(config.epochs)] )
            plt.xlabel('# epoch')
            plt.ylabel(r'lr')
            plt.savefig(self.save_path / "lr_schedule.pdf", dpi=96)

            # RESET SCHEDULERS, THEY HAVE INTERNAL STATES!!!
            self.beta_scheduler = getattr(beta_schedulers, config.beta_scheduler_name)(**config.beta_scheduler_kwargs)
            self.lr_scheduler = getattr(lr_schedulers, config.lr_scheduler_name)(**config.lr_scheduler_kwargs)

            with open(self.save_path / "losses.csv", "w") as f:
                f.write("epoch|train_rec_loss|test_loss\n")
                f.close()
        else:
            # Try loading checkpoint info.
            with open(self.ckpt_path / "info.txt", "r") as f:
                ckpt_ep = ast.literal_eval(f.read())["episode"]
                self.epoch_start = ckpt_ep
                f.close()
                logger.info(f"Continuing from epoch {ckpt_ep}.")

            logger.info(f"Restoring progress file...")
            
            shutil.copyfile(
                self.ckpt_path / "losses_checkpoint.csv",
                self.save_path / "losses.csv"
            )

            # Save models
            if self.encoder: 
                self.encoder.load_weights(self.ckpt_path / "best_encoder")
                logger.info(f"Encoder weights loaded from checkpoint.")

            if self.decoder: 
                self.decoder.load_weights(self.ckpt_path / "best_decoder")
                logger.info(f"Decoder weights loaded from checkpoint.")

    def train(self) -> None:
        min_test_rec_loss = np.inf
        for j in range(self.epoch_start, self.config.epochs):
            batch_rec_losses = []
            
            for k in tqdm(range(len(self.train_set)//self.batch_size)):
                batch = random.sample(self.train_set, self.batch_size)
                with tf.GradientTape() as tape:

                    x = tf.Variable(batch)
                    y = self.encoder(x)
                    
                    z = self.decoder(y)
                    tf.debugging.assert_all_finite(z, "z IS NAN")

                    # rmse loss
                    reconstruction_loss = tf.math.sqrt(tf.math.reduce_mean((z-x)**2))      
                    
                    batch_rec_losses.append(reconstruction_loss.numpy())
                    
                    loss = reconstruction_loss

                    tf.debugging.assert_all_finite(loss, "LOSS IS NAN")
                    

                gradients = tape.gradient(loss, self.encoder.trainable_variables + self.decoder.trainable_variables)
                self.opt.apply_gradients(zip(gradients, list(self.encoder.trainable_variables) + list(self.decoder.trainable_variables)))

            # Evaluate on test set
            x = tf.Variable(self.test_set)
            y = self.encoder(x)
            z = self.decoder(y)

            test_rec_loss = np.sqrt(tf.math.reduce_mean((z-x)**2).numpy())
            
            if min_test_rec_loss > test_rec_loss: 
                
                logger.info(f"[{os.getpid()}] saving encoder to {self.ckpt_path / 'best_encoder'}...")
                self.encoder.save_weights(self.ckpt_path / "best_encoder")
                logger.info(f"[{os.getpid()}] saving decoder to {self.ckpt_path / 'best_decoder'}...")
                self.decoder.save_weights(self.ckpt_path / "best_decoder")

                with open(self.ckpt_path / "info.txt", "w") as f:
                    f.write(str({
                        "episode": j
                    }))
                    f.close()
                shutil.copyfile(
                    self.save_path / "losses.csv",
                    self.ckpt_path / "losses_checkpoint.csv"
                )
                with open(self.ckpt_path / "config.txt", "w") as f:
                    f.write(str(
                        self.config.__dict__
                    ))
                    f.close()
                
                min_test_rec_loss = test_rec_loss

                # Create some sample images
                logger.info(f"Creating sample images...")
                batch = random.sample(self.test_set, 20)
                x = tf.Variable(batch)
                y = self.encoder(x)
                z = self.decoder(y)

                os.makedirs(self.save_path / "best_ae_samples", exist_ok=True)

                for k in range(20):
                    if self.no_pdf == True:
                        np.save(self.save_path / "best_ae_samples" / ("sample%02d_original" % k), x[k].numpy())
                        np.save(self.save_path / "best_ae_samples" / ("sample%02d_latent" % k), y[k].numpy())
                        np.save(self.save_path / "best_ae_samples" / ("sample%02d_reconstructed" % k), z[k].numpy())
                    else:
                        fig, ax = plt.subplots(nrows=1, ncols=2)
                        ax[0].imshow(x[k].numpy())
                        ax[0].set_title('original')
                        ax[1].imshow(z[k].numpy())
                        ax[1].set_title('reconstructed')

                        plt.savefig(self.save_path / "best_ae_samples" / ("sample%02d.pdf" % k))
                        np.save(self.save_path / "best_ae_samples" / ("sample%02d_latent" % k), y[k].numpy())

                        plt.close() #saves memory

            with open(os.path.join(self.save_path, "losses.csv"), "a+") as f:
                f.write(f"{j}|{np.mean(batch_rec_losses)}|{test_rec_loss}\n")
                f.close()

            logger.info(f"[{os.getpid()}] epoch {j} | Train Rec loss: {round(np.mean(batch_rec_losses), 5)} | Test loss: {round(np.sqrt(tf.math.reduce_mean((z-x)**2).numpy()),5)}")



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

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-path', type=Path, required=True)
    parser.add_argument('--save-path', type=Path, required=True)
    parser.add_argument('--config-path', type=Path, required=True)
    parser.add_argument('--dataset-size', type=int)
    parser.add_argument('--no-pdf', action='store_true')
    parser.add_argument('--continued', action='store_true')

    return parser.parse_args()

def main() -> None:
    args = parse_arguments()

    with open(args.config_path, "r") as f:
        config = Config(json.loads(f.read()))

    ae = AE(config, args.save_path, args.dataset_path, args.dataset_size,
            args.continued, args.no_pdf)
    ae.train()


if __name__ == "__main__":
    main()
