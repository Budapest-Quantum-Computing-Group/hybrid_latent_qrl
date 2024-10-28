from dataclasses import dataclass
from pathlib import Path
from matplotlib import pyplot as plt
from tqdm import tqdm

import ae_encoders
import ae_decoders
import state_preprocessings
import beta_schedulers

import numpy as np
import ast
import shutil
import logging
import os

import tensorflow as tf
import tensorflow.keras.optimizers as optimizers  # type: ignore
import tensorflow.keras.optimizers.schedules as lr_schedulers  # type: ignore
import tensorflow.python.keras as keras


from typing import Any, Callable, Dict, Optional, List


logging.basicConfig(
    format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG
)

logger = logging.getLogger(__name__)

@dataclass
class Config:
    epochs: int
    train_size: int
    batch_size: int
    optimizer: str

    encoder_classname: str
    encoder_kwargs: Dict[str, Any]

    decoder_classname: str
    decoder_kwargs: Dict[str, Any]

    state_preprocessing: Optional[str]
    beta_scheduler_name: str
    beta_scheduler_kwargs: Dict[str, Any]

    lr_scheduler_name: str
    lr_scheduler_kwargs: Dict[str, Any]

    save_path: Path
    dataset_path: Path
    dataset_size: int
    continued: bool
    no_pdf: bool
    jit_compile: bool

    width: Optional[float] = None
    penalty_factor: float = 10.0


@dataclass
class AE:
    config: Config
    encoder: keras.Model
    decoder: keras.Model

    ckpt_path: Path

    dataset: tf.data.Dataset
    train_set: tf.data.Dataset
    test_set: tf.data.Dataset

    epoch_start: int

    beta_scheduler: Callable
    lr_scheduler: Callable
    opt: tf.keras.optimizers.Optimizer
    managers: List[tf.train.CheckpointManager]
    width: Optional[float]
    penalty_factor: float = 10.0


def init(config: Config) -> AE:
    ckpt_path = config.save_path / "checkpoint"
    encoder = getattr(ae_encoders, config.encoder_classname)(**config.encoder_kwargs)
    decoder = getattr(ae_decoders, config.decoder_classname)(
        **config.decoder_kwargs
    )

    state_preprocessing = state_preprocessings.identity
    if config.state_preprocessing is not None:
        state_preprocessing = getattr(state_preprocessings, config.state_preprocessing)

    loaded = np.load(config.dataset_path, mmap_mode="c")
    dataset = state_preprocessing(loaded)
    train_set = dataset[:int(round(config.train_size,1)*len(dataset))]
    test_set = dataset[int(round(config.train_size,1)*len(dataset))+1:]
    dataset = tf.data.Dataset.from_tensor_slices(dataset).batch(config.batch_size)

    train_set = tf.data.Dataset.from_tensor_slices(train_set).batch(config.batch_size)
    test_set = tf.data.Dataset.from_tensor_slices(test_set).batch(config.batch_size)

    beta_scheduler = getattr(beta_schedulers, config.beta_scheduler_name)(
        **config.beta_scheduler_kwargs
    )
    lr_scheduler = getattr(lr_schedulers, config.lr_scheduler_name)(
        **config.lr_scheduler_kwargs
    )
    epoch_start = 0
    opt = getattr(optimizers, config.optimizer)(lr_scheduler)

    managers = [
        tf.train.CheckpointManager(
            tf.train.Checkpoint(encoder=encoder),
            str(ckpt_path / "encoder"),
            max_to_keep=1,
            checkpoint_name="encoder"
        ),
        tf.train.CheckpointManager(
            tf.train.Checkpoint(decoder=decoder),
            str(ckpt_path / "decoder"),
            max_to_keep=1,
            checkpoint_name="decoder"
        ),
        tf.train.CheckpointManager(
            tf.train.Checkpoint(opt=opt),
            str(ckpt_path / "opt"),
            max_to_keep=1,
            checkpoint_name="opt"
        )
    ]

    if not config.continued:
        plt.figure()
        plt.plot([lr_scheduler(k) for k in range(config.epochs)])
        plt.xlabel("# epoch")
        plt.ylabel(r"lr")
        plt.savefig(config.save_path / "lr_schedule.pdf", dpi=96)

        # RESET SCHEDULERS, THEY HAVE INTERNAL STATES!!!
        beta_scheduler = getattr(beta_schedulers, config.beta_scheduler_name)(
            **config.beta_scheduler_kwargs
        )
        lr_scheduler = getattr(lr_schedulers, config.lr_scheduler_name)(
            **config.lr_scheduler_kwargs
        )

        with open(config.save_path / "losses.csv", "w") as f:
            f.write("epoch|train_rec_loss|test_loss\n")
            f.close()

        with open(config.save_path / "losses.csv", "w") as f:
            f.write("epoch|train_rec_loss|test_loss\n")
            f.close()
    else:
        # Try loading checkpoint info.
        with open(ckpt_path / "info.txt", "r") as f:
            ckpt_ep = ast.literal_eval(f.read())["episode"]
            epoch_start = ckpt_ep + 1
            f.close()
            logger.info(f"Continuing from epoch {ckpt_ep}.")

        logger.info(f"Restoring progress file...")

        shutil.copyfile(
            ckpt_path / "losses_checkpoint.csv", config.save_path / "losses.csv"
        )

        for manager in managers:
            manager.restore_or_initialize()

        logger.info(f"Loaded models from checkpoint.")

    return AE(
        config=config,
        encoder=encoder,
        decoder=decoder,
        ckpt_path=ckpt_path,
        dataset=dataset,
        train_set=train_set,
        test_set=test_set,
        epoch_start=epoch_start,
        beta_scheduler=beta_scheduler,
        lr_scheduler=lr_scheduler,
        opt=opt,
        width=config.width,
        penalty_factor=config.penalty_factor,
        managers=managers
    )


def compute_loss_with_width(encoder, decoder, batch, width, penalty_factor, *args):
    logger.info("Regularized")
    y = encoder(batch)
    z = decoder(y)

    ae_loss = tf.math.sqrt(tf.math.reduce_mean((z - batch) ** 2))

    zero = tf.constant(0.0, dtype=tf.float32)
    penalty = penalty_factor * tf.reduce_sum(tf.math.maximum(zero, tf.abs(y) - width))
    ae_loss += penalty

    return ae_loss


def compute_loss(encoder, decoder, batch, *args):
    logger.info("Not regularized")
    y = encoder(batch)
    z = decoder(y)

    return tf.math.sqrt(tf.math.reduce_mean((z - batch) ** 2))


def train_step(encoder, decoder, batch, opt, loss_fn, *args):
    with tf.GradientTape() as tape:
        loss = loss_fn(encoder, decoder, batch, *args)

    gradients = tape.gradient(
        loss, encoder.trainable_variables + decoder.trainable_variables
    )
    opt.apply_gradients(
        zip(
            gradients,
            list(encoder.trainable_variables) + list(decoder.trainable_variables),
        )
    )

    return loss


def train(ae: AE):
    decorator = tf.function(jit_compile=ae.config.jit_compile)
    args = []
    loss_fn = compute_loss
    if ae.width is not None:
        args = [
            tf.constant(ae.width, dtype=tf.float32),
            tf.constant(ae.penalty_factor, dtype=tf.float32)
        ]
        loss_fn = compute_loss_with_width
    _train_step = decorator(train_step)
    for epoch in range(ae.epoch_start, ae.config.epochs):
        loss = tf.keras.metrics.Mean()
        for batch in tqdm(ae.train_set):
            train_loss = _train_step(ae.encoder, ae.decoder, batch, ae.opt, loss_fn, *args)
            loss(train_loss)

        # eval on test set
        test_loss = sample(ae)

        with open(os.path.join(ae.config.save_path, "losses.csv"), "a+") as f:
            f.write(f"{epoch}|{loss.result()}|{test_loss}\n")

        save(ae, epoch)
        logger.info(f"[{os.getpid()}] epoch {epoch} | Train Rec loss: {loss.result()} | Test loss {test_loss}")


def save(ae: AE, epoch: int):
    for manager in ae.managers:
        manager.save()

    logger.info(f"[{os.getpid()}] saving encoder to {ae.ckpt_path / 'encoder_weights'}...")
    ae.encoder.save_weights(ae.ckpt_path / "encoder_weights")
    logger.info(f"[{os.getpid()}] saving decoder to {ae.ckpt_path / 'decoder_weights'}...")
    ae.decoder.save_weights(ae.ckpt_path / "decoder_weights")

    logger.info(f"[{os.getpid()}] Models saved...")

    with open(ae.ckpt_path / "info.txt", "w") as f:
        f.write(str({"episode": epoch}))
        f.close()
    shutil.copyfile(
        ae.config.save_path / "losses.csv", ae.ckpt_path / "losses_checkpoint.csv"
    )
    with open(ae.ckpt_path / "config.txt", "w") as f:
        f.write(str(ae.config.__dict__))
        f.close()


def sample(ae: AE) -> tf.Tensor:
    logger.info(f"Creating sample images...")

    batch = ae.test_set.shuffle(len(ae.test_set)).take(ae.config.batch_size)
    batch = tf.concat([x for x in batch], axis=0)
    x = batch
    y = ae.encoder(x)
    z = ae.decoder(y)
    logger.info(batch.dtype)

    loss_fn = compute_loss
    args = []
    if ae.width is not None:
        args = [
            tf.constant(ae.width, dtype=tf.float32),
            tf.constant(ae.penalty_factor, dtype=tf.float32)
        ]
        loss_fn = compute_loss_with_width

    loss = loss_fn(
        ae.encoder,
        ae.decoder,
        batch,
        *args
    )

    x = x.numpy()
    y = y.numpy()
    z = z.numpy()

    os.makedirs(ae.config.save_path / "best_ae_samples", exist_ok=True)
    for k in range(10):
        if ae.config.no_pdf:
            np.save(
                ae.config.save_path / "best_ae_samples" / ("sample%02d_original" % k),
                x[k],
            )
            np.save(
                ae.config.save_path / "best_ae_samples" / ("sample%02d_latent" % k),
                y[k],
            )
            np.save(
                ae.config.save_path
                / "best_ae_samples"
                / ("sample%02d_reconstructed" % k),
                z[k],
            )
        else:
            fig, ax = plt.subplots(nrows=1, ncols=2)
            ax[0].imshow(x[k])
            ax[0].set_title("original")
            ax[1].imshow(z[k])
            ax[1].set_title("reconstructed")

            plt.savefig(
                ae.config.save_path / "best_ae_samples" / ("sample%02d.pdf" % k)
            )
            np.save(
                ae.config.save_path / "best_ae_samples" / ("sample%02d_latent" % k),
                y[k],
            )

            plt.close()  # saves memory

    return loss
