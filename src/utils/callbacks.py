"""Custom Keras callbacks for training: early stopping, LR scheduling, checkpointing."""

import os
import tensorflow as tf
from src.config import (
    EARLY_STOPPING_PATIENCE,
    LR_REDUCE_PATIENCE,
    LR_REDUCE_FACTOR,
    SAVED_MODELS_DIR,
)


def get_training_callbacks(model_name: str = "model", monitor: str = "val_loss"):
    """Return a standard set of training callbacks.

    Args:
        model_name: Name for the checkpoint directory.
        monitor: Metric to monitor for early stopping and LR reduction.

    Returns:
        List of Keras callbacks.
    """
    checkpoint_dir = os.path.join(SAVED_MODELS_DIR, "checkpoints", model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=LR_REDUCE_FACTOR,
            patience=LR_REDUCE_PATIENCE,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "best_weights.keras"),
            monitor=monitor,
            save_best_only=True,
            save_weights_only=True,
            verbose=0,
        ),
    ]
