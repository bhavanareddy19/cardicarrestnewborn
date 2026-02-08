"""Optuna-based hyperparameter optimization for DNN architectures."""

import logging
import os
from typing import Optional

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.keras.utils import to_categorical

from src.config import (
    HPO_RESULTS_DIR,
    NUM_CLASSES,
    NUM_FEATURES,
    OPTUNA_DEFAULT_TRIALS,
)
from src.optimization.search_space import (
    ACTIVATION_CHOICES,
    BATCH_SIZE_CHOICES,
    OPTIMIZER_CHOICES,
    WEIGHT_INIT_CHOICES,
)

logger = logging.getLogger(__name__)


class OptunaTuner:
    """Hyperparameter optimization using Optuna with TPE sampler and Hyperband pruner."""

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        strategy: tf.distribute.Strategy,
        n_trials: int = OPTUNA_DEFAULT_TRIALS,
        storage: Optional[str] = None,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.y_val_onehot = to_categorical(y_val, NUM_CLASSES)
        self.strategy = strategy
        self.n_trials = n_trials
        self.storage = storage or (
            "sqlite:///"
            + os.path.join(HPO_RESULTS_DIR, "optuna_study.db").replace("\\", "/")
        )

    def objective(self, trial) -> float:
        """Single Optuna trial: build model, train, return validation AUC."""
        import optuna

        # Sample hyperparameters
        num_layers = trial.suggest_int("num_layers", 2, 8)
        activation = trial.suggest_categorical("activation", ACTIVATION_CHOICES)
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        dropout = trial.suggest_float("dropout_rate", 0.0, 0.6, step=0.05)
        l2_reg = trial.suggest_float("l2_reg", 1e-6, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", BATCH_SIZE_CHOICES)
        use_bn = trial.suggest_categorical("use_batch_norm", [True, False])
        optimizer_name = trial.suggest_categorical("optimizer", OPTIMIZER_CHOICES)
        weight_init = trial.suggest_categorical("weight_init", WEIGHT_INIT_CHOICES)

        units = []
        for i in range(num_layers):
            units.append(trial.suggest_int(f"units_layer_{i}", 16, 512, step=16))

        # Build model
        with self.strategy.scope():
            model = self._build_model(
                units, activation, dropout, l2_reg, use_bn,
                lr, optimizer_name, weight_init,
            )

        # Training callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=15, restore_best_weights=True
            ),
        ]

        # Add Optuna pruning callback if available
        try:
            from optuna.integration import TFKerasPruningCallback
            callbacks.append(TFKerasPruningCallback(trial, "val_loss"))
        except ImportError:
            pass

        model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=200,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0,
        )

        # Compute validation AUC
        val_probs = model.predict(self.X_val, verbose=0)
        auc_score = roc_auc_score(
            self.y_val_onehot, val_probs, multi_class="ovr", average="macro"
        )

        # Clean up to prevent memory leaks
        tf.keras.backend.clear_session()

        return auc_score

    def run(self) -> dict:
        """Execute the Optuna study and return best hyperparameters.

        Returns:
            Dict with best hyperparameters and best AUC value.
        """
        import optuna

        os.makedirs(HPO_RESULTS_DIR, exist_ok=True)

        study = optuna.create_study(
            study_name="cardiac_arrest_dnn_optuna",
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=100, multivariate=True
            ),
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=10, max_resource=200, reduction_factor=3
            ),
            storage=self.storage,
            load_if_exists=True,
        )

        logger.info("Starting Optuna study with %d trials", self.n_trials)
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            gc_after_trial=True,
            show_progress_bar=True,
        )

        logger.info("Best trial AUC: %.4f", study.best_trial.value)
        logger.info("Best params: %s", study.best_trial.params)

        return {
            "best_auc": study.best_trial.value,
            "best_params": study.best_trial.params,
            "n_trials_completed": len(study.trials),
        }

    def _build_model(
        self, units, activation, dropout, l2_reg, use_bn,
        lr, optimizer_name, weight_init,
    ) -> tf.keras.Model:
        """Dynamically build a Keras model from hyperparameters."""
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(NUM_FEATURES,)))

        for u in units:
            model.add(tf.keras.layers.Dense(
                u,
                activation=activation if not use_bn else None,
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                kernel_initializer=weight_init,
            ))
            if use_bn:
                model.add(tf.keras.layers.BatchNormalization())
                model.add(tf.keras.layers.Activation(activation))
            if dropout > 0:
                model.add(tf.keras.layers.Dropout(dropout))

        model.add(tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"))

        optimizer = self._get_optimizer(optimizer_name, lr)
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    @staticmethod
    def _get_optimizer(name: str, lr: float):
        optimizers = {
            "adam": tf.keras.optimizers.Adam(learning_rate=lr),
            "adamw": tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=0.01),
            "sgd": tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9),
            "rmsprop": tf.keras.optimizers.RMSprop(learning_rate=lr),
        }
        return optimizers[name]
