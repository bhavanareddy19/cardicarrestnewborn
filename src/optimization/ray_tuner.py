"""Ray Tune-based hyperparameter optimization for DNN architectures."""

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
    RAY_DEFAULT_TRIALS,
)
from src.optimization.search_space import (
    ACTIVATION_CHOICES,
    BATCH_SIZE_CHOICES,
    OPTIMIZER_CHOICES,
)

logger = logging.getLogger(__name__)


class RayTuner:
    """Hyperparameter optimization using Ray Tune with ASHA scheduler."""

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        strategy: tf.distribute.Strategy,
        num_samples: int = RAY_DEFAULT_TRIALS,
        max_concurrent: int = 4,
        results_dir: Optional[str] = None,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.y_val_onehot = to_categorical(y_val, NUM_CLASSES)
        self.strategy = strategy
        self.num_samples = num_samples
        self.max_concurrent = max_concurrent
        self.results_dir = results_dir or os.path.join(HPO_RESULTS_DIR, "ray")

    def _train_fn(self, config: dict) -> None:
        """Ray Tune trainable function. Called once per trial."""
        from ray import tune

        # Build model from config
        model = self._build_model(config)

        batch_size = config.get("batch_size", 64)
        max_epochs = config.get("epochs", 200)

        for epoch in range(max_epochs):
            history = model.fit(
                self.X_train, self.y_train,
                validation_data=(self.X_val, self.y_val),
                epochs=1,
                batch_size=batch_size,
                verbose=0,
            )

            val_loss = history.history["val_loss"][0]
            val_probs = model.predict(self.X_val, verbose=0)
            val_auc = roc_auc_score(
                self.y_val_onehot, val_probs,
                multi_class="ovr", average="macro",
            )

            tune.report({"val_loss": val_loss, "val_auc": val_auc})

        tf.keras.backend.clear_session()

    def run(self) -> dict:
        """Launch Ray Tune HPO and return best hyperparameters.

        Returns:
            Dict with best AUC, best config, and number of trials.
        """
        import ray
        from ray import tune
        from ray.tune.schedulers import ASHAScheduler

        os.makedirs(self.results_dir, exist_ok=True)

        ray.init(ignore_reinit_error=True, num_cpus=4)

        search_space = {
            "num_layers": tune.randint(2, 9),
            "activation": tune.choice(ACTIVATION_CHOICES),
            "learning_rate": tune.loguniform(1e-5, 1e-2),
            "dropout_rate": tune.uniform(0.0, 0.6),
            "l2_reg": tune.loguniform(1e-6, 1e-2),
            "batch_size": tune.choice(BATCH_SIZE_CHOICES),
            "use_batch_norm": tune.choice([True, False]),
            "optimizer": tune.choice(OPTIMIZER_CHOICES),
            "units_per_layer": tune.randint(16, 513),
            "epochs": 200,
        }

        scheduler = ASHAScheduler(
            metric="val_auc",
            mode="max",
            max_t=200,
            grace_period=10,
            reduction_factor=3,
        )

        logger.info("Starting Ray Tune with %d samples", self.num_samples)

        analysis = tune.run(
            self._train_fn,
            config=search_space,
            num_samples=self.num_samples,
            scheduler=scheduler,
            local_dir=self.results_dir,
            name="cardiac_ray_hpo",
            max_concurrent_trials=self.max_concurrent,
            verbose=1,
        )

        best_trial = analysis.get_best_trial("val_auc", "max", "last")
        best_config = best_trial.config
        best_auc = best_trial.last_result["val_auc"]

        logger.info("Best Ray Tune trial AUC: %.4f", best_auc)
        logger.info("Best config: %s", best_config)

        ray.shutdown()

        return {
            "best_auc": best_auc,
            "best_params": best_config,
            "n_trials_completed": len(analysis.trials),
        }

    def _build_model(self, config: dict) -> tf.keras.Model:
        """Build a Keras model from a Ray Tune config dict."""
        num_layers = config["num_layers"]
        units = config["units_per_layer"]
        activation = config["activation"]
        dropout = config["dropout_rate"]
        l2_reg = config["l2_reg"]
        use_bn = config["use_batch_norm"]
        lr = config["learning_rate"]
        optimizer_name = config["optimizer"]

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(NUM_FEATURES,)))

        for _ in range(num_layers):
            model.add(tf.keras.layers.Dense(
                units,
                activation=activation if not use_bn else None,
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
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
