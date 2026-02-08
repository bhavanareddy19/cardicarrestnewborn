"""Unified hyperparameter optimization interface supporting Optuna and Ray Tune."""

import logging
from typing import Optional

import numpy as np
import tensorflow as tf

from src.config import OPTUNA_DEFAULT_TRIALS, RAY_DEFAULT_TRIALS

logger = logging.getLogger(__name__)


class HPORunner:
    """Unified interface for hyperparameter optimization.

    Supports running Optuna, Ray Tune, or both combined for 10k+ configurations.
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        strategy: tf.distribute.Strategy,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.strategy = strategy

    def run(
        self,
        backend: str = "optuna",
        n_trials: int = 5000,
    ) -> dict:
        """Run HPO with the specified backend.

        Args:
            backend: 'optuna' or 'ray'.
            n_trials: Number of trials to run.

        Returns:
            Dict with best AUC and best hyperparameters.
        """
        if backend == "optuna":
            return self._run_optuna(n_trials)
        elif backend == "ray":
            return self._run_ray(n_trials)
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'optuna' or 'ray'.")

    def run_combined(
        self,
        optuna_trials: int = OPTUNA_DEFAULT_TRIALS,
        ray_trials: int = RAY_DEFAULT_TRIALS,
    ) -> dict:
        """Run BOTH Optuna and Ray Tune, returning the overall best result.

        Total configurations explored = optuna_trials + ray_trials (10k+ default).

        Returns:
            Dict with best AUC, best params, and results from both backends.
        """
        logger.info(
            "Running combined HPO: %d Optuna + %d Ray Tune = %d total",
            optuna_trials, ray_trials, optuna_trials + ray_trials,
        )

        optuna_result = self._run_optuna(optuna_trials)
        ray_result = self._run_ray(ray_trials)

        total_trials = (
            optuna_result["n_trials_completed"] + ray_result["n_trials_completed"]
        )

        if optuna_result["best_auc"] >= ray_result["best_auc"]:
            best = optuna_result
            best_backend = "optuna"
        else:
            best = ray_result
            best_backend = "ray"

        logger.info(
            "Combined HPO complete. Best AUC: %.4f from %s (%d total trials)",
            best["best_auc"], best_backend, total_trials,
        )

        return {
            "best_auc": best["best_auc"],
            "best_params": best["best_params"],
            "best_backend": best_backend,
            "total_trials": total_trials,
            "optuna_result": optuna_result,
            "ray_result": ray_result,
        }

    def _run_optuna(self, n_trials: int) -> dict:
        from src.optimization.optuna_tuner import OptunaTuner

        tuner = OptunaTuner(
            X_train=self.X_train,
            y_train=self.y_train,
            X_val=self.X_val,
            y_val=self.y_val,
            strategy=self.strategy,
            n_trials=n_trials,
        )
        return tuner.run()

    def _run_ray(self, n_trials: int) -> dict:
        from src.optimization.ray_tuner import RayTuner

        tuner = RayTuner(
            X_train=self.X_train,
            y_train=self.y_train,
            X_val=self.X_val,
            y_val=self.y_val,
            strategy=self.strategy,
            num_samples=n_trials,
        )
        return tuner.run()
