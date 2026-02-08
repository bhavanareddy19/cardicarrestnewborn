"""Ensemble predictor: train, aggregate, and evaluate 12 diverse DNNs."""

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.keras.utils import to_categorical

from src.config import (
    DEFAULT_EPOCHS,
    NUM_CLASSES,
    NUM_FEATURES,
    SAVED_MODELS_DIR,
)
from src.models.architectures import (
    BERT_FUSION_MODEL_BUILDER,
    EMBEDDING_MODEL_BUILDER,
    TABULAR_MODEL_BUILDERS,
)
from src.utils.callbacks import get_training_callbacks

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """Train an ensemble of 12 DNNs and aggregate their predictions."""

    def __init__(self, strategy: tf.distribute.Strategy):
        self.strategy = strategy
        self.trained_models: List[tf.keras.Model] = []
        self.model_names: List[str] = []
        self.individual_aucs: List[float] = []

    def train_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_train_raw: Optional[np.ndarray] = None,
        X_val_raw: Optional[np.ndarray] = None,
        bert_train: Optional[np.ndarray] = None,
        bert_val: Optional[np.ndarray] = None,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = 64,
    ) -> None:
        """Train all 12 ensemble members.

        Args:
            X_train, X_val: Scaled tabular features for models 1-10 and 12.
            X_train_raw, X_val_raw: Unscaled integer features for EmbeddingNet
                (model 11). If not provided, EmbeddingNet is skipped.
            bert_train, bert_val: BERT embeddings for BERTFusion (model 12).
                If not provided, BERTFusion is skipped.
        """
        y_val_onehot = to_categorical(y_val, NUM_CLASSES)

        # --- Models 1-10: Standard tabular DNNs ---
        for i, builder in enumerate(TABULAR_MODEL_BUILDERS):
            name = builder.__name__.replace("build_", "")
            logger.info("Training model %d/12: %s", i + 1, name)

            with self.strategy.scope():
                model = builder()

            callbacks = get_training_callbacks(model_name=name)
            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0,
            )

            val_probs = model.predict(X_val, verbose=0)
            auc = roc_auc_score(
                y_val_onehot, val_probs, multi_class="ovr", average="macro"
            )
            logger.info("  %s validation AUC: %.4f", name, auc)

            self.trained_models.append(model)
            self.model_names.append(name)
            self.individual_aucs.append(auc)

        # --- Model 11: EmbeddingNet (needs raw integer features) ---
        if X_train_raw is not None and X_val_raw is not None:
            logger.info("Training model 11/12: EmbeddingNet")
            X_train_emb = self._split_for_embedding(X_train_raw)
            X_val_emb = self._split_for_embedding(X_val_raw)

            with self.strategy.scope():
                emb_model = EMBEDDING_MODEL_BUILDER()

            callbacks = get_training_callbacks(model_name="EmbeddingNet")
            emb_model.fit(
                X_train_emb, y_train,
                validation_data=(X_val_emb, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0,
            )

            val_probs = emb_model.predict(X_val_emb, verbose=0)
            auc = roc_auc_score(
                y_val_onehot, val_probs, multi_class="ovr", average="macro"
            )
            logger.info("  EmbeddingNet validation AUC: %.4f", auc)

            self.trained_models.append(emb_model)
            self.model_names.append("EmbeddingNet")
            self.individual_aucs.append(auc)
        else:
            logger.warning(
                "Raw features not provided -- skipping EmbeddingNet model"
            )

        # --- Model 12: BERTFusion ---
        if bert_train is not None and bert_val is not None:
            logger.info("Training model 12/12: BERTFusion")
            with self.strategy.scope():
                fusion_model = BERT_FUSION_MODEL_BUILDER()

            callbacks = get_training_callbacks(model_name="BERTFusion")
            fusion_model.fit(
                {"tabular_input": X_train, "bert_input": bert_train},
                y_train,
                validation_data=(
                    {"tabular_input": X_val, "bert_input": bert_val},
                    y_val,
                ),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0,
            )

            val_probs = fusion_model.predict(
                {"tabular_input": X_val, "bert_input": bert_val}, verbose=0
            )
            auc = roc_auc_score(
                y_val_onehot, val_probs, multi_class="ovr", average="macro"
            )
            logger.info("  BERTFusion validation AUC: %.4f", auc)

            self.trained_models.append(fusion_model)
            self.model_names.append("BERTFusion")
            self.individual_aucs.append(auc)
        else:
            logger.warning(
                "BERT embeddings not provided -- skipping BERTFusion model"
            )

        logger.info(
            "Ensemble training complete. %d models trained.", len(self.trained_models)
        )

    def predict_ensemble(
        self,
        X: np.ndarray,
        X_raw: Optional[np.ndarray] = None,
        bert_embeddings: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Soft voting: average predicted probabilities across all models.

        Args:
            X: Scaled features for standard models.
            X_raw: Unscaled integer features for EmbeddingNet.
            bert_embeddings: BERT embeddings for BERTFusion.

        Returns:
            Probability array of shape (N, NUM_CLASSES).
        """
        all_probs = self._collect_predictions(X, X_raw, bert_embeddings)
        return np.mean(all_probs, axis=0)

    def predict_weighted_ensemble(
        self,
        X: np.ndarray,
        X_raw: Optional[np.ndarray] = None,
        bert_embeddings: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Weighted soft voting using per-model validation AUC as weights.

        Returns:
            Probability array of shape (N, NUM_CLASSES).
        """
        all_probs = self._collect_predictions(X, X_raw, bert_embeddings)
        weights = np.array(self.individual_aucs[: len(all_probs)])
        weights = weights / weights.sum()
        return np.average(all_probs, axis=0, weights=weights)

    def evaluate_auc(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        X_raw: Optional[np.ndarray] = None,
        bert_embeddings: Optional[np.ndarray] = None,
        weighted: bool = True,
    ) -> Dict[str, float]:
        """Compute macro AUC for the ensemble and each individual model.

        Returns:
            Dict with 'ensemble_auc', 'weighted_ensemble_auc', and per-model AUCs.
        """
        y_true_onehot = to_categorical(y_true, NUM_CLASSES)

        # Ensemble predictions
        avg_probs = self.predict_ensemble(X, X_raw, bert_embeddings)
        ensemble_auc = roc_auc_score(
            y_true_onehot, avg_probs, multi_class="ovr", average="macro"
        )

        results = {"ensemble_auc": ensemble_auc}

        if weighted and self.individual_aucs:
            w_probs = self.predict_weighted_ensemble(X, X_raw, bert_embeddings)
            w_auc = roc_auc_score(
                y_true_onehot, w_probs, multi_class="ovr", average="macro"
            )
            results["weighted_ensemble_auc"] = w_auc

        # Per-model AUCs on this data
        all_probs = self._collect_predictions(X, X_raw, bert_embeddings)
        for i, (name, probs) in enumerate(
            zip(self.model_names, all_probs)
        ):
            auc = roc_auc_score(
                y_true_onehot, probs, multi_class="ovr", average="macro"
            )
            results[f"{name}_auc"] = auc

        return results

    def save_all(self, directory: Optional[str] = None) -> None:
        """Save all trained models to disk."""
        directory = directory or os.path.join(SAVED_MODELS_DIR, "ensemble")
        os.makedirs(directory, exist_ok=True)

        for model, name in zip(self.trained_models, self.model_names):
            path = os.path.join(directory, name)
            model.save(path)
            logger.info("Saved model: %s", path)

        # Save AUCs for weighted prediction
        np.save(
            os.path.join(directory, "individual_aucs.npy"),
            np.array(self.individual_aucs),
        )

    def load_all(self, directory: Optional[str] = None) -> None:
        """Load all models from disk."""
        directory = directory or os.path.join(SAVED_MODELS_DIR, "ensemble")
        self.trained_models = []
        self.model_names = []

        from src.models.architectures import ALL_MODEL_NAMES

        for name in ALL_MODEL_NAMES:
            path = os.path.join(directory, name)
            if os.path.exists(path):
                model = tf.keras.models.load_model(path)
                self.trained_models.append(model)
                self.model_names.append(name)
                logger.info("Loaded model: %s", path)

        auc_path = os.path.join(directory, "individual_aucs.npy")
        if os.path.exists(auc_path):
            self.individual_aucs = np.load(auc_path).tolist()

    # --- Internal helpers ---

    def _split_for_embedding(self, X: np.ndarray) -> list:
        """Split feature matrix into per-feature integer arrays for EmbeddingNet."""
        return [X[:, i].astype(np.int32) for i in range(NUM_FEATURES)]

    def _collect_predictions(
        self,
        X: np.ndarray,
        X_raw: Optional[np.ndarray],
        bert_embeddings: Optional[np.ndarray],
    ) -> List[np.ndarray]:
        """Collect predictions from all models, handling special input formats."""
        all_probs = []

        for model, name in zip(self.trained_models, self.model_names):
            if name == "EmbeddingNet":
                if X_raw is not None:
                    X_emb = self._split_for_embedding(X_raw)
                    probs = model.predict(X_emb, verbose=0)
                else:
                    logger.warning("Skipping EmbeddingNet -- no raw features")
                    continue
            elif name == "BERTFusion":
                if bert_embeddings is not None:
                    probs = model.predict(
                        {"tabular_input": X, "bert_input": bert_embeddings},
                        verbose=0,
                    )
                else:
                    logger.warning("Skipping BERTFusion -- no BERT embeddings")
                    continue
            else:
                probs = model.predict(X, verbose=0)
            all_probs.append(probs)

        return all_probs
