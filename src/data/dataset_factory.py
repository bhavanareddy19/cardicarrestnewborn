"""tf.data.Dataset creation utilities for tabular and BERT-fused pipelines."""

import numpy as np
import tensorflow as tf

from src.config import DEFAULT_BATCH_SIZE


class DatasetFactory:
    """Create optimized tf.data.Dataset pipelines."""

    @staticmethod
    def create_tabular_dataset(
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = DEFAULT_BATCH_SIZE,
        shuffle: bool = True,
        buffer_size: int = 4096,
    ) -> tf.data.Dataset:
        """Create a dataset from tabular features and labels.

        Args:
            X: Feature matrix of shape (N, num_features).
            y: Label vector of shape (N,).
            batch_size: Batch size.
            shuffle: Whether to shuffle the dataset.
            buffer_size: Shuffle buffer size.

        Returns:
            A batched, prefetched tf.data.Dataset.
        """
        ds = tf.data.Dataset.from_tensor_slices((X, y))
        if shuffle:
            ds = ds.shuffle(buffer_size)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    @staticmethod
    def create_fusion_dataset(
        X_tabular: np.ndarray,
        bert_embeddings: np.ndarray,
        y: np.ndarray,
        batch_size: int = DEFAULT_BATCH_SIZE,
        shuffle: bool = True,
        buffer_size: int = 4096,
    ) -> tf.data.Dataset:
        """Create a dataset combining tabular features and BERT embeddings.

        Args:
            X_tabular: Tabular feature matrix (N, num_features).
            bert_embeddings: BERT [CLS] embeddings (N, 768).
            y: Label vector (N,).
            batch_size: Batch size.
            shuffle: Whether to shuffle.
            buffer_size: Shuffle buffer size.

        Returns:
            A dataset yielding ((tabular_input, bert_input), label) tuples.
        """
        ds = tf.data.Dataset.from_tensor_slices(
            ({"tabular_input": X_tabular, "bert_input": bert_embeddings}, y)
        )
        if shuffle:
            ds = ds.shuffle(buffer_size)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds
