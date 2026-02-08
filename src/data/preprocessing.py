"""Data loading, encoding, stratified splitting, and scaling pipeline."""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    CATEGORY_MAPS,
    CSV_PATH,
    DEFAULT_SEED,
    DEFAULT_TEST_SIZE,
    DEFAULT_VAL_SIZE,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TARGET_MAP,
)


class DataPipeline:
    """End-to-end data loading, encoding, splitting, and scaling."""

    def __init__(self, csv_path: str = CSV_PATH):
        self.csv_path = csv_path
        self.scaler = StandardScaler()

    def load_and_encode(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load CSV and encode categorical features to integers.

        Returns:
            (df_raw, df_encoded) -- raw keeps original strings for clinical text
            generation; encoded has integer-mapped columns.
        """
        df_raw = pd.read_csv(self.csv_path)
        df_encoded = df_raw.copy()

        for col, mapping in CATEGORY_MAPS.items():
            df_encoded[col] = df_encoded[col].map(mapping)

        df_encoded[TARGET_COLUMN] = df_encoded[TARGET_COLUMN].map(TARGET_MAP)
        return df_raw, df_encoded

    def stratified_split(
        self,
        df_encoded: pd.DataFrame,
        test_size: float = DEFAULT_TEST_SIZE,
        val_size: float = DEFAULT_VAL_SIZE,
        seed: int = DEFAULT_SEED,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Stratified train/val/test split preserving class proportions.

        Returns:
            (train_df, val_df, test_df)
        """
        y = df_encoded[TARGET_COLUMN].values

        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df_encoded, test_size=test_size, stratify=y, random_state=seed
        )

        # Second split: separate validation from remaining
        y_train_val = train_val_df[TARGET_COLUMN].values
        relative_val_size = val_size / (1.0 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=relative_val_size,
            stratify=y_train_val,
            random_state=seed,
        )

        return train_df, val_df, test_df

    def extract_features_labels(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract feature matrix X and label vector y from a DataFrame."""
        X = df[FEATURE_COLUMNS].values.astype(np.float32)
        y = df[TARGET_COLUMN].values.astype(np.int32)
        return X, y

    def scale_features(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit StandardScaler on training data, transform all splits.

        Returns:
            (X_train_scaled, X_val_scaled, X_test_scaled)
        """
        X_train_scaled = self.scaler.fit_transform(X_train).astype(np.float32)
        X_val_scaled = self.scaler.transform(X_val).astype(np.float32)
        X_test_scaled = self.scaler.transform(X_test).astype(np.float32)
        return X_train_scaled, X_val_scaled, X_test_scaled

    def prepare_all(self) -> dict:
        """Convenience method: load, encode, split, scale in one call.

        Returns:
            Dict with keys: X_train, X_val, X_test (scaled), y_train, y_val,
            y_test, X_train_raw, X_val_raw, X_test_raw (unscaled integers
            for EmbeddingNet), df_raw (original strings).
        """
        df_raw, df_encoded = self.load_and_encode()
        train_df, val_df, test_df = self.stratified_split(df_encoded)

        X_train_raw, y_train = self.extract_features_labels(train_df)
        X_val_raw, y_val = self.extract_features_labels(val_df)
        X_test_raw, y_test = self.extract_features_labels(test_df)

        X_train, X_val, X_test = self.scale_features(
            X_train_raw, X_val_raw, X_test_raw
        )

        # Store indices for matching clinical texts later
        self.train_indices = train_df.index.tolist()
        self.val_indices = val_df.index.tolist()
        self.test_indices = test_df.index.tolist()

        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "X_train_raw": X_train_raw,
            "X_val_raw": X_val_raw,
            "X_test_raw": X_test_raw,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "df_raw": df_raw,
        }
