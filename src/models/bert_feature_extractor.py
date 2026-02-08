"""BioBERT / ClinicalBERT embedding extraction from clinical text."""

import logging
from typing import List, Optional

import numpy as np

from src.config import (
    BERT_EMBEDDING_DIM,
    BERT_MAX_LENGTH,
    BIOBERT_MODEL_NAME,
    CLINICALBERT_MODEL_NAME,
)

logger = logging.getLogger(__name__)


class BERTFeatureExtractor:
    """Extract [CLS] token embeddings from pretrained BERT medical models.

    Uses HuggingFace transformers (PyTorch backend). Embeddings are extracted
    once and cached as numpy arrays for use by TensorFlow models.
    """

    def __init__(
        self,
        model_name: str = BIOBERT_MODEL_NAME,
        max_length: int = BERT_MAX_LENGTH,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self._model = None
        self._tokenizer = None

    def _load_model(self) -> None:
        """Lazy-load the BERT model and tokenizer."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoModel, AutoTokenizer

        logger.info("Loading BERT model: %s", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.eval()

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)
        logger.info("BERT model loaded on %s", self._device)

    def extract_embeddings(
        self, texts: List[str], batch_size: int = 32
    ) -> np.ndarray:
        """Extract [CLS] token embeddings for a list of clinical texts.

        Args:
            texts: List of clinical narrative strings.
            batch_size: Processing batch size.

        Returns:
            Numpy array of shape (len(texts), 768).
        """
        import torch

        self._load_model()
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self._device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = self._model(**encoded)

            # [CLS] token is at index 0 of last_hidden_state
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)

            if (i // batch_size) % 10 == 0:
                logger.info(
                    "  BERT embedding progress: %d / %d", i + len(batch), len(texts)
                )

        return np.concatenate(all_embeddings, axis=0).astype(np.float32)

    def extract_dual_embeddings(
        self, texts: List[str], batch_size: int = 32
    ) -> np.ndarray:
        """Extract embeddings from both BioBERT and ClinicalBERT, concatenated.

        Returns:
            Numpy array of shape (len(texts), 1536).
        """
        # BioBERT embeddings
        self.model_name = BIOBERT_MODEL_NAME
        self._model = None
        self._tokenizer = None
        bio_emb = self.extract_embeddings(texts, batch_size)
        logger.info("BioBERT embeddings: shape %s", bio_emb.shape)

        # ClinicalBERT embeddings
        self.model_name = CLINICALBERT_MODEL_NAME
        self._model = None
        self._tokenizer = None
        clin_emb = self.extract_embeddings(texts, batch_size)
        logger.info("ClinicalBERT embeddings: shape %s", clin_emb.shape)

        return np.concatenate([bio_emb, clin_emb], axis=1)


def extract_and_save_embeddings(
    texts: List[str],
    save_path: str,
    model_name: str = BIOBERT_MODEL_NAME,
    batch_size: int = 32,
) -> np.ndarray:
    """Convenience function: extract embeddings and save to disk.

    Args:
        texts: Clinical text list.
        save_path: Path to save the .npy file.
        model_name: HuggingFace model identifier.
        batch_size: Processing batch size.

    Returns:
        The embedding array.
    """
    extractor = BERTFeatureExtractor(model_name=model_name)
    embeddings = extractor.extract_embeddings(texts, batch_size)
    np.save(save_path, embeddings)
    logger.info("Saved BERT embeddings to %s (shape: %s)", save_path, embeddings.shape)
    return embeddings
