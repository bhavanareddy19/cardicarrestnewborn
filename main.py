"""
Neural Health Predictor: Deep Learning for Cardiac Arrest in Newborns.

Master orchestration script tying together the data pipeline, BERT embedding
extraction, 12-DNN ensemble, hyperparameter optimization, and evaluation.

Usage:
    python main.py --mode full          # Run everything end-to-end
    python main.py --mode ensemble      # Train 12-model ensemble only
    python main.py --mode bert          # Generate BERT embeddings only
    python main.py --mode hpo           # Run hyperparameter optimization only
    python main.py --mode evaluate      # Evaluate saved models on test set

    Options:
    --hpo-backend optuna|ray|combined   # HPO backend (default: optuna)
    --hpo-trials 10000                  # Number of HPO trials (default: 5000)
    --epochs 200                        # Training epochs (default: 200)
    --batch-size 64                     # Batch size (default: 64)
    --seed 42                           # Random seed (default: 42)
"""

import argparse
import logging
import os
import sys

import numpy as np

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    LOGS_DIR,
    SAVED_MODELS_DIR,
)


def setup_logging():
    """Configure logging to console and file."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(LOGS_DIR, "training.log"), encoding="utf-8"
            ),
        ],
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Neural Health Predictor -- Deep Learning for Cardiac Arrest in Newborns"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "ensemble", "bert", "hpo", "evaluate"],
        default="full",
        help="Execution mode (default: full)",
    )
    parser.add_argument(
        "--hpo-backend",
        choices=["optuna", "ray", "combined"],
        default="optuna",
        help="HPO backend (default: optuna)",
    )
    parser.add_argument(
        "--hpo-trials", type=int, default=5000,
        help="Number of HPO trials per backend (default: 5000)",
    )
    parser.add_argument(
        "--epochs", type=int, default=200,
        help="Training epochs (default: 200)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()


def run_bert_extraction(df_raw, pipeline, logger):
    """Generate clinical text and extract BERT embeddings."""
    from src.data.clinical_text import ClinicalTextGenerator
    from src.models.bert_feature_extractor import BERTFeatureExtractor

    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

    logger.info("Generating clinical text from tabular data...")
    text_gen = ClinicalTextGenerator()
    all_texts = text_gen.generate_all(df_raw)
    logger.info("Generated %d clinical text paragraphs", len(all_texts))
    logger.info("Sample text: %s", all_texts[0][:200])

    # Split texts to match train/val/test indices
    train_texts = [all_texts[i] for i in pipeline.train_indices]
    val_texts = [all_texts[i] for i in pipeline.val_indices]
    test_texts = [all_texts[i] for i in pipeline.test_indices]

    # Extract BioBERT embeddings
    logger.info("Extracting BioBERT embeddings...")
    extractor = BERTFeatureExtractor()
    bert_train = extractor.extract_embeddings(train_texts)
    bert_val = extractor.extract_embeddings(val_texts)
    bert_test = extractor.extract_embeddings(test_texts)

    # Save to disk
    np.save(os.path.join(SAVED_MODELS_DIR, "bert_train.npy"), bert_train)
    np.save(os.path.join(SAVED_MODELS_DIR, "bert_val.npy"), bert_val)
    np.save(os.path.join(SAVED_MODELS_DIR, "bert_test.npy"), bert_test)

    logger.info(
        "BERT embeddings saved -- train: %s, val: %s, test: %s",
        bert_train.shape, bert_val.shape, bert_test.shape,
    )
    return bert_train, bert_val, bert_test


def load_bert_embeddings(logger):
    """Load pre-computed BERT embeddings from disk if available."""
    paths = {
        "train": os.path.join(SAVED_MODELS_DIR, "bert_train.npy"),
        "val": os.path.join(SAVED_MODELS_DIR, "bert_val.npy"),
        "test": os.path.join(SAVED_MODELS_DIR, "bert_test.npy"),
    }
    if all(os.path.exists(p) for p in paths.values()):
        logger.info("Loading pre-computed BERT embeddings from disk...")
        return (
            np.load(paths["train"]),
            np.load(paths["val"]),
            np.load(paths["test"]),
        )
    return None, None, None


def main():
    args = parse_args()
    setup_logging()
    logger = logging.getLogger("main")

    # --- Reproducibility ---
    from src.utils.reproducibility import set_global_seed
    set_global_seed(args.seed)
    logger.info("Random seed set to %d", args.seed)

    # --- Device detection ---
    from src.device_strategy import detect_and_create_strategy, get_device_summary
    strategy = detect_and_create_strategy()
    device_info = get_device_summary()
    logger.info("Device summary: %s", device_info)

    # --- Data pipeline ---
    from src.data.preprocessing import DataPipeline
    logger.info("Loading and preprocessing data...")
    pipeline = DataPipeline()
    data = pipeline.prepare_all()

    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]
    X_train_raw = data["X_train_raw"]
    X_val_raw = data["X_val_raw"]
    X_test_raw = data["X_test_raw"]
    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]
    df_raw = data["df_raw"]

    logger.info(
        "Data splits -- Train: %d, Val: %d, Test: %d",
        len(X_train), len(X_val), len(X_test),
    )

    # --- BERT embedding extraction ---
    bert_train, bert_val, bert_test = None, None, None

    if args.mode in ("full", "bert"):
        bert_train, bert_val, bert_test = run_bert_extraction(
            df_raw, pipeline, logger
        )
    elif args.mode in ("ensemble", "evaluate"):
        bert_train, bert_val, bert_test = load_bert_embeddings(logger)

    if args.mode == "bert":
        logger.info("BERT extraction complete. Exiting.")
        return

    # --- Ensemble training ---
    if args.mode in ("full", "ensemble"):
        from src.models.ensemble import EnsemblePredictor

        logger.info("=" * 60)
        logger.info("TRAINING 12-MODEL ENSEMBLE")
        logger.info("=" * 60)

        ensemble = EnsemblePredictor(strategy=strategy)
        ensemble.train_all(
            X_train, y_train, X_val, y_val,
            X_train_raw=X_train_raw, X_val_raw=X_val_raw,
            bert_train=bert_train, bert_val=bert_val,
            epochs=args.epochs, batch_size=args.batch_size,
        )

        # Save ensemble
        ensemble.save_all()

        # Evaluate on test set
        from src.evaluation.metrics import MetricsCalculator
        from src.evaluation.visualization import (
            plot_confusion_matrix,
            plot_ensemble_comparison,
            plot_roc_curves,
        )

        logger.info("=" * 60)
        logger.info("EVALUATING ENSEMBLE ON TEST SET")
        logger.info("=" * 60)

        auc_results = ensemble.evaluate_auc(
            X_test, y_test,
            X_raw=X_test_raw, bert_embeddings=bert_test,
        )
        for key, val in auc_results.items():
            logger.info("  %s: %.4f", key, val)

        metrics = MetricsCalculator()
        ensemble_probs = ensemble.predict_weighted_ensemble(
            X_test, X_raw=X_test_raw, bert_embeddings=bert_test,
        )
        report = metrics.full_report(y_test, ensemble_probs)

        logger.info("Accuracy: %.4f", report["accuracy"])
        logger.info("Precision (macro): %.4f", report["precision_macro"])
        logger.info("Recall (macro): %.4f", report["recall_macro"])
        logger.info("F1 (macro): %.4f", report["f1_macro"])
        logger.info("AUC (macro): %.4f", report["auc"]["macro"])
        logger.info("\n%s", report["classification_report"])

        # Generate plots
        plots_dir = os.path.join(LOGS_DIR, "plots")
        y_pred = np.argmax(ensemble_probs, axis=1)

        plot_roc_curves(
            y_test, ensemble_probs,
            save_path=os.path.join(plots_dir, "roc_curves.png"),
        )
        plot_confusion_matrix(
            y_test, y_pred,
            save_path=os.path.join(plots_dir, "confusion_matrix.png"),
        )
        plot_ensemble_comparison(
            ensemble.model_names,
            ensemble.individual_aucs,
            auc_results.get("weighted_ensemble_auc", auc_results["ensemble_auc"]),
            save_path=os.path.join(plots_dir, "ensemble_comparison.png"),
        )
        logger.info("Plots saved to %s", plots_dir)

    # --- Hyperparameter optimization ---
    if args.mode in ("full", "hpo"):
        from src.optimization.hpo_runner import HPORunner

        logger.info("=" * 60)
        logger.info("HYPERPARAMETER OPTIMIZATION (%s)", args.hpo_backend.upper())
        logger.info("=" * 60)

        hpo = HPORunner(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            strategy=strategy,
        )

        if args.hpo_backend == "combined":
            result = hpo.run_combined(
                optuna_trials=args.hpo_trials,
                ray_trials=args.hpo_trials,
            )
        else:
            result = hpo.run(backend=args.hpo_backend, n_trials=args.hpo_trials)

        logger.info("HPO Results:")
        logger.info("  Best AUC: %.4f", result["best_auc"])
        logger.info("  Best params: %s", result["best_params"])
        logger.info(
            "  Total trials: %d",
            result.get("total_trials", result.get("n_trials_completed", 0)),
        )

    # --- Evaluate saved models ---
    if args.mode == "evaluate":
        from src.models.ensemble import EnsemblePredictor
        from src.evaluation.metrics import MetricsCalculator

        logger.info("Loading saved ensemble for evaluation...")
        ensemble = EnsemblePredictor(strategy=strategy)
        ensemble.load_all()

        auc_results = ensemble.evaluate_auc(
            X_test, y_test,
            X_raw=X_test_raw, bert_embeddings=bert_test,
        )
        for key, val in auc_results.items():
            logger.info("  %s: %.4f", key, val)

        metrics = MetricsCalculator()
        ensemble_probs = ensemble.predict_weighted_ensemble(
            X_test, X_raw=X_test_raw, bert_embeddings=bert_test,
        )
        report = metrics.full_report(y_test, ensemble_probs)
        logger.info("\n%s", report["classification_report"])

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
