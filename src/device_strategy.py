"""TPU/GPU/CPU detection and tf.distribute strategy creation."""

import logging
import tensorflow as tf

logger = logging.getLogger(__name__)


def detect_and_create_strategy() -> tf.distribute.Strategy:
    """Detect available hardware and return the appropriate tf.distribute.Strategy.

    Priority: TPU -> Multi-GPU -> Single GPU -> CPU.
    """
    # --- Attempt TPU ---
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
        logger.info(
            "TPU detected with %d replicas", strategy.num_replicas_in_sync
        )
        return strategy
    except (ValueError, tf.errors.NotFoundError):
        logger.info("No TPU detected, checking for GPUs...")

    # --- Attempt GPU ---
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        logger.info(
            "Multiple GPUs detected (%d), using MirroredStrategy with %d replicas",
            len(gpus),
            strategy.num_replicas_in_sync,
        )
        return strategy
    elif len(gpus) == 1:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError:
            pass  # Memory growth must be set before GPU is initialized
        strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
        logger.info("Single GPU detected: %s", gpus[0].name)
        return strategy

    # --- CPU fallback ---
    strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
    logger.info("No GPU detected, falling back to CPU")
    return strategy


def get_batch_size_for_strategy(
    strategy: tf.distribute.Strategy, base_batch_size: int = 64
) -> int:
    """Scale batch size by number of replicas for distributed training."""
    return base_batch_size * strategy.num_replicas_in_sync


def get_device_summary() -> dict:
    """Return a summary dict of detected hardware."""
    return {
        "gpu_count": len(tf.config.list_physical_devices("GPU")),
        "gpu_names": [g.name for g in tf.config.list_physical_devices("GPU")],
        "cpu_count": len(tf.config.list_physical_devices("CPU")),
        "tf_version": tf.__version__,
    }
