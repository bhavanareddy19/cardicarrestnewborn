"""Hyperparameter search space definitions shared by Optuna and Ray Tune."""

# Activation function choices
ACTIVATION_CHOICES = ["relu", "elu", "selu", "gelu", "swish", "leaky_relu"]

# Optimizer choices
OPTIMIZER_CHOICES = ["adam", "adamw", "sgd", "rmsprop"]

# Weight initializer choices
WEIGHT_INIT_CHOICES = ["glorot_uniform", "he_normal", "lecun_normal"]

# Batch size choices
BATCH_SIZE_CHOICES = [16, 32, 64, 128, 256]

# Search space bounds
SEARCH_SPACE = {
    "num_layers": {"low": 2, "high": 8},
    "units_per_layer": {"low": 16, "high": 512, "step": 16},
    "activation": {"choices": ACTIVATION_CHOICES},
    "dropout_rate": {"low": 0.0, "high": 0.6, "step": 0.05},
    "learning_rate": {"low": 1e-5, "high": 1e-2, "log": True},
    "optimizer": {"choices": OPTIMIZER_CHOICES},
    "batch_size": {"choices": BATCH_SIZE_CHOICES},
    "l2_reg": {"low": 1e-6, "high": 1e-2, "log": True},
    "use_batch_norm": {"choices": [True, False]},
    "use_layer_norm": {"choices": [True, False]},
    "weight_init": {"choices": WEIGHT_INIT_CHOICES},
    "epochs": {"low": 50, "high": 300, "step": 25},
}
