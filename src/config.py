"""Centralized configuration: feature mappings, column names, paths, training constants."""

import os

# --- Project Paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "8068")
CSV_PATH = os.path.join(DATA_DIR, "infantset.csv")
SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models")
HPO_RESULTS_DIR = os.path.join(PROJECT_ROOT, "hpo_results")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# --- Feature Columns ---
FEATURE_COLUMNS = [
    "BirthWeight", "FamilyHistory", "PretermBirth", "HeartRate",
    "BreathingDifficulty", "SkinTinge", "Responsiveness", "Movement",
    "DeliveryType", "MothersBPHistory",
]
TARGET_COLUMN = "CardiacArrestChance"
NUM_FEATURES = len(FEATURE_COLUMNS)
NUM_CLASSES = 3
CLASS_NAMES = ["Low", "Medium", "High"]

# --- Categorical Mappings (matches existing 8068/ code exactly) ---
CATEGORY_MAPS = {
    "BirthWeight": {"WeightTooLow": 3, "LowWeight": 2, "NormalWeight": 1},
    "FamilyHistory": {"AboveTwoCases": 3, "ZeroToTwoCases": 2, "NoCases": 1},
    "PretermBirth": {"4orMoreWeeksEarlier": 3, "2To4weeksEarlier": 2, "NotaPreTerm": 1},
    "HeartRate": {"RapidHeartRate": 3, "HighHeartRate": 2, "NormalHeartRate": 1},
    "BreathingDifficulty": {"HighBreathingDifficulty": 3, "BreathingDifficulty": 2, "NoBreathingDifficulty": 1},
    "SkinTinge": {"Bluish": 3, "LightBluish": 2, "NotBluish": 1},
    "Responsiveness": {"UnResponsive": 3, "SemiResponsive": 2, "Responsive": 1},
    "Movement": {"Diminished": 3, "Decreased": 2, "NormalMovement": 1},
    "DeliveryType": {"C_Section": 3, "DifficultDelivery": 2, "NormalDelivery": 1},
    "MothersBPHistory": {"VeryHighBP": 3, "HighBP": 2, "BPInRange": 1},
}
TARGET_MAP = {"High": 2, "Medium": 1, "Low": 0}

# --- Training Defaults ---
DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 0.15
DEFAULT_VAL_SIZE = 0.15
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 20
LR_REDUCE_PATIENCE = 10
LR_REDUCE_FACTOR = 0.5

# --- BERT Configuration ---
BIOBERT_MODEL_NAME = "dmis-lab/biobert-v1.1"
CLINICALBERT_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
BERT_MAX_LENGTH = 128
BERT_EMBEDDING_DIM = 768

# --- HPO Defaults ---
OPTUNA_DEFAULT_TRIALS = 5000
RAY_DEFAULT_TRIALS = 5000
