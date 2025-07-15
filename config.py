"""
Configuration settings for the Census Income Prediction project.
"""
import os
from pathlib import Path
from typing import Dict, List, Any

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
REPORTS_DIR = PROJECT_ROOT / "reports"
SRC_DIR = PROJECT_ROOT / "src"
LOGS_DIR = PROJECT_ROOT / "logs"
TESTS_DIR = PROJECT_ROOT / "tests"
API_DIR = PROJECT_ROOT / "api"
DOCKER_DIR = PROJECT_ROOT / "docker"

# Create directories if they don't exist
for dir_path in [LOGS_DIR, REPORTS_DIR, MODELS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Data files
TRAIN_FILE = DATA_DIR / "census_income_learn.csv"
TEST_FILE = DATA_DIR / "census_income_test.csv"
METADATA_FILE = DATA_DIR / "census_income_metadata.txt"

# Column mapping based on metadata analysis
COLUMN_MAPPING = {
    0: 'age',
    1: 'class_of_worker',
    2: 'industry_code',
    3: 'occupation_code',
    4: 'education',
    5: 'wage_per_hour',
    6: 'enrolled_in_edu',
    7: 'marital_status',
    8: 'major_industry_code',
    9: 'major_occupation_code',
    10: 'race',
    11: 'hispanic_origin',
    12: 'sex',
    13: 'union_member',
    14: 'unemployment_reason',
    15: 'employment_status',
    16: 'capital_gains',
    17: 'capital_losses',
    18: 'dividends',
    19: 'tax_filer_status',
    20: 'region',
    21: 'state_code',
    22: 'household_status',
    23: 'household_summary',
    24: 'instance_weight',
    25: 'migration_msa',
    26: 'migration_reg',
    27: 'migration_within_reg',
    28: 'live_here_1_year',
    29: 'migration_sunbelt',
    30: 'num_persons_worked_for_employer',
    31: 'family_members_under_18',
    32: 'country_of_birth_father',
    33: 'country_of_birth_mother',
    34: 'country_of_birth_self',
    35: 'citizenship',
    36: 'own_business_or_self_employed',
    37: 'veterans_admin_questionnaire',
    38: 'veterans_benefits',
    39: 'weeks_worked_in_year',
    40: 'year',
    41: 'income_50k'  # target variable
}

# Target variable
TARGET_COLUMN = "income_50k"
POSITIVE_CLASS = "50000+"
NEGATIVE_CLASS = "-50000"

# Feature categories 
CONTINUOUS_FEATURES = [
    "age", "wage_per_hour", "capital_gains", "capital_losses", 
    "dividends", "num_persons_worked_for_employer", "weeks_worked_in_year"
]

CATEGORICAL_FEATURES = [
    "class_of_worker", "industry_code", "occupation_code",
    "education", "enrolled_in_edu", "marital_status", "major_industry_code",
    "major_occupation_code", "race", "hispanic_origin", "sex", "union_member",
    "unemployment_reason", "employment_status", "tax_filer_status",
    "region", "state_code", "household_status", "household_summary",
    "migration_msa", "migration_reg", "migration_within_reg", 
    "live_here_1_year", "migration_sunbelt", "family_members_under_18",
    "country_of_birth_father", "country_of_birth_mother", "country_of_birth_self",
    "citizenship", "own_business_or_self_employed", 
    "veterans_admin_questionnaire", "veterans_benefits", "year"
]

# Features to ignore 
IGNORE_FEATURES = ["instance_weight"]

# Demographic features for bias analysis
DEMOGRAPHIC_FEATURES = ["age", "race", "sex", "education", "marital_status"]

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Class imbalance parameters
CLASS_WEIGHTS = {
    NEGATIVE_CLASS: 1.0,
    POSITIVE_CLASS: 15.0  # Based on ~6% positive class
}

# Model hyperparameters
MODEL_PARAMS = {
    "logistic_regression": {
        "C": 1.0,
        "class_weight": "balanced",
        "max_iter": 1000,
        "random_state": RANDOM_STATE
    },
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1
    },
    "xgboost": {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 15,
        "random_state": RANDOM_STATE,
        "n_jobs": -1
    },
    "lightgbm": {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 15,
        "random_state": RANDOM_STATE,
        "n_jobs": -1
    },
    "neural_network": {
        "hidden_layer_sizes": (100, 50),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.001,
        "learning_rate": "adaptive",
        "max_iter": 500,
        "random_state": RANDOM_STATE
    }
}

# Evaluation metrics
EVALUATION_METRICS = [
    "accuracy", "precision", "recall", "f1", "roc_auc", "balanced_accuracy"
]

# Fairness metrics
FAIRNESS_METRICS = [
    "demographic_parity_difference",
    "equalized_odds_difference",
    "equal_opportunity_difference"
]

# Production settings
API_HOST = "0.0.0.0"
API_PORT = 8000
MODEL_VERSION = "1.0.0"

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Data processing
CHUNK_SIZE = 10000  # For processing large files
MAX_MEMORY_USAGE = 0.8  # Maximum memory usage for processing

# Feature engineering
MIN_CATEGORY_FREQUENCY = 0.01  # Minimum frequency for rare categories
MAX_CATEGORIES = 50  # Maximum number of categories to keep

# Validation settings
VALIDATION_SPLIT_RATIO = 0.2
EARLY_STOPPING_ROUNDS = 50

# Data leakage detection
LEAKAGE_THRESHOLD = 0.95  # Correlation threshold for potential leakage
LEAKAGE_FEATURES_TO_CHECK = [
    "tax_filer_status", "dividends", "capital_gains", "capital_losses"
]

# Bias analysis settings
BIAS_THRESHOLD = 0.1  # Threshold for bias detection
PROTECTED_ATTRIBUTES = ["race", "sex", "age"]

# Docker settings
DOCKER_IMAGE_NAME = "census-income-prediction"
DOCKER_TAG = "latest"

# API settings
API_TITLE = "Census Income Prediction API"
API_DESCRIPTION = "Production API for predicting income levels based on census data"
API_VERSION = "1.0.0"

# Monitoring settings
MONITORING_METRICS = [
    "prediction_latency", "model_accuracy", "data_drift", "prediction_distribution"
]
ALERT_THRESHOLDS = {
    "accuracy_drop": 0.05,
    "latency_increase": 2.0,
    "data_drift_score": 0.3
}

