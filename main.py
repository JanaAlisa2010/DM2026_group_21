import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from src.features import load_data, build_weekly_features, add_lag_features
from src.model import train_model
from src.predict import build_test_features, generate_submission

# ── Paths ──────────────────────────────────────────────────────────────────
TRAIN_PATH      = 'data/train.csv'
TEST_PATH       = 'data/test.csv'
SAMPLE_SUB_PATH = 'data/sample_submission.csv'
MODEL_PATH      = 'models/lgbm_model.pkl'
OUTPUT_PATH     = 'submission.csv'

# ── 1. Load Data ───────────────────────────────────────────────────────────
print("=" * 50)
print("Step 1: Loading data...")
train, test = load_data(TRAIN_PATH, TEST_PATH)
print(f"Train shape: {train.shape}")
print(f"Test shape:  {test.shape}")

# ── 2. Build Features ──────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("Step 2: Building features...")
train_weekly = build_weekly_features(train)
print(f"Train weekly shape: {train_weekly.shape}")

train_weekly = add_lag_features(train_weekly, n_lags=4)
print(f"Train weekly shape with lags: {train_weekly.shape}")

# ── 3. Train Model ─────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("Step 3: Training model...")
best_model, all_models, oof_mae = train_model(
    train_weekly,
    n_splits=5,
    save_path=MODEL_PATH
)

# ── 4. Build Test Features ─────────────────────────────────────────────────
print("\n" + "=" * 50)
print("Step 4: Building test features...")
test_feats = build_test_features(
    test,
    build_weekly_features,
    add_lag_features,
    train_weekly.drop(columns=['split'], errors='ignore')
)

# ── 5. Generate Submission ─────────────────────────────────────────────────
print("\n" + "=" * 50)
print("Step 5: Generating submission...")
submission = generate_submission(
    all_models,
    test_feats,
    SAMPLE_SUB_PATH,
    OUTPUT_PATH
)

print("\n" + "=" * 50)
print(f"Done! OOF MAE: {oof_mae:.4f}")
print("Upload submission.csv to Kaggle!")