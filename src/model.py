import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import joblib
import os


def get_feature_columns(df):
    #"""Get all feature columns (exclude meta columns)."""
    exclude = ['region_id', 'date', 'score']
    return [c for c in df.columns if c not in exclude]


def train_model(train_df, n_splits=5, save_path='models/lgbm_model.pkl'):
    """
    Train LightGBM model with GroupKFold cross-validation.
    Groups = region_id to avoid data leakage.
    """
    feature_cols = get_feature_columns(train_df)

    # Drop rows where score is missing
    train_df = train_df[train_df['score'].notna()].copy()

    X = train_df[feature_cols].values
    y = train_df['score'].values
    groups = train_df['region_id'].values

    gkf = GroupKFold(n_splits=n_splits)

    oof_preds = np.zeros(len(train_df))
    models = []
    fold_maes = []

    print(f"Training with {len(feature_cols)} features on {len(train_df)} samples...")
    print(f"Features: {feature_cols[:5]} ...")

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                __import__('lightgbm').early_stopping(50, verbose=False),
                __import__('lightgbm').log_evaluation(100)
            ]
        )

        oof_preds[val_idx] = model.predict(X_val)
        fold_mae = mean_absolute_error(y_val, oof_preds[val_idx])
        fold_maes.append(fold_mae)
        models.append(model)

        print(f"  Fold {fold+1} MAE: {fold_mae:.4f}")

    overall_mae = mean_absolute_error(y, oof_preds)
    print(f"\nOverall OOF MAE: {overall_mae:.4f}")
    print(f"Mean Fold MAE:   {np.mean(fold_maes):.4f}")

    # Save best model (lowest fold MAE)
    best_idx = np.argmin(fold_maes)
    best_model = models[best_idx]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(best_model, save_path)
    print(f"\nBest model saved to {save_path}")

    return best_model, models, overall_mae


def predict(models, test_df, clip_min=0.0, clip_max=5.0):
    """
    Predict using ensemble of models (average).
    Clips predictions to valid score range [0, 5].
    """
    feature_cols = get_feature_columns(test_df)
    X_test = test_df[feature_cols].values

    preds = np.zeros(len(test_df))
    for model in models:
        preds += model.predict(X_test)
    preds /= len(models)

    # Clip to valid range
    preds = np.clip(preds, clip_min, clip_max)

    return preds