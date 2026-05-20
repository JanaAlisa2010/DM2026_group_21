import pandas as pd
import numpy as np


def build_test_features(test_df, feature_builder, lag_builder, train_weekly):
    """
    Build features for test data using the last 91 days per region.
    Then append to train_weekly to compute lag features correctly.
    """
    print("Building test weekly features...")
    test_weekly = feature_builder(test_df)

    print(f"Test weekly shape: {test_weekly.shape}")

    # Combine train and test weekly for lag computation
    train_weekly = train_weekly.copy()
    train_weekly['split'] = 'train'
    test_weekly['split'] = 'test'

    combined = pd.concat([train_weekly, test_weekly], ignore_index=True)
    combined = combined.sort_values(['region_id', 'date']).reset_index(drop=True)

    # Add lag features on combined data
    combined = lag_builder(combined)

    # Split back
    test_feats = combined[combined['split'] == 'test'].copy()
    test_feats = test_feats.drop(columns=['split'])

    return test_feats


def generate_submission(models, test_feats, sample_submission_path, output_path='submission.csv'):
    """
    Generate submission.csv in the required format:
    region_id, pred_week1, pred_week2, pred_week3, pred_week4, pred_week5
    """
    from src.model import get_feature_columns

    print("Generating predictions...")

    feature_cols = get_feature_columns(test_feats)

    # Fill NaN with 0 before predicting
    X_test = test_feats[feature_cols].fillna(0).values

    preds = np.zeros(len(test_feats))
    for model in models:
        preds += model.predict(X_test)
    preds /= len(models)
    preds = np.clip(preds, 0.0, 5.0)

    test_feats = test_feats.copy()
    test_feats['pred'] = preds

    # Sort by region and date to get weeks in order
    test_feats = test_feats.sort_values(['region_id', 'date']).reset_index(drop=True)

    # Load sample submission to get all region_ids
    sample = pd.read_csv(sample_submission_path)

    submission_rows = []

    for region in sample['region_id']:
        group = test_feats[test_feats['region_id'] == region].reset_index(drop=True)

        row = {'region_id': region}

        if len(group) == 0:
            # Region not found in test features - use global mean
            for i in range(5):
                row[f'pred_week{i+1}'] = preds.mean()
        else:
            for i in range(min(5, len(group))):
                row[f'pred_week{i+1}'] = group.loc[i, 'pred']

            # Fill missing weeks with last prediction
            if len(group) < 5:
                last_pred = group.loc[len(group)-1, 'pred']
                for i in range(len(group), 5):
                    row[f'pred_week{i+1}'] = last_pred

        submission_rows.append(row)

    submission = pd.DataFrame(submission_rows)

    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(f"Shape: {submission.shape}")
    print(f"Null values: {submission.isnull().sum().sum()}")
    print(submission.head())

    return submission