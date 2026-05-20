import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error


def run(train_path='data/train.csv',
        test_path='data/test.csv',
        sample_sub_path='data/sample_submission.csv',
        output_path='submission_v3.csv'):

    print("Loading data...")
    train = pd.read_csv(train_path)
    train['date'] = pd.to_datetime(train['date'], errors='coerce')
    test = pd.read_csv(test_path)
    test['date'] = pd.to_datetime(test['date'], errors='coerce')

    # Weekly scores only
    train_weekly = train[train['score'].notna()].copy()
    train_weekly['month'] = train_weekly['date'].dt.month

    print(f"Weekly training samples: {len(train_weekly)}")

    # ── Feature Engineering (all vectorized, no loops) ──────────────────────

    print("Computing features...")

    # 1. Per-region statistics
    region_stats = train_weekly.groupby('region_id')['score'].agg(
        r_mean='mean', r_std='std', r_median='median',
        r_min='min', r_max='max'
    ).reset_index()

    # 2. Per-region per-month average
    region_month = train_weekly.groupby(['region_id', 'month'])['score'].mean().reset_index()
    region_month.columns = ['region_id', 'month', 'rm_mean']

    # 3. Weather averages per region
    weather_cols = ['prec', 'tmp', 'humidity', 'wind', 'surf_pre']
    region_weather = train.groupby('region_id')[weather_cols].mean().reset_index()
    region_weather.columns = ['region_id'] + [f'w_{c}' for c in weather_cols]

    # 4. Last 4 weeks average score per region
    last4 = train_weekly.sort_values('date').groupby('region_id').tail(4)
    last4_avg = last4.groupby('region_id')['score'].mean().reset_index()
    last4_avg.columns = ['region_id', 'last4_avg']

    last1 = train_weekly.sort_values('date').groupby('region_id').tail(1)[['region_id','score']]
    last1.columns = ['region_id', 'last_score']

    # 5. Merge all features into train_weekly
    print("Merging features...")
    df = train_weekly[['region_id', 'date', 'month', 'score']].copy()
    df = df.merge(region_stats, on='region_id', how='left')
    df = df.merge(region_month, on=['region_id', 'month'], how='left')
    df = df.merge(region_weather, on='region_id', how='left')
    df = df.merge(last4_avg, on='region_id', how='left')
    df = df.merge(last1, on='region_id', how='left')

    # Week number within sequence (1-5 cycling)
    df['week_num'] = df.groupby('region_id').cumcount() % 5 + 1

    feature_cols = [
        'month', 'week_num',
        'r_mean', 'r_std', 'r_median', 'r_min', 'r_max',
        'rm_mean',
        'w_prec', 'w_tmp', 'w_humidity', 'w_wind', 'w_surf_pre',
        'last4_avg', 'last_score'
    ]

    X = df[feature_cols].fillna(0).values
    y = df['score'].values
    groups = df['region_id'].values

    # ── Train Model ──────────────────────────────────────────────────────────
    print("Training model...")
    gkf = GroupKFold(n_splits=5)
    models = []
    oof_preds = np.zeros(len(df))

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        model = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        model.fit(X[tr_idx], y[tr_idx])
        oof_preds[val_idx] = model.predict(X[val_idx])
        mae = mean_absolute_error(y[val_idx], oof_preds[val_idx])
        print(f"  Fold {fold+1} MAE: {mae:.4f}")
        models.append(model)

    print(f"Overall OOF MAE: {mean_absolute_error(y, oof_preds):.4f}")

    # ── Generate Predictions ─────────────────────────────────────────────────
    print("Generating submission...")
    sample = pd.read_csv(sample_sub_path)
    test_last = test.groupby('region_id')['date'].max().reset_index()
    test_last.columns = ['region_id', 'last_date']

    # Build test feature rows for all 5 weeks at once
    test_records = []
    for _, row in sample.iterrows():
        region = row['region_id']
        last_info = test_last[test_last['region_id'] == region]
        last_date = last_info.iloc[0]['last_date'] if len(last_info) > 0 else pd.Timestamp('2020-01-01')

        for i in range(5):
            future_date = last_date + pd.Timedelta(weeks=i+1)
            test_records.append({
                'region_id': region,
                'month': future_date.month,
                'week_num': i + 1
            })

    test_df = pd.DataFrame(test_records)
    test_df = test_df.merge(region_stats, on='region_id', how='left')
    test_df = test_df.merge(region_month, on=['region_id', 'month'], how='left')
    test_df = test_df.merge(region_weather, on='region_id', how='left')
    test_df = test_df.merge(last4_avg, on='region_id', how='left')
    test_df = test_df.merge(last1, on='region_id', how='left')

    X_test = test_df[feature_cols].fillna(0).values
    preds = np.mean([m.predict(X_test) for m in models], axis=0)
    preds = np.clip(preds, 0.0, 5.0)
    test_df['pred'] = preds

    # Reshape to submission format
    submission_rows = []
    for _, row in sample.iterrows():
        region = row['region_id']
        region_preds = test_df[test_df['region_id'] == region]['pred'].values
        pred_row = {'region_id': region}
        for i in range(5):
            pred_row[f'pred_week{i+1}'] = region_preds[i] if i < len(region_preds) else 0.5
        submission_rows.append(pred_row)

    submission = pd.DataFrame(submission_rows)
    submission.to_csv(output_path, index=False)

    print(f"\nSaved to {output_path}")
    print(f"Shape: {submission.shape}")
    print(f"Null values: {submission.isnull().sum().sum()}")
    print(submission[['pred_week1','pred_week2','pred_week3','pred_week4','pred_week5']].describe())


if __name__ == '__main__':
    run()