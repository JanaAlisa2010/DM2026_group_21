import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error


def run(train_path='data/train.csv',
        test_path='data/test.csv',
        sample_sub_path='data/sample_submission.csv',
        output_path='submission_v4.csv'):

    print("Loading data...")
    train = pd.read_csv(train_path)
    train['date'] = pd.to_datetime(train['date'], errors='coerce')
    test = pd.read_csv(test_path)
    test['date'] = pd.to_datetime(test['date'], errors='coerce')

    # Drop NaT dates
    test = test.dropna(subset=['date'])
    print(f"Test shape after dropping NaT: {test.shape}")

    # ── Training features ────────────────────────────────────────────────────
    print("Building training features...")
    train_weekly = train[train['score'].notna()].copy()
    train_weekly['month'] = train_weekly['date'].dt.month
    train_weekly['dayofyear'] = train_weekly['date'].dt.dayofyear

    weather_cols = ['prec', 'tmp', 'humidity', 'wind', 'surf_pre',
                    'tmp_max', 'tmp_min', 'surf_tmp', 'dp_tmp', 'wb_tmp']

    # Weekly weather aggregates from train (7 days before each score date)
    print("Aggregating weekly weather for train...")
    train_sorted = train.sort_values(['region_id', 'date'])

    weekly_weather = []
    for region, grp in train_sorted.groupby('region_id'):
        grp = grp.reset_index(drop=True)
        score_rows = grp[grp['score'].notna()]
        for _, row in score_rows.iterrows():
            score_date = row['date']
            week = grp[
                (grp['date'] <= score_date) &
                (grp['date'] > score_date - pd.Timedelta(days=7))
            ]
            if len(week) == 0:
                continue
            feat = {'region_id': region, 'date': score_date}
            for col in weather_cols:
                if col in week.columns:
                    feat[f'{col}_mean'] = week[col].mean()
                    feat[f'{col}_std'] = week[col].std()
            weekly_weather.append(feat)

    weekly_weather_df = pd.DataFrame(weekly_weather)
    print(f"Weekly weather shape: {weekly_weather_df.shape}")

    # Merge with scores
    train_weekly = train_weekly[['region_id', 'date', 'month', 'dayofyear', 'score']].merge(
        weekly_weather_df, on=['region_id', 'date'], how='left'
    )

    # Per-region stats
    region_stats = train_weekly.groupby('region_id')['score'].agg(
        r_mean='mean', r_std='std', r_median='median'
    ).reset_index()

    # Per-region per-month avg
    region_month = train_weekly.groupby(['region_id', 'month'])['score'].mean().reset_index()
    region_month.columns = ['region_id', 'month', 'rm_mean']

    # Last 4 scores per region
    last4 = train_weekly.sort_values('date').groupby('region_id').tail(4)
    last4_avg = last4.groupby('region_id')['score'].mean().reset_index()
    last4_avg.columns = ['region_id', 'last4_avg']

    last1 = train_weekly.sort_values('date').groupby('region_id').tail(1)[['region_id', 'score']]
    last1.columns = ['region_id', 'last_score']

    # Add lag features
    train_weekly = train_weekly.sort_values(['region_id', 'date'])
    for lag in range(1, 5):
        train_weekly[f'score_lag{lag}'] = train_weekly.groupby('region_id')['score'].shift(lag)

    # Merge region stats
    train_weekly = train_weekly.merge(region_stats, on='region_id', how='left')
    train_weekly = train_weekly.merge(region_month, on=['region_id', 'month'], how='left')
    train_weekly = train_weekly.merge(last4_avg, on='region_id', how='left')
    train_weekly = train_weekly.merge(last1, on='region_id', how='left')

    weather_feat_cols = [f'{c}_mean' for c in weather_cols] + [f'{c}_std' for c in weather_cols]
    feature_cols = (
        ['month', 'dayofyear', 'r_mean', 'r_std', 'r_median',
         'rm_mean', 'last4_avg', 'last_score',
         'score_lag1', 'score_lag2', 'score_lag3', 'score_lag4'] +
        [c for c in weather_feat_cols if c in train_weekly.columns]
    )

    X = train_weekly[feature_cols].fillna(0).values
    y = train_weekly['score'].values
    groups = train_weekly['region_id'].values

    # ── Train Model ──────────────────────────────────────────────────────────
    print(f"Training on {len(train_weekly)} samples, {len(feature_cols)} features...")
    gkf = GroupKFold(n_splits=5)
    models = []
    oof_preds = np.zeros(len(train_weekly))

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        model = LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        model.fit(
            X[tr_idx], y[tr_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            callbacks=[
                __import__('lightgbm').early_stopping(50, verbose=False),
                __import__('lightgbm').log_evaluation(200)
            ]
        )
        oof_preds[val_idx] = model.predict(X[val_idx])
        mae = mean_absolute_error(y[val_idx], oof_preds[val_idx])
        print(f"  Fold {fold+1} MAE: {mae:.4f}")
        models.append(model)

    print(f"Overall OOF MAE: {mean_absolute_error(y, oof_preds):.4f}")

    # ── Test Features ────────────────────────────────────────────────────────
    print("Building test features...")

    # Weekly weather for test (last 7 days of each region)
    test_sorted = test.sort_values(['region_id', 'date'])
    test_weather = []
    for region, grp in test_sorted.groupby('region_id'):
        grp = grp.reset_index(drop=True)
        last_date = grp['date'].max()
        week = grp[grp['date'] > last_date - pd.Timedelta(days=7)]
        feat = {'region_id': region, 'last_date': last_date}
        for col in weather_cols:
            if col in week.columns:
                feat[f'{col}_mean'] = week[col].mean()
                feat[f'{col}_std'] = week[col].std()
        test_weather.append(feat)

    test_weather_df = pd.DataFrame(test_weather)

    # ── Generate Submission ──────────────────────────────────────────────────
    print("Generating predictions...")
    global_avg = train_weekly['score'].mean()
    sample = pd.read_csv(sample_sub_path)
    submission_rows = []

    for _, row in sample.iterrows():
        region = row['region_id']
        pred_row = {'region_id': region}

        tw = test_weather_df[test_weather_df['region_id'] == region]
        if len(tw) == 0:
            for i in range(5):
                pred_row[f'pred_week{i+1}'] = global_avg
            submission_rows.append(pred_row)
            continue

        last_date = tw.iloc[0]['last_date']

        # Get region stats
        rs = region_stats[region_stats['region_id'] == region]
        r_mean = rs.iloc[0]['r_mean'] if len(rs) > 0 else global_avg
        r_std = rs.iloc[0]['r_std'] if len(rs) > 0 else 0.3
        r_median = rs.iloc[0]['r_median'] if len(rs) > 0 else global_avg

        la = last4_avg[last4_avg['region_id'] == region]
        l4 = la.iloc[0]['last4_avg'] if len(la) > 0 else global_avg

        ls = last1[last1['region_id'] == region]
        l1 = ls.iloc[0]['last_score'] if len(ls) > 0 else global_avg

        # Last known score lags
        last_scores = train_weekly[train_weekly['region_id'] == region].sort_values('date').tail(4)['score'].values
        lags = list(last_scores[::-1]) + [global_avg] * 4
        lags = lags[:4]

        for i in range(5):
            future_date = last_date + pd.Timedelta(weeks=i+1)
            future_month = future_date.month
            future_doy = future_date.dayofyear

            rm = region_month[
                (region_month['region_id'] == region) &
                (region_month['month'] == future_month)
            ]
            rm_mean = rm.iloc[0]['rm_mean'] if len(rm) > 0 else r_mean

            feat = [
                future_month, future_doy,
                r_mean, r_std, r_median,
                rm_mean, l4, l1,
                lags[0], lags[1], lags[2], lags[3]
            ] + [tw.iloc[0].get(c, 0) for c in
                 [f'{col}_mean' for col in weather_cols] +
                 [f'{col}_std' for col in weather_cols]]

            feat_arr = np.array(feat).reshape(1, -1)
            pred = np.mean([m.predict(feat_arr)[0] for m in models])
            pred_row[f'pred_week{i+1}'] = np.clip(pred, 0.0, 5.0)

            # Update lags for next week
            lags = [pred_row[f'pred_week{i+1}']] + lags[:3]

        submission_rows.append(pred_row)

    submission = pd.DataFrame(submission_rows)
    submission.to_csv(output_path, index=False)

    print(f"\nSaved to {output_path}")
    print(f"Shape: {submission.shape}")
    print(f"Null values: {submission.isnull().sum().sum()}")
    print(submission[['pred_week1','pred_week2','pred_week3','pred_week4','pred_week5']].describe())


if __name__ == '__main__':
    run()