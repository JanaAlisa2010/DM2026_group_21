import pandas as pd
import numpy as np


def load_data(train_path, test_path):
    print("Loading train...")
    train = pd.read_csv(train_path)
    train['date'] = pd.to_datetime(train['date'], errors='coerce')
    
    print("Loading test...")
    test = pd.read_csv(test_path)
    test['date'] = pd.to_datetime(test['date'], errors='coerce')
    
    return train, test


def build_weekly_features(df):
    """
    Aggregate daily meteorological data into weekly features.
    Each week = 7 days ending on the score date.
    """
    feature_cols = [
        'prec', 'surf_pre', 'humidity', 'tmp', 'dp_tmp', 'wb_tmp',
        'tmp_max', 'tmp_min', 'tmp_range', 'surf_tmp',
        'wind', 'wind_max', 'wind_min', 'wind_range'
    ]

    records = []
    regions = df['region_id'].unique()
    total = len(regions)

    for i, region in enumerate(regions):
        if i % 100 == 0:
            print(f"  Processing region {i+1}/{total}...")

        group = df[df['region_id'] == region].sort_values('date').reset_index(drop=True)

        # Get score rows (weekly labels)
        if 'score' in group.columns:
            score_rows = group[group['score'].notna()].copy()
        else:
            # For test data, treat every 7th row as week end
            score_rows = group.iloc[6::7].copy()

        for idx, row in score_rows.iterrows():
            score_date = row['date']

            # Get the 7 days up to and including score_date
            week_data = group[
                (group['date'] <= score_date) &
                (group['date'] > score_date - pd.Timedelta(days=7))
            ]

            if len(week_data) == 0:
                continue

            feat = {'region_id': region, 'date': score_date}

            for col in feature_cols:
                if col in week_data.columns:
                    feat[f'{col}_mean'] = week_data[col].mean()
                    feat[f'{col}_max'] = week_data[col].max()
                    feat[f'{col}_min'] = week_data[col].min()
                    feat[f'{col}_std'] = week_data[col].std()

            if 'score' in group.columns:
                feat['score'] = row['score']

            records.append(feat)

    return pd.DataFrame(records)


def add_lag_features(df, n_lags=4):
    """
    Add lag features: previous weeks scores and weather per region.
    """
    df = df.sort_values(['region_id', 'date']).reset_index(drop=True)

    lag_cols = ['score'] + [c for c in df.columns if c.endswith('_mean')]

    for col in lag_cols:
        if col not in df.columns:
            continue
        for lag in range(1, n_lags + 1):
            df[f'{col}_lag{lag}'] = df.groupby('region_id')[col].shift(lag)

    return df