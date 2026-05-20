import pandas as pd
import numpy as np


def run(train_path='data/train.csv',
        test_path='data/test.csv',
        sample_sub_path='data/sample_submission.csv',
        output_path='submission_v10.csv'):

    print("Loading data...")
    train = pd.read_csv(train_path)
    train['date'] = pd.to_datetime(train['date'], errors='coerce')
    test = pd.read_csv(test_path)
    test['date'] = pd.to_datetime(test['date'], errors='coerce')
    test = test.dropna(subset=['date'])

    train_weekly = train[train['score'].notna()].copy()
    train_weekly['month'] = train_weekly['date'].dt.month

    # Per region avg (always available for all 2248 regions)
    region_avg = train_weekly.groupby('region_id')['score'].mean()

    # Per region + month avg (only 1596 combos available)
    region_month = train_weekly.groupby(['region_id', 'month'])['score'].mean()

    # Last 4 weeks per region
    last4 = train_weekly.sort_values('date').groupby('region_id').tail(4)
    last4_avg = last4.groupby('region_id')['score'].mean()

    global_avg = train_weekly['score'].mean()
    test_last = test.groupby('region_id')['date'].max()
    sample = pd.read_csv(sample_sub_path)

    print("Generating predictions...")
    rows = []
    for _, row in sample.iterrows():
        region = row['region_id']
        pred_row = {'region_id': region}

        last_date = test_last.get(region, None)
        if last_date is None:
            for i in range(5):
                pred_row[f'pred_week{i+1}'] = global_avg
            rows.append(pred_row)
            continue

        r_mean = region_avg.get(region, global_avg)
        l4 = last4_avg.get(region, r_mean)

        for i in range(5):
            future_date = last_date + pd.Timedelta(weeks=i+1)
            future_month = future_date.month

            # Get region+month avg if available
            if (region, future_month) in region_month.index:
                monthly = region_month[(region, future_month)]
                # Blend: 60% monthly + 25% region avg + 15% recent trend
                pred = 0.60 * monthly + 0.25 * r_mean + 0.15 * l4
            else:
                # Fallback: 70% region avg + 30% recent trend
                pred = 0.70 * r_mean + 0.30 * l4

            pred_row[f'pred_week{i+1}'] = float(np.clip(pred, 0.0, 5.0))

        rows.append(pred_row)

    submission = pd.DataFrame(rows)
    submission.to_csv(output_path, index=False)

    print(f"Saved to {output_path}")
    print(f"Shape: {submission.shape}")
    print(f"Null values: {submission.isnull().sum().sum()}")
    print(submission[['pred_week1','pred_week2','pred_week3','pred_week4','pred_week5']].describe())


if __name__ == '__main__':
    run()