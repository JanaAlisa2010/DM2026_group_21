import pandas as pd
import numpy as np


def seasonal_average_submission(train_path, test_path, sample_sub_path, output_path='submission_v2.csv'):
    """
    Strategy: For each region, predict the average score
    for the corresponding month/season from training history.
    """
    print("Loading data...")
    train = pd.read_csv(train_path)
    train['date'] = pd.to_datetime(train['date'], errors='coerce')

    test = pd.read_csv(test_path)
    test['date'] = pd.to_datetime(test['date'], errors='coerce')

    # Get weekly scores from train
    train_weekly = train[train['score'].notna()].copy()
    train_weekly['month'] = train_weekly['date'].dt.month
    train_weekly['week_of_year'] = train_weekly['date'].dt.isocalendar().week.fillna(0).astype(int)
    # Compute per-region per-month average score
    region_month_avg = train_weekly.groupby(
        ['region_id', 'month']
    )['score'].mean().reset_index()
    region_month_avg.columns = ['region_id', 'month', 'avg_score']

    # Compute global per-region average as fallback
    region_avg = train_weekly.groupby('region_id')['score'].mean().reset_index()
    region_avg.columns = ['region_id', 'global_avg']

    # Global fallback
    global_avg = train_weekly['score'].mean()
    print(f"Global average score: {global_avg:.4f}")

    # For each region in test, find the last date and predict next 5 weeks
    test_last = test.groupby('region_id')['date'].max().reset_index()
    test_last.columns = ['region_id', 'last_date']

    sample = pd.read_csv(sample_sub_path)
    submission_rows = []

    for _, row in sample.iterrows():
        region = row['region_id']
        pred_row = {'region_id': region}

        # Get last date for this region
        last_info = test_last[test_last['region_id'] == region]
        if len(last_info) == 0:
            for i in range(5):
                pred_row[f'pred_week{i+1}'] = global_avg
            submission_rows.append(pred_row)
            continue

        last_date = last_info.iloc[0]['last_date']

        # Predict 5 weeks ahead
        for i in range(5):
            future_date = last_date + pd.Timedelta(weeks=i+1)
            future_month = future_date.month

            # Try region + month average
            match = region_month_avg[
                (region_month_avg['region_id'] == region) &
                (region_month_avg['month'] == future_month)
            ]

            if len(match) > 0:
                pred = match.iloc[0]['avg_score']
            else:
                # Fallback: region average
                reg_match = region_avg[region_avg['region_id'] == region]
                if len(reg_match) > 0:
                    pred = reg_match.iloc[0]['global_avg']
                else:
                    pred = global_avg

            pred_row[f'pred_week{i+1}'] = np.clip(pred, 0.0, 5.0)

        submission_rows.append(pred_row)

    submission = pd.DataFrame(submission_rows)
    submission.to_csv(output_path, index=False)

    print(f"Saved to {output_path}")
    print(f"Shape: {submission.shape}")
    print(f"Null values: {submission.isnull().sum().sum()}")
    print(submission.describe())

    return submission


if __name__ == '__main__':
    seasonal_average_submission(
        train_path='data/train.csv',
        test_path='data/test.csv',
        sample_sub_path='data/sample_submission.csv',
        output_path='submission_v2.csv'
    )