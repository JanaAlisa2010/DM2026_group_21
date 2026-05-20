import pandas as pd
import numpy as np


def run(train_path='data/train.csv',
        test_path='data/test.csv',
        sample_sub_path='data/sample_submission.csv',
        output_path='submission_v12.csv'):

    print("Loading data...")
    train = pd.read_csv(train_path)
    train['date'] = pd.to_datetime(train['date'], errors='coerce')
    test = pd.read_csv(test_path)
    test['date'] = pd.to_datetime(test['date'], errors='coerce')

    # Drop NaT BEFORE computing last dates
    test = test.dropna(subset=['date'])

    train_weekly = train[train['score'].notna()].copy()
    train_weekly['month'] = train_weekly['date'].dt.month
    train_weekly['week_of_year'] = train_weekly['date'].dt.isocalendar().week.fillna(0).astype(int)

    # Per region + month
    region_month = train_weekly.groupby(['region_id', 'month'])['score'].mean()

    # Per region + week
    region_week = train_weekly.groupby(['region_id', 'week_of_year'])['score'].mean()

    # Per region avg
    region_avg = train_weekly.groupby('region_id')['score'].mean()

    global_avg = train_weekly['score'].mean()

    # Get last date per region AFTER dropping NaT
    test_last = test.groupby('region_id')['date'].max()

    sample = pd.read_csv(sample_sub_path)

    print("Generating predictions...")
    rows = []
    for _, row in sample.iterrows():
        region = row['region_id']
        pred_row = {'region_id': region}
        r_mean = float(region_avg.get(region, global_avg))

        if region not in test_last.index:
            for i in range(5):
                pred_row[f'pred_week{i+1}'] = r_mean
            rows.append(pred_row)
            continue

        last_date = test_last[region]

        if pd.isna(last_date):
            for i in range(5):
                pred_row[f'pred_week{i+1}'] = r_mean
            rows.append(pred_row)
            continue

        for i in range(5):
            future_date = last_date + pd.Timedelta(weeks=i+1)
            future_month = int(future_date.month)
            future_week = int(future_date.isocalendar()[1])

            if (region, future_week) in region_week.index:
                weekly = float(region_week[(region, future_week)])
                monthly = float(region_month.get((region, future_month), r_mean))
                pred = 0.6 * weekly + 0.25 * monthly + 0.15 * r_mean
            elif (region, future_month) in region_month.index:
                monthly = float(region_month[(region, future_month)])
                pred = 0.5 * monthly + 0.5 * r_mean
            else:
                pred = r_mean

            pred_row[f'pred_week{i+1}'] = float(np.clip(pred, 0.0, 5.0))

        rows.append(pred_row)

    submission = pd.DataFrame(rows)
    submission.to_csv(output_path, index=False)

    print(f"Saved to {output_path}")
    print(f"Null values: {submission.isnull().sum().sum()}")
    print(submission[['pred_week1','pred_week2','pred_week3','pred_week4','pred_week5']].describe())


if __name__ == '__main__':
    run()