import pandas as pd
import numpy as np


def run(train_path='data/train.csv',
        test_path='data/test.csv',
        sample_sub_path='data/sample_submission.csv',
        output_path='submission_v9.csv'):

    print("Loading data...")
    train = pd.read_csv(train_path)
    train['date'] = pd.to_datetime(train['date'], errors='coerce')
    test = pd.read_csv(test_path)
    test['date'] = pd.to_datetime(test['date'], errors='coerce')
    test = test.dropna(subset=['date'])

    train_weekly = train[train['score'].notna()].copy()
    train_weekly['month'] = train_weekly['date'].dt.month
    train_weekly['week_of_year'] = train_weekly['date'].dt.isocalendar().week.fillna(0).astype(int)

    # Per region + week_of_year average
    region_week = train_weekly.groupby(
        ['region_id', 'week_of_year']
    )['score'].mean().reset_index()
    region_week.columns = ['region_id', 'week_of_year', 'rw_mean']

    # Per region + month average
    region_month = train_weekly.groupby(
        ['region_id', 'month']
    )['score'].mean().reset_index()
    region_month.columns = ['region_id', 'month', 'rm_mean']

    # Per region average
    region_avg = train_weekly.groupby('region_id')['score'].mean().reset_index()
    region_avg.columns = ['region_id', 'r_mean']

    # Last 4 weeks trend
    last4 = train_weekly.sort_values('date').groupby('region_id').tail(4)
    last4_avg = last4.groupby('region_id')['score'].mean().reset_index()
    last4_avg.columns = ['region_id', 'last4_avg']

    # Last 1 score
    last1 = train_weekly.sort_values('date').groupby('region_id').tail(1)[['region_id','score']]
    last1.columns = ['region_id', 'last_score']

    global_avg = train_weekly['score'].mean()
    test_last = test.groupby('region_id')['date'].max().reset_index()
    test_last.columns = ['region_id', 'last_date']

    print("Generating predictions...")
    sample = pd.read_csv(sample_sub_path)
    submission_rows = []

    for _, row in sample.iterrows():
        region = row['region_id']
        pred_row = {'region_id': region}

        tl = test_last[test_last['region_id'] == region]
        if len(tl) == 0:
            for i in range(5):
                pred_row[f'pred_week{i+1}'] = 0.0
            submission_rows.append(pred_row)
            continue

        last_date = tl.iloc[0]['last_date']

        # Get region stats
        r = region_avg[region_avg['region_id'] == region]
        r_mean = r.iloc[0]['r_mean'] if len(r) > 0 else global_avg

        l4 = last4_avg[last4_avg['region_id'] == region]
        l4_val = l4.iloc[0]['last4_avg'] if len(l4) > 0 else 0.0

        l1 = last1[last1['region_id'] == region]
        l1_val = l1.iloc[0]['last_score'] if len(l1) > 0 else 0.0

        # Key insight: if recent trend is 0, predict 0
        # Only predict non-zero if there's recent drought activity
        is_active = l4_val > 0 or l1_val > 0

        for i in range(5):
            future_date = last_date + pd.Timedelta(weeks=i+1)
            future_week = future_date.isocalendar()[1]
            future_month = future_date.month

            # Get seasonal average
            rw = region_week[
                (region_week['region_id'] == region) &
                (region_week['week_of_year'] == future_week)
            ]
            if len(rw) > 0:
                seasonal = rw.iloc[0]['rw_mean']
            else:
                rm = region_month[
                    (region_month['region_id'] == region) &
                    (region_month['month'] == future_month)
                ]
                seasonal = rm.iloc[0]['rm_mean'] if len(rm) > 0 else r_mean

            # Always blend: seasonal history is the main signal
            # Recent trend only matters if drought is active
            recent = l4_val if is_active else 0.0
            pred = 0.5 * seasonal + 0.3 * r_mean + 0.2 * recent

            pred_row[f'pred_week{i+1}'] = float(np.clip(pred, 0.0, 5.0))

        submission_rows.append(pred_row)

    submission = pd.DataFrame(submission_rows)
    submission.to_csv(output_path, index=False)

    print(f"Saved to {output_path}")
    print(f"Shape: {submission.shape}")
    print(f"Null values: {submission.isnull().sum().sum()}")
    print(submission[['pred_week1','pred_week2','pred_week3','pred_week4','pred_week5']].describe())


if __name__ == '__main__':
    run()