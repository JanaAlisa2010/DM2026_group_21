import pandas as pd
import numpy as np


def run(train_path='data/train.csv',
        test_path='data/test.csv',
        sample_sub_path='data/sample_submission.csv',
        output_path='submission_v6.csv'):

    print("Loading data...")
    train = pd.read_csv(train_path)
    train['date'] = pd.to_datetime(train['date'], errors='coerce')
    test = pd.read_csv(test_path)
    test['date'] = pd.to_datetime(test['date'], errors='coerce')
    test = test.dropna(subset=['date'])

    # Get weekly scores only
    train_weekly = train[train['score'].notna()].copy()
    train_weekly['month'] = train_weekly['date'].dt.month
    train_weekly['week_of_year'] = train_weekly['date'].dt.isocalendar().week.fillna(0).astype(int)
    train_weekly['dayofyear'] = train_weekly['date'].dt.dayofyear

    print(f"Weekly training samples: {len(train_weekly)}")

    # ── Key insight: per region + week_of_year average ───────────────────────
    # This captures seasonal patterns per region

    # 1. Per region + week_of_year average (most specific)
    region_week = train_weekly.groupby(
        ['region_id', 'week_of_year']
    )['score'].agg(['mean', 'std', 'count']).reset_index()
    region_week.columns = ['region_id', 'week_of_year', 'rw_mean', 'rw_std', 'rw_count']

    # 2. Per region + month average (fallback)
    region_month = train_weekly.groupby(
        ['region_id', 'month']
    )['score'].mean().reset_index()
    region_month.columns = ['region_id', 'month', 'rm_mean']

    # 3. Per region average (final fallback)
    region_avg = train_weekly.groupby('region_id')['score'].mean().reset_index()
    region_avg.columns = ['region_id', 'r_mean']

    # 4. Last 8 weeks per region (recent trend)
    last8 = train_weekly.sort_values('date').groupby('region_id').tail(8)
    last8_avg = last8.groupby('region_id')['score'].mean().reset_index()
    last8_avg.columns = ['region_id', 'last8_avg']

    # 5. Last 4 weeks per region
    last4 = train_weekly.sort_values('date').groupby('region_id').tail(4)
    last4_avg = last4.groupby('region_id')['score'].mean().reset_index()
    last4_avg.columns = ['region_id', 'last4_avg']

    global_avg = train_weekly['score'].mean()
    print(f"Global average: {global_avg:.4f}")

    # ── Get test last dates ───────────────────────────────────────────────────
    test_last = test.groupby('region_id')['date'].max().reset_index()
    test_last.columns = ['region_id', 'last_date']

    # ── Generate predictions ─────────────────────────────────────────────────
    print("Generating predictions...")
    sample = pd.read_csv(sample_sub_path)

    # Merge all lookups
    test_last = test_last.merge(region_avg, on='region_id', how='left')
    test_last = test_last.merge(last4_avg, on='region_id', how='left')
    test_last = test_last.merge(last8_avg, on='region_id', how='left')

    submission_rows = []

    for _, row in sample.iterrows():
        region = row['region_id']
        pred_row = {'region_id': region}

        tl = test_last[test_last['region_id'] == region]
        if len(tl) == 0:
            for i in range(5):
                pred_row[f'pred_week{i+1}'] = global_avg
            submission_rows.append(pred_row)
            continue

        last_date = tl.iloc[0]['last_date']
        r_mean = tl.iloc[0].get('r_mean', global_avg)
        l4 = tl.iloc[0].get('last4_avg', r_mean)
        l8 = tl.iloc[0].get('last8_avg', r_mean)

        for i in range(5):
            future_date = last_date + pd.Timedelta(weeks=i+1)
            future_week = future_date.isocalendar()[1]
            future_month = future_date.month

            # Try region + week_of_year (most specific)
            rw = region_week[
                (region_week['region_id'] == region) &
                (region_week['week_of_year'] == future_week)
            ]

            if len(rw) > 0 and rw.iloc[0]['rw_count'] >= 3:
                seasonal = rw.iloc[0]['rw_mean']
            else:
                # Fallback: region + month
                rm = region_month[
                    (region_month['region_id'] == region) &
                    (region_month['month'] == future_month)
                ]
                seasonal = rm.iloc[0]['rm_mean'] if len(rm) > 0 else r_mean

            # Blend: 50% seasonal + 30% last4 trend + 20% region avg
            pred = 0.5 * seasonal + 0.3 * l4 + 0.2 * r_mean

            pred_row[f'pred_week{i+1}'] = float(np.clip(pred, 0.0, 5.0))

        submission_rows.append(pred_row)

    submission = pd.DataFrame(submission_rows)
    submission.to_csv(output_path, index=False)

    print(f"\nSaved to {output_path}")
    print(f"Shape: {submission.shape}")
    print(f"Null values: {submission.isnull().sum().sum()}")
    print(submission[['pred_week1','pred_week2','pred_week3','pred_week4','pred_week5']].describe())


if __name__ == '__main__':
    run()