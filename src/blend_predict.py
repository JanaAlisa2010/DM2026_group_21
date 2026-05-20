import pandas as pd
import numpy as np

train = pd.read_csv('data/train.csv')
train['date'] = pd.to_datetime(train['date'], errors='coerce')
test = pd.read_csv('data/test.csv')
test['date'] = pd.to_datetime(test['date'], errors='coerce')

train_weekly = train[train['score'].notna()].copy()
train_weekly['month'] = train_weekly['date'].dt.month

# Per region per month average
region_month = train_weekly.groupby(['region_id','month'])['score'].mean()

# Last 4 weeks average per region
last4 = train_weekly.sort_values('date').groupby('region_id').tail(4)
last4_avg = last4.groupby('region_id')['score'].mean()

# Global fallback
global_avg = train_weekly['score'].mean()

test_last = test.groupby('region_id')['date'].max()
sample = pd.read_csv('data/sample_submission.csv')

rows = []
for _, row in sample.iterrows():
    region = row['region_id']
    last_date = test_last.get(region, pd.Timestamp('2020-01-01'))
    pred_row = {'region_id': region}
    
    for i in range(5):
        future_month = (last_date + pd.Timedelta(weeks=i+1)).month
        
        # Weighted blend: 50% monthly avg + 50% last4 avg
        monthly = region_month.get((region, future_month), global_avg)
        recent = last4_avg.get(region, global_avg)
        pred = 0.5 * monthly + 0.5 * recent
        
        pred_row[f'pred_week{i+1}'] = np.clip(pred, 0.0, 5.0)
    rows.append(pred_row)

submission = pd.DataFrame(rows)
submission.to_csv('submission_v3.csv', index=False)
print("Done!")
print(f"Null values: {submission.isnull().sum().sum()}")
print(submission[['pred_week1','pred_week2','pred_week3','pred_week4','pred_week5']].describe())