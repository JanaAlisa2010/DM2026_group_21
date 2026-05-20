import pandas as pd

train = pd.read_csv('data/train.csv')
train['date'] = pd.to_datetime(train['date'], errors='coerce')
test = pd.read_csv('data/test.csv')
test['date'] = pd.to_datetime(test['date'], errors='coerce')

# Check test data structure
print("Test regions:", test['region_id'].nunique())
print("Test rows per region:")
print(test.groupby('region_id').size().describe())
print("\nFirst region dates:")
print(test[test['region_id'] == test['region_id'].iloc[0]]['date'].sort_values().tail(10))
print("\nLast 3 dates in test:")
print(test.sort_values('date').tail(3)[['region_id','date']])