import pandas as pd
import numpy as np

# Reproduce submission_v2 exactly and check predictions
sub2 = pd.read_csv('submission_v2.csv')
sub10 = pd.read_csv('submission_v10.csv')

print("submission_v2 stats:")
print(sub2[['pred_week1','pred_week2','pred_week3','pred_week4','pred_week5']].describe())

print("\nsubmission_v10 stats:")
print(sub10[['pred_week1','pred_week2','pred_week3','pred_week4','pred_week5']].describe())

print("\nDifference (v10 - v2):")
diff = sub10[['pred_week1','pred_week2','pred_week3','pred_week4','pred_week5']].values - sub2[['pred_week1','pred_week2','pred_week3','pred_week4','pred_week5']].values
print(pd.DataFrame(diff).describe())