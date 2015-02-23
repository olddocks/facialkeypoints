import pandas as pd

a = pd.read_csv('result.txt',header=0)
b = pd.read_csv('IDlookup.csv',header=0)

b = b.drop('Location',axis=1)

merged = b.merge(a,on=['ImageId','FeatureName'])

merged.to_csv('kaggle.csv', cols=['RowId_x','Location'], index=False, index_label='RowId' )