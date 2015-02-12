import pandas as pd

a = pd.read_csv('/users/prabhubalakrishnan/Desktop/result.txt',header=0)
b = pd.read_csv('/users/prabhubalakrishnan/Desktop/IDlookup.csv',header=0)

b = b.drop('Location',axis=1)

merged = b.merge(a,on=['ImageId','FeatureName'])

merged.to_csv('output.csv', cols=['RowId_x','Location'], index=False, index_label='RowId' )