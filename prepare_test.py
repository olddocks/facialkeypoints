import numpy as np
import pandas as pd
import pylab as pl
from skimage import transform
from numpy import ravel

from sklearn import linear_model, cross_validation
from sklearn.svm import SVR


df = pd.read_csv('test.csv', header=0)
#df = df.interpolate()

x = df['Image'].values


X = []

for k in xrange(len(x)):
 img = np.fromstring(x[k], dtype = np.uint8, sep=' ', count=96*96)
 X.append ( ravel(transform.resize (img.reshape(96,96) , (48,48))) )
 
X = np.asarray(X,'float32')

# Scaling 0-1

X = X/256


print 'X shape:', X.shape


outputs = 30


# Writing training data
f = open("ftest.txt", "w")
f.write ( str(X.shape[0]) + ' ' + str(X.shape[1]) + '\n' )

for k in range(len(X)): 
 f.write( ' '.join(map(str, X[k] )) + '\n' )     

f.close()
