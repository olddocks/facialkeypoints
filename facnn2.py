import numpy as np
import pandas as pd
import pylab as pl
from skimage import transform
from skimage import data, io, filter
from skimage.filter import threshold_otsu

from numpy import ravel

from sklearn import linear_model, cross_validation
from sklearn.svm import SVR


df = pd.read_csv('/users/prabhubalakrishnan/Desktop/training.csv', header=0)
df = df.interpolate()

x = df['Image'].values

df = df.drop('Image',axis=1)
y = df.values 
#print 'y orig', y[0]

y = y.astype('float32');

y = y/6

X = []

for k in xrange(len(x)):
 img = np.fromstring(x[k], dtype = np.uint8, sep=' ', count=96*96)
 #thresh = threshold_otsu(img.reshape(96,96))
 #img = img.reshape(96,96) > thresh
 img = filter.sobel(img.reshape(96,96))
 X.append ( ravel(transform.resize (img.reshape(96,96) , (16,16))) )
 
X = np.asarray(X,'float32')

# Scaling 0-1

X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)

print 'X,y shape:', X.shape, y.shape


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.1, random_state=0)


outputs = 30

print 'ytrain', y_train.shape

# Writing training data
f = open("facial-train2.txt", "w")
f.write ( str(X_train.shape[0]) + ' ' + str(X_train.shape[1]) + ' ' + str(outputs) + '\n' )

for k in range(len(X_train)): 
 f.write( ' '.join(map(str, X_train[k] )) + '\n' )     
 f.write( ' '.join(map(str, y_train[k] )) + '\n' ) 

f.close()

# Writing test data

f = open("facial-test2.txt", "w")
f.write ( str(X_test.shape[0]) + ' ' + str(X_test.shape[1]) + ' ' + str(outputs) + '\n' )

for  n in range(len(X_test)):
 f.write( ' '.join(map(str, X_test[n] )) + '\n' )
 f.write( ' '.join(map(str, y_test[n] )) + '\n' )
f.close()