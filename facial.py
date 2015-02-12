import numpy as np
import pandas as pd
import pylab as pl
from skimage import transform
from numpy import ravel

from sklearn import linear_model, cross_validation
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.neural_network import BernoulliRBM
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('/users/prabhubalakrishnan/Desktop/training.csv', header=0)


#df = df.drop('Image',axis=1)
#df = df.interpolate(axis=1)

df = df.dropna()

x = df['Image'].values

y = df.values
#print 'y orig', y[0]

#x = x.astype('float32');
#y = y.astype('float32');

#y = y/6
print 'x,y shape:', x.shape, y.shape



X = []

for k in xrange(len(x)):
 img = np.fromstring(x[k], dtype = np.uint8, sep=' ', count=96*96)
 X.append ( ravel(transform.resize (img.reshape(96,96) , (16,16))) )
 
X = np.asarray(X,'float32')

# Scaling 0-1

X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)

#X = X > 0.50

print 'X shape:', X.shape
#print 'y scaled', y[0]


#pca = PCA(n_components=150)
#X = pca.fit_transform(X)


#rbm = BernoulliRBM(n_components=100, n_iter= 100, verbose=True)
#X = rbm.fit_transform(X)

#print 'PCA X shape:', X.shape

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y[:,0], test_size=0.2, random_state=0)
#

#clf = linear_model.LinearRegression()
#clf = DecisionTreeRegressor(max_depth=50)


clf = RandomForestRegressor(n_estimators=100)

clf.fit(X_train, y_train)

print 'Score:', clf.score(X_test, y_test)

#pl.scatter( y[:,0], y[:,1] )
#pl.show()

