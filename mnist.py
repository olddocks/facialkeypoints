import numpy as np
from sklearn import datasets, preprocessing, cross_validation
from numpy import array_str, ravel
from skimage import transform

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')

x,y = mnist.data, mnist.target

x.astype("float32")

X = []
for i in xrange(len(x)):
    X.append ( ravel(transform.resize (x[i].reshape(28,28) , (16,16))) )

X = np.asarray(X, 'float32')

#Grayscale between -1 and 1
#$X = X/255.0*2 - 1

# GRAYSCALE 0-1

X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)

outputs = max(y) + 1

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=0)

ytrain = preprocessing.LabelBinarizer().fit_transform(y_train)
ytest =  preprocessing.LabelBinarizer().fit_transform(y_test)


# Writing training data
f = open("mnist-train.txt", "w")
f.write ( str(X_train.shape[0]) + ' ' + str(X_train.shape[1]) + ' ' + str(outputs) + '\n' )

for k in range(len(X_train)): 
 f.write( ' '.join(map(str, X_train[k] )) + '\n' )     
 f.write( ' '.join(map(str, ytrain[k] )) + '\n' ) 

f.close()

# Writing test data

f = open("mnist-test.txt", "w")
f.write ( str(X_test.shape[0]) + ' ' + str(X_test.shape[1]) + ' ' + str(outputs) + '\n' )

for  n in range(len(X_test)):
 f.write( ' '.join(map(str, X_test[n] )) + '\n' )
 f.write( ' '.join(map(str, ytest[n] )) + '\n' )
f.close()