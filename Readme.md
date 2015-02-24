#Kaggle Facial Keypoints regression using FANN neural network in C

This code is used to perform simple linear regression to detect 30 facial keypoints on grayscale images using FANN neural network library. Very fast using native c code. 

#What is Needed

FANN library
Python
Numpy/Scipy
Scikit-learn
Pandas

Note: It works in windows, linux and mac

#Description 

 prepare_test.py  -> Dumps the images as training data for FANN to read
 prepare_train.py -> Prepares and dumps the test data to FANN format
 facial.c -> Neural network trainer
 ftest.c -> Testing and predictions (produces results.txt)
 kaggle.py -> Produces kaggle.csv from results.txt (results to upload to kaggle)

# How to Run

First you have to compile all the C code. Make sure you download the training.csv and test.csv from Kaggle facial keypoints project

gcc facial.c -o facial -lfann2 -lm -I /usr/local/include/fann
gcc ftest.c -o ftest -lfann2 -lm -I /usr/local/include/fann

then

python prepare_train.py
python prepare_test.py
./facial
./ftest
python kaggle.py


#More information

http://corpocrat.com/2015/02/24/facial-keypoints-extraction-using-deep-learning-with-caffe/
