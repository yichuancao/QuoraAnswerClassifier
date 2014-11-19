QuoraAnswerClassifier
=====================

This is my implementation and experiment (in progress) with Quora's machine learning challenge question. http://www.quora.com/challenges#answer_classifier
I have already tried k-NN algorithm and it has about 70% to 78% accuracy rate, depending on how many data points, how I choose K, and whether I reduce the dimension of the data. However, it is quite slow without sklearn library. The current stage is to try using perceptron and further speed up data preprocessing.