#!/usr/bin/python
import re
import sys
import fileinput
import math
import operator
from random import choice
#from sklearn import preprocessing
#from sklearn.neighbors import KNeighborsClassifier

class Instance:
    def __init__(self, ans_id, label, feature):
        self.ans_id = ans_id
        self.label = label
        self.feature = feature
    def setFeatures(self, feature):
        self.feature = feature
    def getAnsID(self):
        return self.ans_id
    def getLabel(self):
        return self.label
    def getFeatures(self):
        return self.feature

# load data from STDIN
data = []
for line in fileinput.input():
    data.append(line)
N, M = map(int, re.findall(r'\d+', data[0])) # N = number of training data, M = number of dimension
train = data[1:N+1]
n = int(data[N+1]) # n = number of testing data
test = data[(N+2):]

# dimension reduction (MANUALLY)
remove = {'1': True,
          '2': False,
          '3': True,
          '4': False,
          '5': False, 
          '6': True, 
          '7': True, 
          '8': True, 
          '9': True, 
          '10': True, 
          '11': False,
          '12': True, 
          '13': True, 
          '14': True, 
          '15': True, 
          '16': True, 
          '17': True, 
          '18': True, 
          '19': True,
          '20': False,
          '21': False, 
          '22': True, 
          '23': True}
p_count = 0
trainingData = []
testingData = []
for k in range(N):
    features = []
    for i in range(M):
        leave_out = remove[train[k].split()[i + 2].split(':')[0]]
        if leave_out == False:    
            features.append(float(train[k].split()[i + 2].split(':')[1]))
        reduced_feature_dimension = len(features)
        t_label = train[k].split()[1]
        if t_label == '+1':
            p_count += 1
    instance = Instance(train[k].split()[0], t_label, features)
    trainingData.append(instance)
for k in range(n):
    features = []
    for i in range(M):
        leave_out = remove[test[k].split()[i + 1].split(':')[0]]
        if leave_out == False:    
            features.append(float(test[k].split()[i + 1].split(':')[1]))
    instance = Instance(test[k].split()[0], '', features)
    testingData.append(instance)

# Normalization
for i in range(0, N):
    sum = 0.0
    for j in range(0, reduced_feature_dimension):
        sum = float(sum) + trainingData[i].getFeatures()[j]**2
    features = []
    for k in range(0, reduced_feature_dimension):
        features.append(float(trainingData[i].getFeatures()[k]) / float(math.sqrt(sum)))
    trainingData[i].setFeatures(features)
for i in range(0, n):
    sum = 0.0
    for j in range(0, reduced_feature_dimension):
        sum = float(sum) + testingData[i].getFeatures()[j]**2
    features = []
    for k in range(0, reduced_feature_dimension):
        features.append(float(testingData[i].getFeatures()[k]) / float(math.sqrt(sum)))
    testingData[i].setFeatures(features)

# Perceptron Learning Algorithm
# TODO

# ------ working KNN without Numpy, Sklearn (Caveat: TOO SLOW on large input [40,000+ training data points]) ----------
# distance
def euclideanDistance(training_instance, testing_instance):
    distance = 0.0
    for x in range(0, reduced_feature_dimension):
        distance = distance + (training_instance[x] - testing_instance[x])**2
    return math.sqrt(distance)

# find k neighbors
def getNeighbors(training_set, testing_instance, k):
    distances = []
    for x in range(0, N):
        dist = euclideanDistance(training_set[x].getFeatures(), testing_instance)
        distances.append((training_set[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

# set k
k = 5

# find prediction label
def majorityVote(neighbors):
    positive_count = 0
    negative_count = 0
    retVal = ''
    for i in range(k):
        if neighbors[i].getLabel() == '+1':
            positive_count += 1
        else:
            negative_count += 1
    if positive_count > negative_count:
        retVal = '+1'
    elif positive_count < negative_count:
        retVal = '-1'
    else:
        if p_count >= (N - p_count):
            retVal = '+1'
        else:
            retVal = '-1'
    return retVal

def classify(training_set, testing_instance, k):
    neighbors = getNeighbors(training_set, testing_instance, k)
    prediction = majorityVote(neighbors)
    return prediction

# Classification
for i in range(0, n):
    result = classify(trainingData, testingData[i].getFeatures(), k)
    print testingData[i].getAnsID(), result

# --------------------- working KNN without Numpy, Sklearn (TOO SLOW) ----------------------

# --------------------- working KNN with sklearn --------------------
## KNN with K = 15
## Normalizing features
#X_train = []
#y_train = []
#X_test = []
#for i in range(0, N):
#    X_train.append(trainingData[i].getFeatures())
#    y_train.append(int(trainingData[i].getLabel()))
#for i in range(0, n):
#    X_test.append(testingData[i].getFeatures())

#X_train_normalized = preprocessing.normalize(X_train, norm='l2')
#X_test_normalized = preprocessing.normalize(X_test, norm='l2')

## KNN with k = 15
#knn = KNeighborsClassifier(n_neighbors=15)
#knn.fit(X_train_normalized, y_train)
#for i in range(0, n):
#    output = knn.predict(X_test_normalized[i])[0]
#    if output == 1:
#        print testingData[i].getAnsID(), '+1'
#    else:
#        print testingData[i].getAnsID(), '-1'
# -----------------------------------------------------------------------