#!/usr/bin/python
import re
import sys
import fileinput
import math
import operator
import random
from random import choice
#from numpy import random
#from sklearn import preprocessing
#from sklearn.neighbors import KNeighborsClassifier

# Note: perceptron seems to perform worse in accuracy than k-NN but significantly faster

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

# used for dimension reduction (MANUALLY. Not good enough. Should do a cross-validation later on.)
remove = {'1': True,
        '2': False,
        '3': False,
        '4': False,
        '5': False,
        '6': False,
        '7': False,
        '8': False,
        '9': True,
        '10': False,
        '11': False,
        '12': False,
        '13': False,
        '14': False,
        '15': False,
        '16': True,
        '17': False,
        '18': True,
        '19': True,
        '20': False,
        '21': False,
        '22': True,
        '23': True}
# data preprocessing (dimension reduction but decided not to normalize to L2 norm for perceptron)
# some commented code is from the experiment with k-NN algorithm
#p_count = 0
trainingData = []
testingData = []
for k in range(N):
    features = []
    #sum = 0.0
    if random.uniform(0.0, 1.0) >= 0.1: # sample ~10% of training data
        continue
    list_data = train[k].split()
    for i in range(M):
        list_feature = list_data[i + 2].split(':')
        leave_out = remove[list_feature[0]]
        if leave_out == False:
            value = float(list_feature[1])    
            features.append(value)
    #        sum = float(sum) + value**2
    #features = [x / float(math.sqrt(sum)) for x in features]
    #reduced_feature_dimension = len(features)
    t_label = list_data[1]
    #if t_label == '+1':
    #    p_count += 1
    instance = Instance(list_data[0], t_label, features)
    trainingData.append(instance)
for k in range(n):
    features = []
    #sum = 0.0
    list_data = test[k].split()
    for i in range(M):
        list_feature = list_data[i + 1].split(':')
        leave_out = remove[list_feature[0]]
        if leave_out == False:
            value = float(list_feature[1])
            features.append(value)
    #        sum = float(sum) + value**2
    #features = [x / float(math.sqrt(sum)) for x in features]
    instance = Instance(list_data[0], '', features)
    testingData.append(instance)

# Initialize random weights to start
w = []
reduced_dimension = len(trainingData[0].getFeatures())
for k in range(reduced_dimension):
    w.append(random.uniform(-1.0, 1.0)) 
#print w

# Perceptron Learning Algorithm
run = True
while run == True:
    mistake = 0
    for i in range(len(trainingData)):
        sum = 0.0
        for k in range(reduced_dimension):
            sum += w[k] * trainingData[i].getFeatures()[k]
        #print sum
        label = int(trainingData[i].getLabel())
        #print sum, label
        if sum * label < 0:
            mistake += 1
            for j in range(reduced_dimension):
                w[j] = w[j] + trainingData[i].getFeatures()[j] * label
    #print i, len(trainingData)
    #print mistake
    if i == len(trainingData) - 1 and mistake < len(trainingData) * (1.0 - 0.735): # STOP threshold (Not perfect. 0.735 came from playing with feature selection)
        run = False
#print w

# Classify
for i in range(n):
    sum = 0.0
    for k in range(reduced_dimension):
        sum += w[k] * testingData[i].getFeatures()[k]
    if sum < 0.0:
        print testingData[i].getAnsID(), "-1"
    else:
        print testingData[i].getAnsID(), "+1"

# ------ working KNN without Numpy, Sklearn (Caveat: TOO SLOW on large input [40,000+ training data points]) ----------
## distance
#def euclideanDistance(training_instance, testing_instance):
#    distance = 0.0
#    for x in range(0, reduced_feature_dimension):
#        distance = distance + (training_instance[x] - testing_instance[x])**2
#    return math.sqrt(distance)

## find k neighbors
#def getNeighbors(training_set, testing_instance, k):
#    distances = []
#    for x in range(0, N):
#        dist = euclideanDistance(training_set[x].getFeatures(), testing_instance)
#        distances.append((training_set[x], dist))
#    distances.sort(key=operator.itemgetter(1))
#    neighbors = []
#    for x in range(k):
#        neighbors.append(distances[x][0])
#    return neighbors

## set k
#k = 5

## find prediction label
#def majorityVote(neighbors):
#    positive_count = 0
#    negative_count = 0
#    retVal = ''
#    for i in range(k):
#        if neighbors[i].getLabel() == '+1':
#            positive_count += 1
#        else:
#            negative_count += 1
#    if positive_count > negative_count:
#        retVal = '+1'
#    elif positive_count < negative_count:
#        retVal = '-1'
#    else:
#        if p_count >= (N - p_count):
#            retVal = '+1'
#        else:
#            retVal = '-1'
#    return retVal

#def classify(training_set, testing_instance, k):
#    neighbors = getNeighbors(training_set, testing_instance, k)
#    prediction = majorityVote(neighbors)
#    return prediction

## Classification
#for i in range(0, n):
#    result = classify(trainingData, testingData[i].getFeatures(), k)
#    print testingData[i].getAnsID(), result

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