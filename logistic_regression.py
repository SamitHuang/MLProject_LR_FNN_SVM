#!/usr/bin/env python
# coding=utf8
'''*************************************************************************
    > File Name: logistic_regression.py
    > Author:HUANG Yongxiang
    > Mail:
    > Created Time: Wed Mar  7 22:22:40 2018
    > Usage:
*************************************************************************'''

import numpy as np
import argparse
import glob
#import logging
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss,roc_curve, auc, accuracy_score, confusion_matrix
from datetime import datetime


DATA_PATH="../datasets/"

DEBUG_MODE = True

#logger.setLevel(logging.DEBUG)

#params
epoch = 1000
#lr = 0.002 # learning rate
lr = 40e-5 #learning rate
learning_rate = {"breast-cancer":2e-3,"wine":50e-9, "digit":1e-3,"diabetes":16e-4, "iris":40e-5 }

def debug_log( info ):
    if(DEBUG_MODE):
        print("DEBUG:" ,info)

class logistic_regression(object):
    def __init__(self, dataset):
        #self.dataset = dataset
        #print(self.dataset)
        npzfile = np.load(dataset)
        self.trainX, self.trainY= npzfile['train_X'], npzfile['train_Y']
        self.testX, self.testY= npzfile['test_X'], npzfile['test_Y']
        print("Finish loading " + dataset)

        #initial weights with small random value in (-0.01, 0.01)
        #self.W = (np.random.random(self.testX.shape[1]) - 0.5)/50.0
        #self.w0 = (np.random.random(1) -0.5)/50.0
        self.W = np.random.random(self.testX.shape[1])
        self.w0 = np.random.random()

        dataset_name = dataset.split('/')[-1].split(".")[0]
        self.lr = learning_rate[dataset_name]

    def sigmoid(self, x):
        return 1.0/(1.0+np.exp(-x))

    def hypothesis_function(self, X, W, b):
        return self.sigmoid(X @ W.T + b)

    def classifier(self):
        from sklearn.linear_model import SGDClassifier
        clf = SGDClassifier(loss='log', penalty='none', n_iter=epoch, shuffle=False, verbose=True, learning_rate = 'constant', eta0 = self.lr)
        clf.fit(self.trainX, self.trainY)
        predY = clf.predict(self.testX)
        acc=accuracy_score(self.testY, predY)
        print("test accurarcy: ", acc)

    def train(self):
        #parameters
        X = self.trainX
        Y = self.trainY
        testX = self.testX
        testY = self.testY

        loss = np.empty(epoch)
        test_loss = np.empty(epoch)
        start_time = datetime.now()
        for i in range(0, epoch):
            #calc hythoposis values, i.e. predictions
            #theta = X @ self.W.T + self.w0
            #H = self.sigmoid(theta).flatten() # r
            H = self.hypothesis_function(X, self.W, self.w0)

            #calc cross-entropy loss, i.e. log loss
            loss[i] = log_loss(Y, H)
            testH= self.hypothesis_function(testX, self.W, self.w0)
            test_loss[i] = log_loss(testY, testH)
            debug_log("{}\t{}".format(loss[i], test_loss[i]))

            #batch gradient descent
            err = Y-H
            self.W += self.lr * np.sum((err) @ X)
            self.w0 += self.lr * np.sum(err)
        time_used =  datetime.now() - start_time
        print("Finish training for {} epoch in {}".format(epoch, time_used))
        print("Final traing loss: ", loss[epoch-1])
        print("Final test loss: ", test_loss[epoch-1])

    def evaluate(self):
        score = self.hypothesis_function(self.testX, self.W, self.w0)
        predY = (score > 0.5) * 1
        acc=accuracy_score(self.testY, predY)
        print("Classification Accuracy on test set: ", acc)
    # plot loss and test_loss here, report accuracy here
    def plot_loss(train_loss, test_loss):
        pass
    #def validation(self):

if __name__=="__main__":
    # interpret args
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--dataset', type=str, default = None)
    #parser.add_argument('--epoch', type=int, default = 100)
    args = parser.parse_args()
    dataset = args.dataset
    #epoch = args.epoch

    if(dataset!=None):
        LR = logistic_regression(args.dataset)
        LR.train()
        LR.evaluate()

        #LR.classifier()
    else:
        datasets = glob.glob( '../datasets' + '/*.npz')
        for dataset in datasets:
            LR = logistic_regression(dataset)
            LR.train()
