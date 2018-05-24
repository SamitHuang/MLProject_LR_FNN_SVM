#!/usr/bin/env python
# coding=utf8
'''*************************************************************************
    > File Name: neural_network.py
    > Author: HUANG Yongxiang
    > Mail:
    > Created Time: Thu Mar  8 11:17:26 2018
    > Usage:
*************************************************************************'''
import numpy as np
import argparse
import glob
#import logging
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss,roc_curve, auc, accuracy_score, confusion_matrix

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import operator
from datetime import datetime
from utils import *

#hyperparameters
H_range = list(range(1,11)) # 1,...,10
bestH= {"breast-cancer":2, "wine": 4, "digit": 7, "diabetes": 9, "iris": 3 }
lr = 2e-3   #learning rate
learning_rate = {"breast-cancer":2e-3,"wine":1e-4, "digit":1e-2,"diabetes":2e-3, "iris":1e-3 }

epoch = 2000

#setting
TUNING_MODE =False #True
VERBOSE=False # print the detail info (loss) during training

class feedforward_neural_network(object):
    def __init__(self, dataset, args=None):
        npzfile = np.load(dataset)
        self.trainX, self.trainY= npzfile['train_X'], npzfile['train_Y']
        self.testX, self.testY= npzfile['test_X'], npzfile['test_Y']

        self.dataset_name = dataset.split('/')[-1].split(".")[0]
        self.lr = learning_rate[self.dataset_name]
        self.bestH = bestH[self.dataset_name]

        print("\nDataset: {}".format( self.dataset_name) )
        #print("Learning rate: ", self.lr)

    '''
    params: H_ is the number of hidden units, lr_ is learning rate
    return: the MLPClassifier
    '''
    def classifier(self, H_ =1, lr_=1e-3):
        # batch gradient descent, self.trainY.shape[0]
        # This model optimizes the log-loss function
        clf =  MLPClassifier(hidden_layer_sizes=[H_], activation='logistic', solver='sgd', batch_size=32 , alpha = 0, learning_rate='constant',learning_rate_init =self.lr, max_iter=epoch, shuffle=True, early_stopping=False, validation_fraction=0.2, verbose=VERBOSE)
        return clf

    # train and tune the best H in [1,..,10]
    def train(self):
        print("--- Start training with H = {} ---".format(self.bestH))
        start_time = datetime.now()

        clf = self.classifier(self.bestH, self.lr)
        clf.fit(self.trainX, self.trainY)

        time = datetime.now() - start_time
        print("Finish training in ", time)

        predY_train = clf.predict(self.trainX)
        predY = clf.predict(self.testX)
        acc_train = accuracy_score(self.trainY, predY_train)
        #print("train loss curve: ", clf.loss_curve_, clf.best_loss_)
        print("=> Loss in training set: {:.4f}".format(clf.best_loss_))
        print("=> Accuracy in training set: {:.4f}".format(acc_train))
        #print("test acc:", acc_test)

        # save the well trained classifier
        self.clf = clf

    def tune_using_cross_validation(self):
        bestH, best_score = -1, 0
        mean_score=[0]*len(H_range)

        print("--- Start tuning H using Stratified 5-fold cross_validation ---")
        for H in H_range:
            clf = self.classifier(H, self.lr)
            skf = StratifiedKFold(n_splits=5)
            #"score" here reprent the mean accuracy on the given test data and labels
            scores = cross_val_score(clf, self.trainX, self.trainY, cv=skf, n_jobs=-1, verbose=VERBOSE)
            mean_score[H-1] = np.mean(scores)
            #print("Stratified 5-fold cross-validation scores under H= ", H)
            #print(scores)
            #print("Average score:", mean_score[H-1])
            print("H = {}, average accuracy of 5-folds: {:.4f}".format(H, mean_score[H-1]))
        index, best_score = max(enumerate(mean_score), key=operator.itemgetter(1))
        self.bestH = index + 1
        print("=> Best H is ", self.bestH )

        return bestH

    def evalute(self):
        prob = self.clf.predict_proba(self.testX)
        predY = self.clf.predict(self.testX)
        loss = log_loss(self.testY, prob)
        acc = accuracy_score(self.testY, predY)
        print("=> Loss in test set: {:.4f}".format(loss))
        print("=> Accuracy in test set: {:.4f}".format(acc))
        cm = confusion_matrix(self.testY, predY)
        plot_confusion_matrix(cm, ['Class 0','Class 1'] , title=self.dataset_name)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--path', type=str, default = "../datasets")
    parser.add_argument('--tune', type=int, default = 1)
    args = parser.parse_args()
    #TUNING_MODE = args.tune

    if("npz" in args.path):
        datasets = [args.path]
    else:
        datasets = glob.glob( args.path + '/*.npz')

    for dataset in datasets:
        FNN = feedforward_neural_network(dataset)
        if(args.tune):
            FNN.tune_using_cross_validation()
        FNN.train()
        FNN.evalute()

