#!/usr/bin/env python
# coding=utf8
'''*************************************************************************
    > File Name: svm.py
    > Author: HUANG Yongxiang
    > Mail:
    > Created Time: Thu Mar  8 17:18:44 2018
    > Usage:
*************************************************************************'''
import numpy as np
import argparse
import glob
#import logging
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, hinge_loss, roc_curve, auc, accuracy_score, confusion_matrix
from sklearn import svm
from sklearn.model_selection import StratifiedKFold, cross_val_score
import operator

#hyperparams
gamma_range = [1, 1e-1, 1e-2, 1e-3]
bestGamma= {"breast-cancer": 0.1, "wine": 0.01, "digit": 0.001, "diabetes":0.1, "iris": 1 }
kernel_type = 'linear'

#setting
epoch = -1
VERBOSE = False
TUNING_MODE = True

class SVM(object):
    def __init__(self, dataset):
        npzfile = np.load(dataset)
        self.trainX, self.trainY= npzfile['train_X'], npzfile['train_Y']
        self.testX, self.testY= npzfile['test_X'], npzfile['test_Y']

        self.dataset_name = dataset.split('/')[-1].split(".")[0]
        self.bestGamma = bestGamma[self.dataset_name]


    def classifier(self, kernel_='linear', gamma_ = 1, verbose_=False):
        #notice: by default, shrinking heuristic is on.
        if(kernel_ == 'linear'):
            # Default, penalty='l2'and C=1  to penalize the slack varaible. dual=True. Setting: hinge loss,
            clf = svm.LinearSVC(loss = "hinge", verbose = verbose_)
        else:
            clf = svm.SVC(kernel=kernel_, gamma = gamma_, verbose=verbose_, max_iter=epoch)
        return clf

    def train(self, kernel_type):
        print("--- Training {} SVM with Gamma = {} ---".format(kernel_type, self.bestGamma))
        # Build the SVM with linear/RBF kernel
        clf = self.classifier(kernel_= kernel_type, gamma_=self.bestGamma, verbose_=VERBOSE)
        clf.fit(self.trainX, self.trainY)

        # Compute the loss of the SVM on the training set and test set
        pred_decision_train = clf.decision_function(self.trainX)
        loss_train = hinge_loss(self.trainY, pred_decision_train)

        pred_decision_test = clf.decision_function(self.testX)
        loss_test = hinge_loss(self.testY, pred_decision_test)
        print("=> Loss in training set: {:.4f}".format(loss_train))
        print("=> Loss in test set: {:.4f}".format(loss_test))

        # Compute the accuray
        predY_train = clf.predict(self.trainX)
        predY = clf.predict(self.testX)
        acc_train = accuracy_score(self.trainY, predY_train)

        predY_test = clf.predict(self.testX)
        predY = clf.predict(self.testX)
        acc_test = accuracy_score(self.testY, predY_test)
        print("=> Accuracy in training set: {:.4f}".format(acc_train))
        print("=> Accuracy in test set: {:.4f}".format( acc_test))
        # save the well trained classifier

        self.clf = clf

    def tune_using_cross_validation(self):
        bestGamma, best_score = -1, 0
        mean_score=[0]*len(gamma_range)
        print("--- Tuning Gamma in rbf SVM with cross_validation ---")
        for idx,gamma in enumerate(gamma_range):
            clf = self.classifier(kernel_='rbf', gamma_ = gamma)
            skf = StratifiedKFold(n_splits=5)
            #TODO: how the scores are calculated? by accuracy?
            scores = cross_val_score(clf, self.trainX, self.trainY, cv=skf, n_jobs=-1, verbose=VERBOSE)
            mean_score[idx] = np.mean(scores)
            #print(scores)
            #print("Average score:", mean_score[idx])
            print("Gamma = {}, average accuracy in 5 folds: {:.4f}".format(gamma, mean_score[idx]))
        index, best_score = max(enumerate(mean_score), key=operator.itemgetter(1))
        self.bestGamma = gamma_range[index]
        print("=> Best gamma is ",self.bestGamma )

        return bestGamma


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--path', type=str, default = "../datasets")
    parser.add_argument('--tune', type=int, default = 1)
    args = parser.parse_args()

    if("npz" in args.path):
        datasets = [args.path]
    else:
        datasets = glob.glob( args.path + '/*.npz')

    for dataset in datasets:
        print("\nDataset: {}".format(dataset.split('/')[-1].split(".")[0] ))
        # Build the SVM with linear kernel
        linearSVM= SVM(dataset)
        linearSVM.train(kernel_type="linear")

        # Build SVM with RBF kernel
        rbfSVM= SVM(dataset)
        if(args.tune):
            rbfSVM.tune_using_cross_validation()
        rbfSVM.train(kernel_type="rbf")


