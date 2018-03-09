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

#hyperparameters
epoch = 1000
learning_rate = {"breast-cancer":2e-3,"wine":4e-7, "digit":1e-5,"diabetes":1e-3, "iris":40e-5 }

#settings
VALID_INTERVAL = 10

class LogReg(object):
    def __init__(self, dataset):
        npzfile = np.load(dataset)
        self.trainX, self.trainY= npzfile['train_X'], npzfile['train_Y']
        self.testX, self.testY= npzfile['test_X'], npzfile['test_Y']

        self.dataset_name = dataset.split('/')[-1].split(".")[0]
        self.lr = learning_rate[self.dataset_name]

        print("\nDataset: {}".format( self.dataset_name) )
        print("Learning rate: ", self.lr)

    def sigmoid(self, x):
        return 1.0/(1.0+np.exp(-x))

    def lr_func(self, X, W, b):
        z = W @ X.T + b
        return self.sigmoid(z)

    def train(self):
        self.loss_record=[]
        self.acc_record=[]

        #initial weights with small random value in (-0.01, 0.01)
        self.W = (np.random.random(self.testX.shape[1]) - 0.5)/50.0
        self.w0 = (np.random.random(1) -0.5)/50.0

        start_time = datetime.now()
        for i in range(0, epoch):
            #batch gradient descent
            prob = self.lr_func(self.trainX, self.W, self.w0)
            e = self.trainY - prob
            self.W += (e @ self.trainX) * self.lr
            self.w0 += np.sum(e, 0) * self.lr

            #watch and record performance change over time
            if( (i+1) % VALID_INTERVAL== 0):
                '''
                loss_train = log_loss(self.trainY, prob)
                pred = (prob > 0.5) * 1
                acc_train = accuracy_score(self.trainY, pred)
                '''
                # evaluate performance on test set
                loss_test ,acc_test = self.validate()
                self.loss_record.append(loss_test)
                self.acc_record.append(acc_test)

        # evaluate everything needed here
        loss_train = log_loss(self.trainY, prob)
        acc_train = accuracy_score(self.trainY, (prob > 0.5) * 1)

        loss_test ,acc_test = self.validate()

        time = datetime.now() - start_time
        print("Finish training 1000 epoches in ", time)
        print("=> Loss in training set: {:.4f}".format(loss_train))
        print("=> Accuracy in training set: {:.4f}".format(acc_train))
        print("=> Loss in test set: {:.4f}".format(loss_test))
        print("=> Accuracy in test set: {:.4f}".format(acc_test))

        #plot the performance changing curve
        #self.plot_curve(self.loss_record

        #return self.acc_record

    def validate(self):
        #use Z score or prob as decision function
        prob = self.lr_func(self.testX, self.W, self.w0)
        pred = (prob > 0.5) * 1
        loss = log_loss(self.testY, prob)
        acc = accuracy_score(self.testY, pred)
        #print("Loss on test set: ", loss)
        #print("Classification Accuracy on test set: ", acc)

        return loss,acc

def plot_curve( records):
    # plot with various axes scales
    plt.figure("Performance (test accurarcy) change over time during training")
    x = list(range(0,1000,VALID_INTERVAL))

    pos=231
    for dn in records:
        plt.subplot(pos)
        plt.title(dn)
        plt.plot(x, records[dn])
        plt.xlabel("Epoch")
        plt.ylabel("Acc")
        plt.axis([0, 1000, 0, 1])
        plt.grid(True)
        pos+=1
    plt.show()

if __name__=="__main__":
    # interpret args
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--path', type=str, default = "../datasets")
    args = parser.parse_args()

    if("npz" in args.path):
        datasets = [args.path]
    else:
        datasets = glob.glob( args.path + '/*.npz')

    records ={}
    for dataset in datasets:
        LR = LogReg(dataset)
        LR.train()

        records[LR.dataset_name] = LR.acc_record

    plot_curve(records)
