#!/usr/bin/env python
# coding=utf8
'''*************************************************************************
    > File Name: utils.py
    > Author: HUANG Yongxiang
    > Mail:
    > Created Time: Sat Mar 10 02:11:12 2018
    > Usage:
*************************************************************************'''

import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
import numpy as np
import itertools

def plot_confusion_matrix(cm, classes,normalize=False, title='Confusion matrix',cmap=plt.cm.Blues):
    if(normalize):
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             fontsize=16,
             color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Ground truth')
    plt.xlabel('Prediction')
    plt.show()

def plot_auc_curve(fpr, tpr, AUC):
    plt.figure()
    lw = 1.5
    plt.plot(fpr, tpr, color='darkorange',
	     lw=lw, label='AUC = %0.3f' % AUC)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    print("confusion_matrix:\r\n",cm2, "\r\n accuaracy of 2-class ",acc2)
    print("AUC="+str(AUC))
