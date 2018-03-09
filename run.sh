#!/bin/bash

if [$1 == '']; then
    DATA_PATH="../datasets"
else
    DATA_PATH=$1
fi

echo "\n -------------------------------Model: logistic regression --------------------------"
python logistic_regression.py --path=$DATA_PATH

echo "\n -------------------------------Model: neural network --------------------------"
python neural_network.py --path=$DATA_PATH

echo "\n -------------------------------Model: SVM --------------------------"
python svm.py --path=$DATA_PATH
