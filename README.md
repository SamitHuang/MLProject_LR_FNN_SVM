Dependency:
    python 3.6
        important packages: matplotlib, sklearn

Usage:

1) For simplicity, just open "code" folder and run the following command, which will run all models on all datasets for several minutes. (Make sure "datasets" folder is in the same directory with "code" folder. I have done it)
  
  $  sh run.sh
or, turn off parameter tunning which will save your time.
  $ sh run.sh --tune=0 

If you don't want to spend time on running all of datasets and models, feel free to open "result_xxx.txt" to read the experiment results.
Or you can run one specific module or dataset as illustrated below.

2) run one model seperately
2.1 logistic regression

$ python logistic_regression.py --path={DATA_PATH}

--path: the path of the datasets folder or one speciifc dataset. Both is acceptable. The same as the above python file.

e.g. python logistic_regression.py --path=../datasets/breast-cancer.npz

2.2 neural network
$ python neural_network.py --path={DATA_PATH} --tune={0,1}
--tune: 1, tune parameters with cross-validation and train the model with the best parameters
        0, don't tune and train with the pre-tuned parameters

e.g. python neural_network.py --path=../datasets/iris.npz --tune=1

2.3 SVM
$ python svm.py --path={DATA_PATH} --tune={0,1}
e.g. python svm.py --path=../datasets/iris.npz   --tune=0

