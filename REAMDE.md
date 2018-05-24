Dependency:
    python 3.6
        packages: matplotlib, sklearn

Usage:
For simplicity, you can just copy my "src" folder to the same path with the "datasets" folder (not inside it). Then open "src" folder and run the following command, which will run all models on all datasets for several minutes. 
  
  $  sh run.sh

If you don't want to waste time on running all of them, just open "result_xxx.txt" to read the experiment results.

Or you can run one specific module or dataset as illustrated below.

1) run all models
$ sh run.sh {DATA_PATH}
  DATA_PATH represents the path of the datasets folder or one speciifc dataset. Both is acceptable.
  e.g. sh run.sh ../datasets
    => run all models on all datasets 
  e.g. sh run.sh ../datasets/breast-cancer.npz
    -> run all models on breast cancer dataset

2) run one model seperately
1.1 logistic regression

$ python logistic_regression.py --path={DATA_PATH}

--path: the path of the datasets folder or one speciifc dataset. Both is acceptable. The same as the above python file.

e.g. python logistic_regression.py --path=../datasets/breast-cancer.npz

1.2 neural network
$ python neural_network.py --path={DATA_PATH} --tune={0,1}
--tune: 1, tune parameters with cross-validation and train the model with the best parameters
        0, don't tune and train with the pre-tuned parameters
e.g. python neural_network.py --path=../datasets/iris.npz --tune=1

1.3 SVM
$ python svm.py --path={DATA_PATH} --tune={0,1}
e.g. python svm.py --path=../datasets/iris.npz   --tune=0

