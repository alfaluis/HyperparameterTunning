from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn import datasets
import numpy as np
from hyperoptimization import HyperOptimization
import parameters_config as config

iris = datasets.load_iris()
X = iris.data
y = iris.target
y[y == 2] = 1

tunning_params = HyperOptimization(classifier='knn')
tunning_params.find(space=config.knn_parameters, xtrain=X, ytrain=y)

def hyperopt_train_test(params):
    clf = KNeighborsClassifier(**params)
    clf.fit(X, y)

    results = cross_val_score(clf, X, y, scoring='accuracy').mean()
    return {'loss': 1 - results, 'status': STATUS_OK}


space = {'n_neighbors': hp.choice('n_neighbors', range(1, 100))}
trials = Trials()
best = fmin(fn=hyperopt_train_test, space=space, algo=tpe.suggest,
            max_evals=100, trials=trials, rstate=np.random.RandomState(50))

clf = KNeighborsClassifier(**best)
clf.fit(X, y)
clf.predict(X)

param = {'gamma': hp.loguniform('gamma', 0.1, 2),
         'max_depth':  hp.randint('max_depth', 10),
         'learning_rate': hp.loguniform('learning_rate', 0.001, 3),
         'min_child_weight': hp.randint('min_child_weight', 10),
         'subsample': hp.uniform('subsample', 0.1, 1)}
tunning_params = HyperOptimization(classifier='xgboost')
tunning_params.find(space=config.xgb_parameters, xtrain=X, ytrain=y)
tunning_params = HyperOptimization(classifier='svm')
tunning_params.find(space=config.svm_parameters, xtrain=X, ytrain=y)

