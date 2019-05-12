from hyperopt import tpe, Trials, STATUS_OK, fmin
import pandas
from testing_xgb_early_stopping import get_num_estimators_xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import xgboost as xgb


class HyperOptimization(object):
    def __init__(self, classifier='xgboost', metrics='auc', verbose=False, plot_auc=False, tpe_algorithm=tpe.suggest):
        self.metrics = metrics
        self.xtrain: pandas.DataFrame = pandas.DataFrame
        self.ytrain: pandas.Series = pandas.Series
        self.params: dict = {}
        self.method = 'cv'
        self.verbose = verbose
        self.plot_ = plot_auc
        self.tpe_algorithm = tpe_algorithm
        self.clf = None
        self.result = pd.DataFrame
        self.classifier = classifier
        self.trials = Trials()

    def save_info(self, space, xtrain, ytrain):
        self.params = space
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.trials = Trials()

    def find(self, space, xtrain, ytrain, max_eval=100):
        self.save_info(space=space, xtrain=xtrain, ytrain=ytrain)
        if self.classifier == 'knn':
            best = fmin(fn=self.knn_train_cv, space=space, algo=tpe.suggest,
                        max_evals=max_eval, trials=self.trials, rstate=np.random.RandomState(50))
        elif self.classifier == 'xgboost':
            best = fmin(fn=self.xgb_train_cv, space=space, algo=tpe.suggest,
                        max_evals=max_eval, trials=self.trials, rstate=np.random.RandomState(50))
        elif self.classifier == 'svm':
            best = fmin(fn=self.svm_train_cv, space=space, algo=tpe.suggest,
                        max_evals=max_eval, trials=self.trials, rstate=np.random.RandomState(50))
        elif self.classifier == 'rf':
            pass
        else:
            assert ValueError, ("Please select a valid classifier to evaluate. The model available are:"
                                " knn, xgboost, rf")
        print(best)
        for key, value in best.items():
            self.params[key] = value
        return self.params

    def rf_train_cv(self, params):
        self.clf = RandomForestClassifier(**params)
        results = cross_val_score(self.clf, self.xtrain, self.ytrain, scoring='roc_auc').mean()
        return {'loss': 1 - results, 'status': STATUS_OK}

    def lr_train_cv(self, params):
        self.clf = LogisticRegression(**params)
        results = cross_val_score(self.clf, self.xtrain, self.ytrain, scoring='roc_auc').mean()
        return {'loss': 1 - results, 'status': STATUS_OK}

    def knn_train_cv(self, params):
        self.clf = KNeighborsClassifier(**params)

        results = cross_val_score(self.clf, self.xtrain, self.ytrain, scoring='roc_auc').mean()
        return {'loss': 1 - results, 'status': STATUS_OK}

    def svm_train_cv(self, params):
        aux_param = dict()
        aux_param['C'] = params['C']
        aux_param['class_weight'] = params['class_weight'] if params['class_weight'] == 'balanced' else None
        aux_param['kernel'] = params['param']['kernel']
        if params['param']['kernel'] == 'rbf':
            aux_param['gamma'] = params['param']['gamma']
        elif params['param']['kernel'] == 'poly':
            aux_param['degree'] = params['param']['degree']
            aux_param['gamma'] = params['param']['gamma']
            aux_param['coef0'] = params['param']['coef0']

        self.clf = SVC(**aux_param)
        results = cross_val_score(self.clf, self.xtrain, self.ytrain, scoring='roc_auc').mean()
        return {'loss': 1 - results, 'status': STATUS_OK}

    def xgb_train_cv(self, params):
        """
        :param params:
        :return:
        """
        # assert type(self.xtrain) == pd.DataFrame, "xtrain must be a pandas DataFrame"
        assert not self.xtrain.shape[0] == 0, "xtrain can't be empty"
        assert type(params) == dict, "params must be a dict type"

        # 1- Select the best number of estimators using Cross Validation function
        xg_train = xgb.DMatrix(self.xtrain, label=self.ytrain.ravel())
        # do cross validation - this going to return the best number of estimators
        aux_params = params
        if params['scale_pos_weight'] == 'balanced':
            aux_params['scale_pos_weight'] = self.ytrain[self.ytrain == 0].shape[0] / self.ytrain[self.ytrain == 1].shape[0]

        results = xgb.cv(aux_params, xg_train, num_boost_round=1000, nfold=10, metrics=self.metrics,
                         early_stopping_rounds=5, stratified=True, seed=1,
                         callbacks=[xgb.callback.print_evaluation(show_stdv=False)])
        print(results)
        # the best number of estimator is exactly the shape of the returned DataFrame from the last function
        self.params['n_estimators'] = results.shape[0]
        auc = results.loc[results.shape[0]-1, 'test-auc-mean']
        return {'loss': 1 - auc, 'status': STATUS_OK}
