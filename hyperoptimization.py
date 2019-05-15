from hyperopt import tpe, Trials, STATUS_OK, fmin
import pandas
from testing_xgb_early_stopping import get_num_estimators_xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import xgboost as xgb


class HyperOptimization(object):
    def __init__(self, classifier='xgboost', metrics='roc_auc', params=dict(), verbose=False,
                 plot_auc=False, tpe_algorithm=tpe.suggest, max_eval=10):
        self.metrics = metrics
        self.xtrain: pandas.DataFrame = pandas.DataFrame
        self.ytrain: pandas.Series = pandas.Series
        self.params: dict = params
        self.verbose = verbose
        self.plot_ = plot_auc
        self.tpe_algorithm = tpe_algorithm
        self.clf = None
        self.result = pd.DataFrame
        self.classifier = classifier
        self.trials = Trials()
        self.max_eval = max_eval

    def _save_info(self, xtrain, ytrain):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.trials = Trials()

    def gradient_opt(self, space, xtrain, ytrain):
        """ Function to init the optimization process for the given classifier
        :param space: A dictionary with the parameters to be explorer
        :param xtrain: DataSet with training feature examples. Can be Array or DataFrame
        :param ytrain: DataSet with targets. Can be Array or DataFrame
        :return: A dictionary with the best parameters found
        """
        self._save_info(xtrain=xtrain, ytrain=ytrain)
        if self.classifier == 'knn':
            best = fmin(fn=self._knn_train_cv, space=space, algo=tpe.suggest,
                        max_evals=self.max_eval, trials=self.trials, rstate=np.random.RandomState(50))
        elif self.classifier == 'xgboost':
            best = fmin(fn=self._xgb_train_cv, space=space, algo=tpe.suggest,
                        max_evals=self.max_eval, trials=self.trials, rstate=np.random.RandomState(50))
        elif self.classifier == 'svm':
            best = fmin(fn=self._svm_train_cv, space=space, algo=tpe.suggest,
                        max_evals=self.max_eval, trials=self.trials, rstate=np.random.RandomState(50))
        elif self.classifier == 'linear_svm':
            best = fmin(fn=self._linear_svm_train_cv, space=space, algo=tpe.suggest,
                        max_evals=self.max_eval, trials=self.trials, rstate=np.random.RandomState(50))
        elif self.classifier == 'rf':
            best = fmin(fn=self._rf_train_cv, space=space, algo=tpe.suggest,
                        max_evals=self.max_eval, trials=self.trials, rstate=np.random.RandomState(50))
        elif self.classifier == 'lr':
            best = fmin(fn=self._lr_train_cv, space=space, algo=tpe.suggest,
                        max_evals=self.max_eval, trials=self.trials, rstate=np.random.RandomState(50))
        else:
            assert ValueError, ("Please select a valid classifier to evaluate. The model available are:"
                                " KNN, XGBoost, RandomForest, LogisticRegression and SVM")
        return self.params

    def random_opt(self, space, xtrain, ytrain):
        pass

    def _rf_train_cv(self, params):
        """ Configuration parameters for RandomForest classifier
        :param params: A dictionary with parameters for RandomForest
        :return: A dictionary with the loss (1 - AUC)
        """
        self.params = params
        self.clf = RandomForestClassifier(**params)

        results = cross_val_score(self.clf, self.xtrain, self.ytrain, scoring=self.metrics).mean()
        return {'loss': 1 - results, 'status': STATUS_OK}

    def _lr_train_cv(self, params):
        """ Configuration parameters for LogisticRegression classifier
        :param params: A dictionary with parameters for LogisticRegression
        :return: A dictionary with the loss (1 - AUC)
        """
        aux_param = dict()
        aux_param['C'] = params['C']
        aux_param['max_iter'] = params['max_iter']
        aux_param['class_weight'] = params['class_weight'] if params['class_weight'] == 'balanced' else None
        aux_param['penalty'] = params['param']['penalty']
        if params['param']['penalty'] == 'l1':
            aux_param['solver'] = params['param']['solver']
        elif params['param']['penalty'] == 'l2':
            aux_param['solver'] = params['param']['solver']
        self.params = aux_param
        self.clf = LogisticRegression(**aux_param)

        results = cross_val_score(self.clf, self.xtrain, self.ytrain, scoring=self.metrics).mean()
        return {'loss': 1 - results, 'status': STATUS_OK}

    def _knn_train_cv(self, params):
        """ Configuration parameters for KNN classifier
        :param params: a dictionary with parameters for KNN
        :return: A dictionary with the loss (1 - AUC)
        """
        self.params = params
        self.clf = KNeighborsClassifier(**params)

        results = cross_val_score(self.clf, self.xtrain, self.ytrain, scoring=self.metrics).mean()
        return {'loss': 1 - results, 'status': STATUS_OK}

    def _svm_train_cv(self, params):
        """ NEED TO BE REIMPLEMENTED """
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
        self.params = aux_param
        print(aux_param)
        self.clf = SVC(verbose=False, **aux_param)
        results = cross_val_score(self.clf, self.xtrain, self.ytrain, scoring=self.metrics).mean()
        return {'loss': 1 - results, 'status': STATUS_OK}

    def _linear_svm_train_cv(self, params):
        """ Configuration parameters for KNN classifier
        :param params: a dictionary with parameters for KNN
        :return: A dictionary with the loss (1 - AUC)
        """
        self.params = params
        self.clf = LinearSVC(**params)

        results = cross_val_score(self.clf, self.xtrain, self.ytrain, scoring=self.metrics).mean()
        return {'loss': 1 - results, 'status': STATUS_OK}

    def _xgb_train_cv(self, params):
        """ Configuration parameters for Extreme Gradient Boosting classifier
        :param params: A dictionary with parameters for Extreme Gradient Boosting
        :return: A dictionary with the loss (1 - AUC)
        """
        # assert type(self.xtrain) == pd.DataFrame, "xtrain must be a pandas DataFrame"
        assert not self.xtrain.shape[0] == 0, "xtrain can't be empty"
        assert type(params) == dict, "params must be a dict type"

        self.params = params
        # 1- Select the best number of estimators using Cross Validation function
        xg_train = xgb.DMatrix(self.xtrain, label=self.ytrain.ravel())
        # do cross validation - this going to return the best number of estimators
        aux_params = params
        if params['scale_pos_weight'] == 'balanced':
            aux_params['scale_pos_weight'] = self.ytrain[self.ytrain == 0].shape[0] / self.ytrain[self.ytrain == 1].shape[0]

        results = xgb.cv(aux_params, xg_train, num_boost_round=1000, nfold=10, metrics=self.metrics,
                         early_stopping_rounds=15, stratified=True, seed=1,
                         callbacks=[xgb.callback.print_evaluation(show_stdv=False)])
        print(results)
        # the best number of estimator is exactly the shape of the returned DataFrame from the last function
        self.params['n_estimators'] = results.shape[0]
        auc = results.loc[results.shape[0]-1, 'test-auc-mean']
        return {'loss': 1 - auc, 'status': STATUS_OK}
