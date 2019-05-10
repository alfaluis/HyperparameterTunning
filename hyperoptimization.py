from hyperopt import tpe, Trials, STATUS_OK, fmin
import pandas
from testing_xgb_early_stopping import get_num_estimators_xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import xgboost as xgb


class HyperOptimization(object):
    def __init__(self, metrics='auc', verbose=False, plot_auc=False, tpe_algorithm=tpe.suggest):
        self.metrics = metrics
        self.xtrain: pandas.DataFrame = pandas.DataFrame
        self.ytrain: pandas.Series = pandas.Series
        self.params: dict = {}
        self.method = 'cv'
        self.xtest = None
        self.ytest = None
        self.verbose = verbose
        self.plot_ = plot_auc
        self.tpe_algorithm = tpe_algorithm
        self.clf = None
        self.cvresult = pd.DataFrame

    def hyperopt_train_test(self, params):
        self.clf = KNeighborsClassifier(**params)
        self.clf.fit(self.xtrain, self.ytrain)

        results = cross_val_score(self.clf, self.xtrain, self.ytrain, scoring='accuracy').mean()
        return {'loss': 1 - results, 'status': STATUS_OK}

    def find(self, space, xtrain, ytrain):
        self.params = space
        self.xtrain = xtrain
        self.ytrain = ytrain
        trials = Trials()
        best = fmin(fn=self.hyperopt_train_test, space=space, algo=tpe.suggest,
                    max_evals=100, trials=trials, rstate=np.random.RandomState(50))
        return best

    def find_xgb(self, space, xtrain, ytrain):
        self.params = space
        self.xtrain = xtrain
        self.ytrain = ytrain
        trials = Trials()
        best = fmin(fn=self.get_num_estimators_xgb, space=space, algo=tpe.suggest,
                    max_evals=100, trials=trials, rstate=np.random.RandomState(50))
        return best

    def get_num_estimators_xgb(self, params):
        """
        :param xtrain:
        :param ytrain:
        :param params:
        :param method:
        :param xtest:
        :param ytest:
        :param metrics:
        :param verbose:
        :param show:
        :return:
        """
        assert type(self.xtrain) == pd.DataFrame, "xtrain must be a pandas DataFrame"
        assert not self.xtrain.empty, "xtrain can't be empty"
        assert type(params) == dict, "params must be a dict type"

        if self.method == 'cv':
            print('Training with Cross Validation...')
            # 1- Select the best number of estimators using Cross Validation function
            xg_train = xgb.DMatrix(self.xtrain, label=self.ytrain.ravel())
            # do cross validation - this going to return the best number of estimators
            print('Start cross validation')
            self.results = xgb.cv(params, xg_train, num_boost_round=1000, nfold=10, metrics=['auc'],
                                  early_stopping_rounds=5, stratified=True, seed=1,
                                  callbacks=[xgb.callback.print_evaluation(show_stdv=False)])

            # the best number of estimator is exactly the shape of the returned DataFrame from the last function
            self.params['n_estimators'] = self.cvresult.shape[0]
            auc = self.results.loc[self.results.shape[0]-1, 'test-auc-mean']
            return {'loss': 1 - auc, 'status': STATUS_OK}

        elif (self.method == 'standard') and (self.xtest is not None) and (self.ytest is not None):
            print('Training with Cross Validation...')
            # 2- Select the best number if estimators using the provided Test Set
            # param mustn't include n_estimators parameter
            model = xgb.XGBClassifier(**params)
            eval_set = [(self.xtrain, self.ytrain), (self.xtest, self.ytest)]
            # when we use multiple eval_metric the early_stopping_rounds use the last one
            model.fit(self.xtrain, self.ytrain,
                      eval_metric=self.metrics,
                      eval_set=eval_set,
                      early_stopping_rounds=10, verbose=self.verbose)
            pred = model.predict_proba(self.xtest)
            auc = roc_auc_score(y_true=self.ytest, y_score=pred[:, 1])
            return {'loss': 1 - auc, 'status': STATUS_OK}
        else:
            assert ValueError, "parameters method, xtrain, ytrain, xtest and ytest must be a valid combination "
            return None
