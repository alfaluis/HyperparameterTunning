from hyperopt import hp
import numpy as np

# Parameters configuration for KNN
knn_params = {'n_neighbors': hp.choice('n_neighbors', range(1, 100))}

# parameters configuration for xgb
xgb_params = {'learning_rate': hp.loguniform('learning_rate', 0.001, 3),
              'gamma': hp.loguniform('gamma', 0.1, 2),
              'max_depth': hp.randint('max_depth', 20),
              'min_child_weight': hp.randint('min_child_weight', 10),
              'subsample': hp.uniform('subsample', 0.1, 1),
              'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1),
              'colsample_bylevel': hp.uniform('colsample_bylevel', 0.3, 1),
              'colsample_bynode': hp.uniform('colsample_bynode', 0.3, 1),
              'scale_pos_weight': hp.choice('scale_pos_weight', [1, 'balanced']),
              'max_delta_step': hp.randint('max_delta_step', 10)
              # 'booster': hp.choice('booster', ['gbtree', 'gblinear', 'dart']),
              }

# parameters configuration for svm
svm_params = {'param': hp.choice('param', [{'kernel': 'linear',
                                            },
                                           {'kernel': 'rbf',
                                            'gamma':  hp.loguniform('gamma_rbf', 0.1, 3)
                                            },
                                           {'kernel': 'poly',
                                            'degree': hp.quniform('degree', 2, 10, 2),
                                            'gamma':  hp.uniform('gamma_poly', 0.1, 3),
                                            'coef0': hp.uniform('coef0', 0, 1)
                                            }
                                           ]),
              'C': hp.uniform('C', 0.01, 100),
              'class_weight': hp.choice('class_weight', ['balanced', 1])
              }

lr_params = {'param': hp.choice('param', [{'penalty': 'l1',
                                           'solver': hp.choice('solver-l1', ['liblinear', 'saga']),
                                           },
                                          {'penalty': 'l2',
                                           'solver': hp.choice('solver-l2', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
                                           }
                                          ]),
             'C': hp.loguniform('C', np.log(0.01), np.log(100)),
             'class_weight': hp.choice('class_weight', ['balanced', 1]),
             'max_iter': 500 + hp.randint('max_iter', 1000)
             }

# Parameters for Random Forest
rf_params = {'n_estimators': 50 + hp.randint('n_estimators', 1950),
             'max_features': hp.choice('max_features', ['auto', 'sqrt']),
             'criterion': hp.choice('criterion', ['gini', 'entropy']),
             'class_weight': hp.choice('class_weight', ['balanced', 'balanced_subsample', None]),
             'bootstrap': hp.choice('bootstrap', [True, False]),
             'min_samples_split': 2 + hp.randint('min_samples_split', 10),
             'min_samples_leaf': 1 + hp.randint('min_samples_leaf', 10),
             'max_depth': 1 + hp.randint('max_depth', 10)}
