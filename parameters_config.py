from hyperopt import hp

# Parameters configuration for KNN
knn_parameters = {'n_neighbors': hp.choice('n_neighbors', range(1, 100))}

# parameters configuration for xgb
xgb_parameters = {'learning_rate': hp.loguniform('learning_rate', 0.001, 3),
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
svm_parameters = {
    'param': hp.choice('param', [{'kernel': 'linear',
                                  },
                                 {'kernel': 'rbf',
                                  'gamma':  hp.loguniform('gamma_rbf', 0.1, 3),
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

lr_params = {'penalty': hp.choice('penalty', ['l1', 'l2']),
             'C': hp.uniform('C', 0.01, 100),
             'class_weight': hp.choice('class_weight', ['balanced', 1]),
             'solver': hp.choice('solver', ['lbfgs', 'liblinear', 'newton-cg'])
             }
rf_params = {}
"""
hp.choice('classifier',[
        {'model': KNeighborsClassifier,
        'param': {'n_neighbors': hp.choice('n_neighbors',range(3,11)),
                  'algorithm':hp.choice('algorithm',['ball_tree','kd_tree']),
                  'leaf_size':hp.choice('leaf_size',range(1,50)),
                  'metric':hp.choice('metric', ["euclidean","manhattan", "chebyshev","minkowski"])}
        },
        {'model': SVC,
        'param':{'C':hp.lognormal('C',0,1),
        'kernel':hp.choice('kernel',['rbf','poly','rbf','sigmoid']),
        'degree':hp.choice('degree',range(1,15)),
        'gamma':hp.uniform('gamma',0.001,10000)}
        }
        ])
"""