import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb


def train_xgb_model(params, xtrain, ytrain):
    model = xgb.XGBClassifier(**params)
    eval_set = [(xtrain, ytrain)]
    model.fit(xtrain, y_train,
              eval_metric=['logloss', 'auc'],
              eval_set=eval_set,
              verbose=True)
    return model


def get_num_estimators_xgb(xtrain: pd.DataFrame, ytrain: pd.Series, params: dict, method='cv', xtest=None, ytest=None,
                           metrics='auc', verbose=False, show=True):
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
    assert type(xtrain) == pd.DataFrame, "xtrain must be a pandas DataFrame"
    assert not xtrain.empty, "xtrain can't be empty"
    assert type(params) == dict, "params must be a dict type"

    if method == 'cv':
        print('Training with Cross Validation...')
        # 1- Select the best number of estimators using Cross Validation function
        xg_train = xgb.DMatrix(xtrain, label=ytrain.ravel())
        # do cross validation - this going to return the best number of estimators
        print('Start cross validation')
        cvresult = xgb.cv(params, xg_train, num_boost_round=1000, nfold=10, metrics=['auc'],
                          early_stopping_rounds=50, stratified=True, seed=1,
                          callbacks=[xgb.callback.print_evaluation(show_stdv=False)])

        # the best number of estimator is exactly the shape of the returned DataFrame from the last function
        print(cvresult)
        params['n_estimators'] = cvresult.shape[0]

        # train the model
        model = xgb.XGBClassifier(**params)
        eval_set = [(xtrain, ytrain)]
        model.fit(xtrain, ytrain,
                  eval_metric=metrics,
                  eval_set=eval_set,
                  verbose=verbose)

        if xtest is not None and ytest is not None:
            pred = model.predict_proba(xtest)
            if show:
                plot_performance(y_true=ytest, y_score=pred[:, 1])
        return model

    elif (method == 'standard') and (xtest is not None) and (ytest is not None):
        print('Training with Cross Validation...')
        # 2- Select the best number if estimators using the provided Test Set
        # param mustn't include n_estimators parameter
        model = xgb.XGBClassifier(**params)
        eval_set = [(xtrain, ytrain), (xtest, ytest)]
        # when we use multiple eval_metric the early_stopping_rounds use the last one
        model.fit(xtrain, ytrain,
                  eval_metric=metrics,
                  eval_set=eval_set,
                  early_stopping_rounds=10, verbose=verbose)

        if xtest is not None and ytest is not None:
            pred = model.predict_proba(xtest)
            if show:
                plot_performance(y_true=ytest, y_score=pred[:, 1])

        return model

    else:
        assert ValueError, "parameters method, xtrain, ytrain, xtest and ytest must be a valid combination "
        return None


def plot_performance(y_true, y_score):
    fpr, tpr, thres = roc_curve(y_true=y_true, y_score=y_score)
    auc_score = auc(fpr, tpr)
    print('ROCAUC:', auc_score)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(fpr, tpr)
    ax[1].plot(fpr, thres)
    ax[0].grid()
    ax[1].grid()
    fig.suptitle('AUC: {}'.format(auc_score))
    plt.show()


def identity_variables_types(data, thres=10):
    """ Automatically estimate the type of variable from each columns in the DataFrame. The discriminate between
    categorical or numeric variable the function use a threshold that is set by user
    :param data: A DataFrame to be processed
    :param thres: Threshold to be used to discriminate between categorical or numeric
    :return: A new DataFrame where each columns renamed, including at the end of the name '_cat' or '_num' for each case
    """
    data_copy = data.copy()
    res = data.apply(lambda x: x.unique().shape[0], axis=0)
    cat_cols_names = res.index[res <= thres]
    new_cat_names = {name: name + '_cat' for name in cat_cols_names}
    num_cols_names = res.index[res > thres]
    new_num_names = {name: name + '_num' for name in num_cols_names}
    data_copy.rename(columns=new_cat_names, inplace=True)
    data_copy.rename(columns=new_num_names, inplace=True)
    return data_copy


def columns_with_na_values(data):
    """
    Return columns that have na values
    :param data: DataFrame to be analyzed
    :return: Array with columns name
    """
    aux = data.isna().sum() > 0
    return aux.index[aux.values].values


def replace_comma(data):
    """ Function that try con convert string to numeric, it also covert the case when the decimal separator is a
    comma and can't be properly interpreted by the cast function
    :param data: DataFrame with columns to be converted
    :return: DataFrame with columns transformed
    """
    try:
        aux = np.float32(data.replace(',', '.'))
    except AttributeError as err:
        print('No string. Convert to numeric')
        aux = np.float32(data)
    return aux


if __name__ == '__main__':
    root = os.getcwd()
    # database = os.path.join(root, '../../Data')
    database = os.path.join(root, 'database')
    df = pd.read_csv(os.path.join(database, 'FINAL_DATA_PRO_APV_7.csv'), sep=';', encoding='latin-1')
    df.set_index('CUENTA', drop=True, inplace=True)
    main_mask = (df.B_MARCA_PROD == 1) & (df.FALLECIDOS == 0) & (df.B_PENSION == 0) & (df.ALIANZA == 0)
    df_clean = df.loc[main_mask, :].copy()

    df_clean.loc[:, 'DateFirstContribution'] = pd.to_datetime(df_clean.FECHA_PRIMER_APORTE,
                                                              yearfirst=True,
                                                              format='%d-%m-%Y',
                                                              errors='coerce')

    # get customer behaviour before september. If customer didn't have movement before this date,
    # we can't make predictions
    df_with_customer_sept = df_clean.loc[df_clean.DateFirstContribution < '2018-09-01', :].copy()

    df_with_customer_sept.loc[:, 'DateAccountChurn'] = pd.to_datetime(df_with_customer_sept.FIX_FECHA_FUGA,
                                                                      yearfirst=True,
                                                                      format='%d-%m-%Y',
                                                                      errors='coerce')
    # obtain accounts (customers) that left the company after 2018-08-31
    df_accounts_left_company = df_with_customer_sept.loc[df_with_customer_sept.DateAccountChurn >= '2018-09-01', :]
    # obtain account that did't left the company
    df_accounts_in_company = df_with_customer_sept.loc[df_with_customer_sept.DateAccountChurn.isna(), :]

    # join DataFrame (churn / no churn)
    df_cleaned = pd.concat([df_accounts_left_company, df_accounts_in_company], ignore_index=False)

    # **********************************************
    # FIRST APPROACH - REPLICATE THE CURRENT RESULTS
    # **********************************************
    columns_used = ['ANTIGU_PROD_FIX', 'ANOS_APORTANDO', 'CANT_TRASP_12M', 'TOTAL_TRASPASO_12M', 'CANT_RETIRO_12M',
                    'CANT_TRASPIN_12M', 'TOTAL_TRASPIN_12M', 'PROM_TRASPIN_12M', 'CANT_DEPOSA_12M', 'TOTAL_DEPOSA_12M',
                    'CANT_DEPOSB_12M', 'TOTAL_DEPOSB_9M', 'PROM_DEPOSB_9M', 'CANT_DEPOSC_12M', 'TOTAL_DEPOSC_12M',
                    'PROM_DEPOSC_12M', 'EDAD', 'FFMM', 'SCA', 'FUGADO90_ULT_6M']

    # use columns considered initially
    df_with_features_used = df_cleaned.loc[:, columns_used].copy()
    df_with_features_used.rename(columns={'FUGADO90_ULT_6M': 'Target'}, inplace=True)
    # get columns with na values
    columns_na = columns_with_na_values(data=df_with_features_used)
    # check percentage
    print(df_with_features_used.loc[:, columns_na].isna().sum(axis=0) / df_with_features_used.shape[0])
    # fill na with median
    median_age = df_with_features_used.loc[df_with_features_used.EDAD >= 18, :].median()
    df_with_features_used.EDAD.fillna(value=median_age)
    # eliminate customer with bad age
    df_with_features_used = df_with_features_used.loc[df_with_features_used.EDAD >= 18, :]

    # convert object columns to numeric
    df_columns_to_numeric = df_with_features_used.select_dtypes('object').applymap(replace_comma)
    df_without_obj_columns = df_with_features_used.drop(columns=df_with_features_used.select_dtypes('object').columns)
    df_with_features_used = pd.merge(left=df_without_obj_columns, right=df_columns_to_numeric, how='inner', on='CUENTA')

    # apply log transformation
    transformer = FunctionTransformer(np.log1p, validate=True)
    arr_logs = transformer.transform(df_with_features_used)
    log_cols = pd.DataFrame(data=arr_logs,
                            columns=[col + '_log' for col in df_with_features_used.columns],
                            index=df_with_features_used.index)
    # Join normal features and log features
    new_df = pd.merge(left=df_with_features_used.loc[:, ['Target']], right=log_cols, how='inner', on='CUENTA')

    # Split train and test set
    train_set, test_set = train_test_split(df_with_features_used,
                                           test_size=0.3,
                                           stratify=df_with_features_used.Target,
                                           random_state=1)
    X_train = train_set.loc[:, [c for c in df_with_features_used.columns if 'Target' not in c]]
    X_test = test_set.loc[:, [c for c in df_with_features_used.columns if 'Target' not in c]]
    y_train = train_set.loc[:, 'Target']
    y_test = test_set.loc[:, 'Target']
    std_scaler = StandardScaler().fit(X_train)
    max_min_scaler = MinMaxScaler().fit(X_train)
    X_train_final = std_scaler.transform(X_train)
    X_test_final = std_scaler.transform(X_test)

    # *****************************************
    # Training a XGBoost
    # *****************************************
    positive_weight = train_set.loc[train_set.Target == 0, :].shape[0]/train_set.loc[train_set.Target == 1, :].shape[0]
    param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
    get_num_estimators_xgb(xtrain=X_train,
                           ytrain=y_train,
                           xtest=X_test,
                           ytest=y_test,
                           params=param,
                           method='cv')
    # 1- Select the best number of estimators using Cross Validation function
    positive_weight = train_set.loc[train_set.Target == 0, :].shape[0]/train_set.loc[train_set.Target == 1, :].shape[0]
    param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
    xgtrain = xgb.DMatrix(X_train_final, label=y_train.ravel())
    # do cross validation - this going to return the best number of estimators
    print('Start cross validation')
    cvresult = xgb.cv(param, xgtrain, num_boost_round=1000, nfold=10, metrics=['auc'],
                      early_stopping_rounds=50, stratified=True, seed=1,
                      callbacks=[xgb.callback.print_evaluation(show_stdv=False)])
    print(cvresult.iloc[-1, :])
    # the best number of estimator is exactly the shape of the returned DataFrame from the last function
    param['n_estimators'] = cvresult.shape[0]
    # train the model
    model = xgb.XGBClassifier(**param)
    eval_set = [(X_train_final, y_train), (X_test_final, y_test)]
    model.fit(X_train_final, y_train,
              eval_metric=['logloss', 'auc'],
              eval_set=eval_set,
              verbose=True)
    pred = model.predict_proba(X_test_final)
    plot_performance(y_true=y_test, y_score=pred[:, 1])
    # 2- Select the best number if estimators using the provided Test Set
    # param mustn't include n_estimators parameter
    param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
    model2 = xgb.XGBClassifier(**param)
    eval_set = [(X_train_final, y_train), (X_test_final, y_test)]
    # when we use multiple eval_metric the early_stopping_rounds use the last one
    res = model2.fit(X_train_final, y_train,
                     eval_metric=['auc'],
                     eval_set=eval_set,
                     early_stopping_rounds=10, verbose=True)
    pred = model2.predict_proba(X_test_final)
    plot_performance(y_true=y_test, y_score=pred[:, 1])

    get_num_estimators_xgb(xtrain=X_train,
                           ytrain=y_train,
                           params=param,
                           method='standard',
                           xtest=X_test)
    # OLD SOLUTION

    parameters = {'learning_rate': [float(x) for x in np.linspace(start=0.001, stop=2, num=10)],
                  'gamma': [float(x) for x in np.linspace(start=0.1, stop=2, num=5)],
                  'n_estimators': [int(x) for x in np.linspace(start=100, stop=2000, num=10)],
                  'max_depth': [int(x) for x in np.linspace(10, 110, num=11)],
                  'min_child_weight': [int(x) for x in np.linspace(start=1, stop=7, num=3)],
                  'subsample': [0.5, 0.8, 1],
                  'colsample_bytree': [0.5, 0.8, 1],
                  'scale_pos_weight': [positive_weight, 1],
                  'max_delta_step': [int(x) for x in np.linspace(1, 10, num=5)]}

    model = XGBClassifier(**ind_params)
    xgb_random = RandomizedSearchCV(estimator=model, param_distributions=parameters,
                                    scoring='roc_auc', cv=5, n_iter=10,
                                    verbose=50, n_jobs=1)
    xgb_random.fit(X=X_train_final, y=y_train)

    with open(os.path.join(root, 'random_search_models_xgb6.pickle'), 'wb') as f:
        pickle.dump(xgb_random, f)
    results = pd.DataFrame(xgb_random.cv_results_)
    results.to_csv(os.path.join(root, 'ModelResults_xgb6.csv'))

    # check results
    with open(os.path.join(root, 'random_search_models_xgb6.pickle'), 'rb') as f:
        check_model = pickle.load(f)

    best_model = xgb_random.best_estimator_
    # best_model = XGBClassifier(**check_model.best_params_).fit(X_train_final, y_train)
    pred = best_model.predict_proba(X_test_final)
    fpr, tpr, thres = roc_curve(y_true=y_test, y_score=pred[:, 1])
    print(roc_auc_score(y_test, pred[:, 1]))
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(fpr, tpr)
    ax[1].plot(fpr, thres)
    ax[0].grid()
    ax[1].grid()
    fig.suptitle('aaaa')
    plt.show()

