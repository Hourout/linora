import sys
import time
import itertools
from multiprocessing import cpu_count

import numpy as np
import xgboost as xgb
from linora.sample_splits import kfold, train_test_split


def GridSearch(feature, label, loss, metrics, scoring=0.5, cv=5, cv_num=3,
               metrics_min=True, speedy=True, speedy_param=(20000, 0.3), gpu=False):
    """XGBRegressor model params search use GridSearch method.
    
    Args:
        feature: pandas dataframe, model's feature.
        label: pandas series, model's label.
        loss: XGBRegressor param 'objective'.
        metrics: model metrics function.
        scoring: metrics error opt base line value.
        cv: cross validation fold.
        cv_num: minimum cross validation fold.
        metrics_min: metrics value whether the smaller the better.
        speedy: whether use speedy method.
        speedy_param: if use speedy method, test_size will be set, 
                      test_size = 1-round(min(speedy_param[0], feature.shape[0]*speedy_param[1])/feature.shape[0], 2).
        gpu: whether use gpu.
    Returns:
        a best XGBRegressor model params dict.
    Raises:
        params error.
    """
    def product(x):
        if len(x)==1:
            return itertools.product(x[0])
        elif len(x)==2:
            return itertools.product(x[0], x[1])
        else:
            return itertools.product(x[0], x[1], x[2])
    start = time.time()
    if speedy:
        test_size = 1-round(min(speedy_param[0], feature.shape[0]*speedy_param[1])/feature.shape[0], 2)
    tree_method = 'gpu_hist' if gpu else 'auto'
    n_job = 1 if gpu else int(np.ceil(cpu_count()*0.8))
    params = {'learning_rate': 0.1, 'n_estimators': 300, 'max_depth': 5, 'min_child_weight': 1,
              'reg_alpha': 0, 'reg_lambda': 1, 'gamma': 0,
              'subsample': 0.8, 'colsample_bytree': 0.8, 'colsample_bylevel': 0.8,
              'max_delta_step': 0, 'scale_pos_weight': 1, 'random_state': 27,
              'n_jobs':n_job, 'objective': loss, 'tree_method':tree_method}
    cv_params = {'param1':{'n_estimators': list(range(100, 850, 50))},
                 'param2':{'max_depth': [3, 4, 5, 6, 7],
                           'min_child_weight': [1, 2, 3, 4, 5]},
                 'param3':{'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]},
                 'param4':{'subsample': [0.6, 0.7, 0.8, 0.9],
                           'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
                           'colsample_bylevel': [0.6, 0.7, 0.8, 0.9]},
                 'param5':{'reg_alpha': [0.05, 0.1, 1, 2, 3],
                           'reg_lambda': [0.05, 0.1, 1, 2, 3]},
                 'param6':{'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.09, 0.1]}}
    for _, cv_param in cv_params.items():
        cv_param_name = [i for i in cv_param]
        cv_param_value = [cv_param[i] for i in cv_param_name]
        cv_param_iter = product(cv_param_value)
        for value in cv_param_iter:
            params.update({name:name_value for name, name_value in zip(cv_param_name, value)})
            model = xgb.XGBRegressor(**params)
            score = []
            if speedy:
                for i in range(cv_num):
                    index_list = train_test_split(feature, test_size=test_size, shuffle=True, random_state=np.random.choice(range(100), 1)[0])
                    model.fit(feature.loc[index_list[0]], label[index_list[0]])
                    cv_pred = pd.Series(model.predict(feature.loc[index_list[1]]), index=label[index_list[1]].index)
                    score.append(metrics(label[index_list[1]], cv_pred))
            else:
                index_list = kfold(feature, n_splits=cv, shuffle=True, random_state=np.random.choice(range(100), 1)[0])
                for n, index in enumerate(index_list):
                    if n == cv_num:
                        break
                    model.fit(feature.loc[index[0]], label[index[0]])
                    cv_pred = pd.Series(model.predict(feature.loc[index[1]]), index=label[index[1]].index)
                    score.append(metrics(label[index[1]], cv_pred))
            cv_score = round(np.mean(score), 4)
            if metrics_min:
                if cv_score<scoring:
                    scoring = cv_score
                    best_params = params.copy()
            else:
                if cv_score>scoring:
                    scoring = cv_score
                    best_params = params.copy()
        params = best_params.copy()
        sys.stdout.write("XGBRegressor grid search run time {} min, best score: {}, best param：{}\r".format(
            divmod((time.time()-start),60)[0], scoring, best_params))
        sys.stdout.flush()
    print("XGBRegressor param finetuning with grid search run time: %d min %.2f s" % divmod((time.time() - start), 60))
    return best_params
