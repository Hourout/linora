import sys
import time
from multiprocessing import cpu_count

import numpy as np
import xgboost as xgb
from linora.sample_splits import kfold, train_test_split


def RandomSearch(feature, label, loss, metrics, iter_num=1000, scoring=0.5, cv=5, cv_num=3,
                 metrics_min=True, speedy=True, speedy_param=(20000, 0.3), gpu=False):
    """XGBRegressor model params search use RandomSearch method.
    
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
    start = time.time()
    best_params={}
    if speedy:
        test_size = 1-round(min(speedy_param[0], feature.shape[0]*speedy_param[1])/feature.shape[0], 2)
    tree_method = 'gpu_hist' if gpu else 'auto'
    n_job = 1 if gpu else int(np.ceil(cpu_count()*0.8))
    for i in range(1, iter_num+1):
        params = {'learning_rate': np.random.choice(np.linspace(0.01, 0.1, 10).round(2)),
                  'n_estimators': np.random.choice(list(range(100, 850, 50))),
                  'max_depth': int(np.random.choice(np.linspace(3, 7, 5))),
                  'min_child_weight': int(np.random.choice(np.linspace(1, 7, 7))),
                  'reg_alpha': np.random.choice(np.concatenate([np.linspace(0, 1, 101), np.linspace(2, 100, 99)]).round(2)),
                  'reg_lambda': np.random.choice(np.concatenate([np.linspace(0, 1, 101), np.linspace(2, 100, 99)]).round(2)),
                  'subsample': np.random.choice(np.linspace(0.5, 1, 6)).round(1),
                  'colsample_bytree': np.random.choice(np.linspace(0.5, 1, 6)).round(1),
                  'colsample_bylevel': np.random.choice(np.linspace(0.5, 1, 6)).round(1),
                  'gamma': np.random.choice(np.linspace(0, 0.6, 13)).round(0),
                  'max_delta_step': 0, 'scale_pos_weight': 1, 'random_state': 27, 
                  'n_jobs':n_job, 'objective': loss, 'tree_method':tree_method}
        model = xgb.XGBRegressor(**params)
        score = []
        if speedy:
            for _ in range(cv_num):
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
        sys.stdout.write("XGBRegressor random search percent: {}%, run time {} min, best score: {}, best param：{}\r".format(
            round(i/iter_num*100,2), divmod((time.time()-start),60)[0], scoring, best_params))
        sys.stdout.flush()
    print("XGBRegressor param finetuning with random search run time: %d min %.2f s" % divmod((time.time() - start), 60))
    return best_params
