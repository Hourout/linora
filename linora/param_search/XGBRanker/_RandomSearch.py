import sys
import time
from collections import Counter
from multiprocessing import cpu_count

import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split


def RandomSearch(feature, label, group, metrics, iter_num=1000, scoring=0.5, cv=5, cv_num=3,
                 metrics_min=True, speedy=True, speedy_param=(20000, 0.3), gpu=False):
    """XGBRanker model params search use RandomSearch method.
    
    Args:
        feature: pandas dataframe, model's feature.
        label: pandas series, model's label.
        loss: XGBRanker param 'objective'.
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
        a best XGBRanker model params dict.
    Raises:
        params error.
    """
    start = time.time()
    if gpu:
        raise "XGBRanker is not supported currently."
    best_params={}
    if speedy:
        test_size = 1-round(min(speedy_param[0], feature.shape[0]*speedy_param[1])/feature.shape[0], 2)
    tree_method = 'gpu_hist' if gpu else 'auto'
    n_job = 1 if gpu else int(np.ceil(cpu_count()*0.9))
    weight_dict = Counter(label)
    if len(weight_dict)==2:
        weight = int(np.ceil(weight_dict[min(weight_dict)]/weight_dict[max(weight_dict)]))
    else:
        weight_dict = {j:i for i,j in weight_dict.items()}
        weight = int(np.ceil(weight_dict[max(weight_dict)]/weight_dict[min(weight_dict)]))
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
                  'max_delta_step': int(np.random.choice(np.linspace(0, 10, 11))),
                  'scale_pos_weight': int(np.random.choice(np.linspace(1, weight, weight))),
                  'n_jobs':n_job, 'random_state': 27, 'objective': 'rank:pairwise', 'tree_method':tree_method}
        model = xgb.XGBRanker(**params)
        score = []
        if speedy:
            for _ in range(cv_num):
                X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(feature, label, group, 
                                                                                     test_size=test_size, stratify=label, 
                                                                                     random_state=np.random.choice(range(100), 1)[0])
                model.fit(X_train, y_train, g_train)
                cv_pred = model.predict(X_test)
                score.append(metrics(y_test.values, cv_pred))
        else:
            skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=np.random.choice(range(100), 1)[0])
            for n, (train_index, test_index) in enumerate(skf.split(feature, label)):
                if n == cv_num:
                    break
                model.fit(feature.loc[train_index], label[train_index], group[train_index])
                cv_pred = model.predict(feature.loc[test_index])
                score.append(metrics(label[test_index].values, cv_pred))
        cv_score = round(np.mean(score), 4)
        if metrics_min:
            if cv_score<scoring:
                scoring = cv_score
                best_params = params.copy()
        else:
            if cv_score>scoring:
                scoring = cv_score
                best_params = params.copy()
        sys.stdout.write("XGBRanker random search percent: {}%, run time {} min, best score: {}, best param：{}\r".format(
            round(i/iter_num*100,2), divmod((time.time()-start),60)[0], scoring, best_params))
        sys.stdout.flush()
    print("XGBRanker param finetuning with random search run time: %d min %.2f s" % divmod((time.time() - start), 60))
    return best_params
