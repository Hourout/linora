import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
from linora.sample_splits import kfold, train_test_split


def RandomSearch(feature, label, estimator, params, metrics, iter_num=1000, scoring=0.5, cv=5, cv_num=3,
                 metrics_min=True, speedy=True, speedy_param=(20000, 0.3)):
    """General model params search use RandomSearch method.
    
    Args:
        feature: pandas dataframe, model's feature.
        label: pandas series, model's label.
        estimator: general model.
        params: a general model params dict.
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
        a best General model params dict.
    Raises:
        params error.
    """
    start = time.time()
    best_params={}
    if speedy:
        test_size = round(min(speedy_param[0], feature.shape[0]*speedy_param[1])/feature.shape[0], 2)
    for i in range(1, iter_num+1):
        param = {i:np.random.choice(j) for i,j in params.items()}
        model = estimator(**param)
        score = []
        if speedy:
            for _ in range(cv_num):
                index_list = train_test_split(feature, test_size=test_size,
                                              shuffle=True, random_state=np.random.choice(range(100), 1)[0])
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
                best_params = param.copy()
        else:
            if cv_score>scoring:
                scoring = cv_score
                best_params = param.copy()
        sys.stdout.write("GERegressor random search percent: {}%, run time {} min, best score: {}, best param：{}\r".format(
            round(i/iter_num*100,2), divmod((time.time()-start),60)[0], scoring, best_params))
        sys.stdout.flush()
    print("GERegressor param finetuning with random search run time: %d min %.2f s" % divmod((time.time() - start), 60))
    return best_params
