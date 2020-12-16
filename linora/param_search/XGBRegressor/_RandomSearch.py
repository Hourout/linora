import os
import sys
import json
import time
import pickle
from collections import Counter
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
from linora.sample_splits import kfold, train_test_split
from linora.param_search._HyperParameters import HyperParametersRandom


def RandomSearch(feature, label, loss, metrics, iter_num=1000, scoring=0.5, cv=5, cv_num=3,
                 metrics_min=True, speedy=True, speedy_param=(20000, 0.3), gpu=False, 
                 save_model_dir=None
                ):
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
        save_model_dir: save model folder.
    Returns:
        a best XGBRegressor model params dict.
    Raises:
        params error.
    """
    import xgboost as xgb
    assert xgb.__version__>='1.2.0', 'xgboost version should be >1.2.0'
    start = time.time()
    best_params={}
    if speedy:
        test_size = 1-round(min(speedy_param[0], feature.shape[0]*speedy_param[1])/feature.shape[0], 2)
    tree_method = ['gpu_hist'] if gpu else ['auto', 'exact', 'approx', 'hist']
    n_job = 1 if gpu else int(np.ceil(cpu_count()*0.8))
    
    hp = HyperParametersRandom()
    hp.Float('learning_rate', 0.01, 0.1)
    hp.Int('n_estimators', 100, 850)
    hp.Choice('max_depth', [3, 4, 5, 6, 7])
    hp.Choice('min_child_weight', [1, 2, 3, 4, 5, 6, 7])
    hp.Choice('max_delta_step', [0])
    hp.Choice('reg_alpha', np.concatenate([np.linspace(0, 1, 101), np.linspace(2, 100, 99)]).round(2))
    hp.Choice('reg_lambda', np.concatenate([np.linspace(0, 1, 101), np.linspace(2, 100, 99)]).round(2))
    hp.Choice('subsample', [0.5, 0.6, 0.7, 0.8, 0.9, 1. ])
    hp.Choice('colsample_bytree', [0.5, 0.6, 0.7, 0.8, 0.9, 1. ])
    hp.Choice('colsample_bylevel', [0.5, 0.6, 0.7, 0.8, 0.9, 1. ])
    hp.Choice('colsample_bynode', [0.5, 0.6, 0.7, 0.8, 0.9, 1. ])
    hp.Choice('gamma', np.concatenate([np.linspace(0, 1, 101), np.linspace(2, 100, 99)]).round(2))
    hp.Choice('scale_pos_weight', [1])
    hp.Choice('n_jobs', [n_job])
    hp.Choice('random_state', [27])
    hp.Choice('objective', [loss])
    hp.Choice('booster', ['gbtree'])
    hp.Choice('tree_method', tree_method)
    hp.Choice('importance_type', ["gain", "weight", "cover", "total_gain", "total_cover"])
    
    for i in range(1, iter_num+1):
        hp.update()
        model = xgb.XGBRegressor(**hp.params)
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
                if save_model_dir is not None:
                    pickle.dump(model, open(os.path.join(save_model_dir, "xgb_model.pkl"), "wb"))
                    with open(os.path.join(save_model_dir, "xgb_params.json"),'w') as f:
                        json.dump(best_params, f)
        else:
            if cv_score>scoring:
                scoring = cv_score
                best_params = params.copy()
                if save_model_dir is not None:
                    pickle.dump(model, open(os.path.join(save_model_dir, "xgb_model.pkl"), "wb"))
                    with open(os.path.join(save_model_dir, "xgb_params.json"),'w') as f:
                        json.dump(best_params, f)
        sys.stdout.write("XGBRegressor random search percent: {}%, run time {} min, best score: {}, best param：{}\r".format(
            round(i/iter_num*100,2), divmod((time.time()-start),60)[0], scoring, best_params))
        sys.stdout.flush()
    print("XGBRegressor param finetuning with random search run time: %d min %.2f s" % divmod((time.time() - start), 60))
    return best_params
