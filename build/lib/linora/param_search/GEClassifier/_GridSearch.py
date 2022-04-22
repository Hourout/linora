import sys
import time
import logging
import itertools

import numpy as np
import pandas as pd
from linora.sample_splits import kfold, train_test_split


def GridSearch(feature, label, estimator, params, cv_params, metrics, scoring=0.5, cv=5, cv_num=3,
               metrics_min=True, speedy=True, speedy_param=(20000, 0.3)):
    def product(x):
        if len(x)==1:
            return itertools.product(x[0])
        elif len(x)==2:
            return itertools.product(x[0], x[1])
        else:
            return itertools.product(x[0], x[1], x[2])
    logging.basicConfig(level=logging.ERROR)
    start = time.time()
    if speedy:
        test_size = 1-round(min(speedy_param[0], feature.shape[0]*speedy_param[1])/feature.shape[0], 2)
    for _, cv_param in cv_params.items():
        cv_param_name = [i for i in cv_param]
        cv_param_value = [cv_param[i] for i in cv_param_name]
        cv_param_iter = product(cv_param_value)
        for value in cv_param_iter:
            params.update({name:name_value for name, name_value in zip(cv_param_name, value)})
            model = xgb.XGBClassifier(**params)
            score = []
            if speedy:
                for i in range(cv_num):
                    index_list = train_test_split(feature, stratify=label, test_size=test_size,
                                              shuffle=True, random_state=np.random.choice(range(100), 1)[0])
                    model.fit(feature.loc[index_list[0]], label[index_list[0]])
                    cv_pred = pd.Series(model.predict_proba(feature.loc[index_list[1]]), index=label[index_list[1]].index)
                    score.append(metrics(label[index_list[1]], cv_pred))
            else:
                index_list = kfold(feature, stratify=label, n_splits=cv, shuffle=True, random_state=np.random.choice(range(100), 1)[0])
                for n, index in enumerate(index_list):
                    if n == cv_num:
                        break
                    model.fit(feature.loc[index[0]], label[index[0]])
                    cv_pred = pd.Series(model.predict_proba(feature.loc[index[1]]), index=label[index[1]].index)
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
        sys.stdout.write("GEClassifier grid search run time {} min, best score: {}, best param：{}\r".format(
            divmod((time.time()-start),60)[0], scoring, best_params))
        sys.stdout.flush()
    print("GEClassifier param finetuning with grid search run time: %d min %.2f s" % divmod((time.time() - start), 60))
    return best_params
