import sys
import time
from collections import Counter

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


def RandomSearch(feature, label, estimator, params, metrics, iter_num=1000, scoring=0.5, cv=5, cv_num=3,
                 metrics_min=True, speedy=True, speedy_param=(20000, 0.3)):
    start = time.time()
    best_params={}
    if speedy:
        train_size = round(min(speedy_param[0], feature.shape[0]*speedy_param[1])/feature.shape[0], 2)
    for i in range(1, iter_num+1):
        param = {i:np.random.choice(j) for i,j in params.items()}
        model = estimator(**param)
        score = []
        if speedy:
            for _ in range(cv_num):
                X_train, X_test, y_train, y_test = train_test_split(feature, label, train_size=train_size, stratify=label,
                                                                    random_state=np.random.choice(range(100), 1)[0])
                model.fit(X_train, y_train)
                cv_pred = model.predict_proba(X_test)
                score.append(metrics(y_test.values, cv_pred))
        else:
            skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=np.random.choice(range(100), 1)[0])
            for n, (train_index, test_index) in enumerate(skf.split(feature, label)):
                if n == cv_num:
                    break
                model.fit(feature.loc[train_index], label[train_index])
                cv_pred = model.predict_proba(feature.loc[test_index])
                score.append(metrics(label[test_index].values, cv_pred))
        cv_score = round(np.mean(score), 4)
        if metrics_min:
            if cv_score<scoring:
                scoring = cv_score
                best_params = param.copy()
        else:
            if cv_score>scoring:
                scoring = cv_score
                best_params = param.copy()
        sys.stdout.write("XGBClassifier random search percent: {}%, run time {} min, best score: {}, best param：{}\r".format(
            round(i/iter_num*100,2), divmod((time.time()-start),60)[0], scoring, best_params))
        sys.stdout.flush()
    print("GEClassifier param finetuning with random search run time: %d min %.2f s" % divmod((time.time() - start), 60))
    return best_params
