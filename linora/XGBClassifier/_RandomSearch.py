import sys
import time
from collections import Counter
from multiprocessing import cpu_count

import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split


def RandomSearch(feature, label, loss, metrics, iter_num=1000, scoring=0.5, cv=5, cv_num=3,
                 metrics_min=True, speedy=True, gpu=False, timeseries=None):
    start = time.time()
    if speedy:
        train_size = round(min(20000, feature.shape[0]*0.3)/feature.shape[0], 2)
    tree_method = 'gpu_hist' if gpu else 'auto'
    n_job = 1 if gpu else int(np.ceil(cpu_count()*0.8))
    weight_dict = Counter(label)
    if len(weight_dict)==2:
        weight = np.ceil(weight_dict[min(weight_dict)]/weight_dict[max(weight_dict)])
    else:
        weight_dict = {j:i for i,j in weight_dict.items()}
        weight = np.ceil(weight_dict[max(weight_dict)]/weight_dict[min(weight_dict)])
    for i in range(1, iter_num+1):
        params = {'learning_rate': np.random.choice(np.linspace(0.01, 0.1, 10).round(2)),
                  'n_estimators': int(np.random.choice(np.linspace(50, 500, 46))),
                  'max_depth': int(np.random.choice(np.linspace(3, 7, 5))),
                  'min_child_weight': int(np.random.choice(np.linspace(1, 7, 7))),
                  'reg_alpha': np.random.choice(np.concatenate([np.linspace(0, 1, 101), np.linspace(2, 100, 99)]).round(2)),
                  'reg_lambda': np.random.choice(np.concatenate([np.linspace(0, 1, 101), np.linspace(2, 100, 99)]).round(2)),
                  'subsample': np.random.choice(np.linspace(0.5, 1, 6)).round(2),
                  'colsample_bytree': np.random.choice(np.linspace(0.5, 1, 6)).round(2),
                  'colsample_bylevel': np.random.choice(np.linspace(0.5, 1, 6)).round(2),
                  'gamma': np.random.choice(np.linspace(0, 0.6, 13)).round(0),
                  'max_delta_step': int(np.random.choice(np.linspace(0, 10, 11))),
                  'scale_pos_weight': int(np.random.choice(np.linspace(0, weight, weight))),
                  'n_jobs':n_job, 'random_state': 27, 'objective': loss, 'tree_method':tree_method}
        model = xgb.XGBClassifier(**params)
        score = []
        if speedy:
            for i in range(cv_num):
                X_train, X_test, y_train, y_test = train_test_split(feature, label, train_size=train_size, stratify=label,
                                                                    random_state=np.random.choice(range(100), 1)[0])
                model.fit(X_train, y_train)
                cv_pred = model.predict(X_test)
                score.append(metrics(y_test.values, cv_pred))
        else:
            skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=np.random.choice(range(100), 1)[0])
            for n, (train_index, test_index) in enumerate(skf.split(feature, label)):
                if n == cv_num:
                    break
                model.fit(feature.loc[train_index], label[train_index])
                cv_pred = model.predict(feature.loc[test_index])
                score.append(metrics(label[test_index].values, cv_pred))
        cv_score = round(np.mean(score), 4)
        if metrics_min:
            if cv_score<scoring:
                scoring = cv_score
                best_params = params
        else:
            if cv_score>scoring:
                scoring = cv_score
                best_params = params
        sys.stdout.write("XGBClassifier random search percent: {}, run time {} min, best score: {}, best param：{}\r".format(
            round(i/iter_num*100,2), divmod((time.time()-start),60)[0], scoring, best_params)
        sys.stdout.flush()
    print("XGBClassifier param finetuning with random search run time: %d min %.2f s" % divmod((time.time() - start), 60))
    return best_params
