import time
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold


def RandomSearch(feature, label, iter_num=1000, scoring=0.5, cv=5, cv_num=3):
    start = time.time()
    for i in range(1, iter_num):
        params = {'learning_rate': np.random.choice(np.linspace(0.01, 0.1, 10).round(2)),
                  'n_estimators': np.random.choice(np.linspace(50, 500, 46)).round(0),
                  'max_depth': np.random.choice(np.linspace(3, 7, 5)).round(0),
                  'min_child_weight': np.random.choice(np.linspace(1, 7, 7)).round(0),
                  'reg_alpha': np.random.choice(np.concatenate([np.linspace(0, 1, 101), np.linspace(2, 100, 99)]).round(2)),
                  'reg_lambda': np.random.choice(np.concatenate([np.linspace(0, 1, 101), np.linspace(2, 100, 99)]).round(2)),
                  'subsample': np.random.choice(np.linspace(0.5, 1, 6)).round(2),
                  'colsample_bytree': np.random.choice(np.linspace(0.5, 1, 6)).round(2),
                  'colsample_bylevel': np.random.choice(np.linspace(0.5, 1, 6)).round(2),
                  'gamma': np.random.choice(np.linspace(0, 0.6, 13)).round(0),
                  'max_delta_step': 0,
                  'n_jobs': 14, 'scale_pos_weight': 1,'random_state': 27, 'objective': 'reg:linear'}
        model = xgb.XGBRegressor(**params)
        score = []
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=int(str(time.time())[-1]))
        for n, (train_index, test_index) in enumerate(skf.split(feature, label)):
            if n == cv_num:
                break
            model.fit(feature.loc[train_index], label[train_index])
            cv_pred = model.predict(feature.loc[test_index])
            score.append(np.sqrt(metrics.mean_squared_error(label[test_index].values, cv_pred)))
        cv_score = round(np.mean(score), 4)
        if cv_score<scoring:
            scoring = cv_score
            best_params = params
        sys.stdout.write("random search percent: {}, run time {} min, best score: {}, best param：{}\r".format(
            round(i/iter_num*100,2), divmod((time.time()-start),60)[0], scoring, best_params)
        sys.stdout.flush()
    print("XGBoost param finetuning with random search run time: %d min %.2f s" % divmod((time.time() - start), 60))
    return best_params
