# import sys
# import time
# from collections import Counter
# from multiprocessing import cpu_count

# import numpy as np
# import lightgbm as lgb
# from sklearn.model_selection import StratifiedKFold, train_test_split


# def RandomSearch(feature, label, loss, metrics, iter_num=1000, scoring=0.5, cv=5, cv_num=3,
#                  metrics_min=True, speedy=True, speedy_param=(20000, 0.3), gpu=False):
#     """LGBClassifier model params search use RandomSearch method.
    
#     Args:
#         feature: pandas dataframe, model's feature.
#         label: pandas series, model's label.
#         loss: XGBClassifier param 'objective'.
#         metrics: model metrics function.
#         scoring: metrics error opt base line value.
#         cv: cross validation fold.
#         cv_num: minimum cross validation fold.
#         metrics_min: metrics value whether the smaller the better.
#         speedy: whether use speedy method.
#         speedy_param: if use speedy method, test_size will be set, 
#                       test_size = 1-round(min(speedy_param[0], feature.shape[0]*speedy_param[1])/feature.shape[0], 2).
#         gpu: whether use gpu.
#     Returns:
#         a best LGBClassifier model params dict.
#     Raises:
#         params error.
#     """
#     start = time.time()
#     best_params={}
#     if speedy:
#         test_size = 1-round(min(speedy_param[0], feature.shape[0]*speedy_param[1])/feature.shape[0], 2)
#     tree_method = 'gpu_hist' if gpu else 'auto'
#     n_job = 1 if gpu else int(np.ceil(cpu_count()*0.8))
#     weight_dict = Counter(label)
#     if len(weight_dict)==2:
#         weight = int(np.ceil(weight_dict[min(weight_dict)]/weight_dict[max(weight_dict)]))
#     else:
#         weight_dict = {j:i for i,j in weight_dict.items()}
#         weight = int(np.ceil(weight_dict[max(weight_dict)]/weight_dict[min(weight_dict)]))
#     for i in range(1, iter_num+1):
#         max_depth = int(np.random.choice(np.linspace(3, 8, 6)))
#         params = {'learning_rate': np.random.choice(np.linspace(0.01, 0.1, 10).round(2)),
#                   'n_estimators':,
#                   'num_leaves': int(np.random.choice(range(min(20, 2**max_depth/2), 2**max_depth))),
#                   'max_depth': max_depth,
#                   'max_bin': int(np.random.choice(range(1, 255))),
#                   'subsample_for_bin':,
#                   'min_split_gain': np.random.choice(np.linspace(0, 1, 11)).round(1),
#                   'min_child_weight': np.random.choice(np.linspace(0.0005, 0.002, 31)).round(5),
#                   'min_child_samples':,
#                   'subsample': np.random.choice(np.linspace(0.5, 1, 11)).round(2),
#                   'colsample_bytree': np.random.choice(np.linspace(0.5, 1, 11)).round(2),
#                   'reg_alpha': np.random.choice(np.concatenate([np.linspace(0, 1, 101), np.linspace(2, 100, 99)]).round(2)),
#                   'reg_lambda': np.random.choice(np.concatenate([np.linspace(0, 1, 101), np.linspace(2, 100, 99)]).round(2)),
#                   'random_state':27, n_jobs=n_job, 'boosting_type':'gbdt', 'objective':loss}
#         model = lgb.LGBMClassifier(**params)
#         score = []
#         if speedy:
#             for _ in range(cv_num):
#                 X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=test_size, stratify=label,
#                                                                     random_state=np.random.choice(range(100), 1)[0])
#                 model.fit(X_train, y_train)
#                 cv_pred = model.predict_proba(X_test)
#                 score.append(metrics(y_test.values, cv_pred))
#         else:
#             skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=np.random.choice(range(100), 1)[0])
#             for n, (train_index, test_index) in enumerate(skf.split(feature, label)):
#                 if n == cv_num:
#                     break
#                 model.fit(feature.loc[train_index], label[train_index])
#                 cv_pred = model.predict_proba(feature.loc[test_index])
#                 score.append(metrics(label[test_index].values, cv_pred))
#         cv_score = round(np.mean(score), 4)
#         if metrics_min:
#             if cv_score<scoring:
#                 scoring = cv_score
#                 best_params = params.copy()
#         else:
#             if cv_score>scoring:
#                 scoring = cv_score
#                 best_params = params.copy()
#         sys.stdout.write("LGBMClassifier random search percent: {}%, run time {} min, best score: {}, best param：{}\r".format(
#             round(i/iter_num*100,2), divmod((time.time()-start),60)[0], scoring, best_params))
#         sys.stdout.flush()
#     print("LGBMClassifier param finetuning with random search run time: %d min %.2f s" % divmod((time.time() - start), 60))
#     return best_params
