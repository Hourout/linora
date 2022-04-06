import os
import json
from collections import Counter
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
from linora.utils._logger import Logger
from linora.sample_splits import kfold, train_test_split
from linora.param_search._HyperParameters import HyperParametersGrid
from linora.param_search._config import __xgboost_version__


class GridSearch():
    def __init__(self):
        hp = HyperParametersGrid()
        hp.Choice('learning_rate', [0.01, 0.03, 0.05, 0.07, 0.09, 0.1], 0.1, rank=7)
        hp.Choice('n_estimators', list(range(100, 850, 50)), 300, rank=1)
        hp.Choice('max_depth', [3, 4, 5, 6, 7], 5, rank=2)
        hp.Choice('min_child_weight', [1, 2, 3, 4, 5], 1, rank=2)
        hp.Choice('max_delta_step', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1, rank=6)
        hp.Choice('reg_alpha', [0.05, 0.1, 1, 2, 3], rank=5)
        hp.Choice('reg_lambda', [0.05, 0.1, 1, 2, 3], rank=5)
        hp.Choice('subsample', [0.6, 0.7, 0.8, 0.9], 0.8, rank=4)
        hp.Choice('colsample_bytree', [0.6, 0.7, 0.8, 0.9], 0.8, rank=4)
        hp.Choice('colsample_bylevel', [0.6, 0.7, 0.8, 0.9], 0.8, rank=4)
        hp.Choice('colsample_bynode', [0.6, 0.7, 0.8, 0.9], 0.8, rank=4)
        hp.Choice('gamma', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], rank=3)
        hp.Choice('random_state', [27])
        hp.Choice('booster', ['gbtree'])
        hp.Choice('importance_type', ["gain", "weight", "cover", "total_gain", "total_cover"])
        hp.Choice('verbosity', [0])
        self.HyperParameter = hp
        
    def search(self, feature, label, loss, metrics, scoring=0.5, cv=5, cv_num=3,
               metrics_min=True, speedy=True, speedy_param=(20000, 0.3), gpu=False, save_model_dir=None):
        """XGBClassifier model params search use GridSearch method.

        Args:
            feature: pandas dataframe, model's feature.
            label: pandas series, model's label.
            loss: XGBClassifier param 'objective'.
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
            a best XGBClassifier model params dict.
        Raises:
            params error.
        """
        import warnings
        warnings.filterwarnings("ignore")
        import xgboost as xgb
        assert xgb.__version__>=__xgboost_version__, f'xgboost version should be >={__xgboost_version__}.'
        logger = Logger(name='xgb')
        best_params={}
        if speedy:
            test_size = 1-round(min(speedy_param[0], feature.shape[0]*speedy_param[1])/feature.shape[0], 2)
        tree_method = ['gpu_hist'] if gpu else ['auto', 'exact', 'approx', 'hist']
        n_job = 1 if gpu else int(np.ceil(cpu_count()*0.8))
        gpu_id = 0 if gpu else None
        weight_dict = Counter(label)
        if len(weight_dict)==2:
            weight = int(np.ceil(weight_dict[min(weight_dict)]/weight_dict[max(weight_dict)]))
        else:
            weight_dict = {j:i for i,j in weight_dict.items()}
            weight = int(np.ceil(weight_dict[max(weight_dict)]/weight_dict[min(weight_dict)]))

        self.HyperParameter.Choice('n_jobs', [n_job])
        self.HyperParameter.Choice('objective', [loss])
        self.HyperParameter.Choice('tree_method', tree_method)
        self.HyperParameter.Choice('gpu_id', [gpu_id])
        self.HyperParameter.Choice('scale_pos_weight', [1, weight], 1, rank=6)

        logger.info(f"Start XGBClassifier hyperparameter grid search.")
        rank = sorted(self.HyperParameter._rank)
        for i in rank:
            for params in self.HyperParameter.update(i):
                model = xgb.XGBClassifier(**params)
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
                cv_score = np.mean(score)
                if metrics_min:
                    if cv_score<scoring:
                        scoring = cv_score
                        best_params = params.copy()
                        if save_model_dir is not None:
                            model.save_model(os.path.join(save_model_dir, "xgb_model.json"))
                            with open(os.path.join(save_model_dir, "xgb_params.json"),'w') as f:
                                json.dump(best_params, f)
                else:
                    if cv_score>scoring:
                        scoring = cv_score
                        best_params = params.copy()
                        if save_model_dir is not None:
                            model.save_model(os.path.join(save_model_dir, "xgb_model.json"))
                            with open(os.path.join(save_model_dir, "xgb_params.json"),'w') as f:
                                json.dump(best_params, f)
                logger.info(f"grid search progress: {round((i+1)/len(rank)*100,1)}%, best score: {scoring:.4}", enter=False if (i+1)<len(rank) else True)
        logger.info(f"XGBClassifier grid search best score: {scoring:.4}", close=True, time_mode=1)
        return best_params
