import os
import json
from multiprocessing import cpu_count

import numpy as np
import pandas as pd

from linora.utils._logger import Logger
from linora.metrics._regression import mean_squared_error
from linora.sample._fold import kfold, train_test_split
from linora.param_search._HyperParameters import HyperParametersRandom
from linora.param_search._config import __xgboost_version__


class RandomSearch():
    def __init__(self):
        hp = HyperParametersRandom()
        hp.Float('learning_rate', 0.01, 0.1)
        hp.Int('n_estimators', 100, 850)
        hp.Choice('max_depth', [3, 4, 5, 6, 7])
        hp.Choice('min_child_weight', [1, 2, 3, 4, 5, 6, 7])
        hp.Choice('max_delta_step', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        hp.Choice('reg_alpha', np.concatenate([np.linspace(0, 1, 101), np.linspace(2, 100, 99)]).round(2))
        hp.Choice('reg_lambda', np.concatenate([np.linspace(0, 1, 101), np.linspace(2, 100, 99)]).round(2))
        hp.Choice('subsample', [0.5, 0.6, 0.7, 0.8, 0.9, 1. ])
        hp.Choice('colsample_bytree', [0.5, 0.6, 0.7, 0.8, 0.9, 1. ])
        hp.Choice('colsample_bylevel', [0.5, 0.6, 0.7, 0.8, 0.9, 1. ])
        hp.Choice('colsample_bynode', [0.5, 0.6, 0.7, 0.8, 0.9, 1. ])
        hp.Choice('gamma', np.concatenate([np.linspace(0, 1, 101), np.linspace(2, 100, 99)]).round(2))
        hp.Choice('scale_pos_weight', [1])
        hp.Choice('random_state', [27])
        hp.Choice('booster', ['gbtree'])
        hp.Choice('importance_type', ["gain", "weight", "cover", "total_gain", "total_cover"])
        hp.Choice('verbosity', [0])
        self.HyperParameter = hp
        self.best_params = dict()
        self.best_params_history = dict()

    def search(self, feature, label, vaild_data=None, sample_weight=None, 
               metrics=mean_squared_error, loss='reg:squarederror', 
               iter_num=100, scoring=0.5, cv=5, cv_num=3, metrics_min=True, 
               speedy=True, speedy_param=(20000, 0.3), gpu_id=-1, 
               save_model_dir=None, save_model_name='xgb'):
        """XGBRegressor model params search use RandomSearch method.

        Args:
            feature: pandas dataframe, model's feature.
            label: pandas series, model's label.
            vaild_data: A list of (X, y, sample_weight) tuple pairs to use as validation sets, for which metrics will be computed. 
            sample_weight: pd.Series or np.array, sample weight, shape is (n,).
            metrics: model metrics function, default is `la.metircs.mean_squared_error`.
            loss: XGBRegressor param 'objective'.
            iter_num: random search count.
            scoring: metrics error opt base line value.
            cv: cross validation fold.
            cv_num: if use speedy method, minimum cross validation fold.
            metrics_min: metrics value whether the smaller the better.
            speedy: whether use speedy method.
            speedy_param: if use speedy method, test_size will be set, 
                          test_size = 1-round(min(speedy_param[0], feature.shape[0]*speedy_param[1])/feature.shape[0], 2).
            gpu_id: int, use gpu device ordinal, -1 is not use gpu.
            save_model_dir: str, save model folder.
            save_model_name: str, save model name prefix, "`xgb`_model.json" and "`xgb`_params.json".
        Returns:
            a best XGBRegressor model params dict.
        Raises:
            params error.
        """
        logger = Logger(name='xgb')
        logger.info(f"api is deprecated and will be removed in 1.5.0")
        logger.info(f"please use la.param_search.RandomSearch")
        import warnings
        warnings.filterwarnings("ignore")
        import xgboost as xgb
        assert xgb.__version__>=__xgboost_version__, f'xgboost version should be >={__xgboost_version__}.'
        if speedy:
            test_size = 1-round(min(speedy_param[0], feature.shape[0]*speedy_param[1])/feature.shape[0], 2)
        tree_method = ['gpu_hist'] if gpu_id>-1 else ['auto', 'exact', 'approx', 'hist']
        n_job = int(np.ceil(cpu_count()*0.8))

        self.HyperParameter.Choice('n_jobs', [n_job])
        self.HyperParameter.Choice('objective', [loss])
        self.HyperParameter.Choice('tree_method', tree_method)
        self.HyperParameter.Choice('gpu_id', [gpu_id])

        if vaild_data is not None:
            cv_score_list = []
            
        logger.info(f"Start XGBRegressor hyperparameter random search.")
        for i in range(1, iter_num+1):
            self.HyperParameter.update()
            model = xgb.XGBRegressor(**self.HyperParameter.params)
            score = []
            if speedy:
                for _ in range(cv_num):
                    index_list = train_test_split(feature, test_size=test_size, shuffle=True, seed=np.random.choice(range(100), 1)[0])
                    weight = None if sample_weight is None else sample_weight[index_list[0]]
                    model.fit(feature.loc[index_list[0]], label[index_list[0]], sample_weight=weight)
                    cv_pred = pd.Series(model.predict(feature.loc[index_list[1]]), index=label[index_list[1]].index)
                    if sample_weight is None:
                        score.append(metrics(label[index_list[1]], cv_pred))
                    else:
                        score.append(metrics(label[index_list[1]], cv_pred, sample_weight=sample_weight))
            else:
                index_list = kfold(feature, n_splits=cv, shuffle=True, seed=np.random.choice(range(100), 1)[0])
                for n, index in enumerate(index_list):
                    weight = None if sample_weight is None else sample_weight[index[0]]
                    model.fit(feature.loc[index[0]], label[index[0]], sample_weight=weight)
                    cv_pred = pd.Series(model.predict(feature.loc[index[1]]), index=label[index[1]].index)
                    if sample_weight is None:
                        score.append(metrics(label[index[1]], cv_pred))
                    else:
                        score.append(metrics(label[index[1]], cv_pred, sample_weight=sample_weight))
            cv_score = np.mean(score)
            if vaild_data is not None:
                cv_score_list.append(cv_score)
                if metrics_min:
                    cv_score_list.sort()
                    if cv_score_list[int(len(cv_score_list)*0.2)]>=cv_score:
                        cv_pred = pd.Series(model.predict(vaild_data[0]), index=vaild_data[1].index)
                        if len(vaild_data)==2:
                            cv_score = metrics(vaild_data[1], cv_pred)
                        else:
                            cv_score = metrics(vaild_data[1], cv_pred, sample_weight=vaild_data[2])
                    else:
                        logger.info(f"Random search progress: {i/iter_num*100:.1f}%, best score: {scoring:.4f}", enter=False if i<iter_num else True)
                        continue
                else:
                    cv_score_list.sort(reverse=1)
                    if cv_score_list[int(len(cv_score_list)*0.2)]<=cv_score:
                        cv_pred = pd.Series(model.predict(vaild_data[0]), index=vaild_data[1].index)
                        cv_score = metrics(vaild_data[1], cv_pred)
                    else:
                        logger.info(f"Random search progress: {i/iter_num*100:.1f}%, best score: {scoring:.4f}", enter=False if i<iter_num else True)
                        continue
            if metrics_min:
                if cv_score<scoring:
                    scoring = cv_score
                    self.best_params = self.HyperParameter.params.copy()
                    self.best_params_history[i] = {'score':scoring, 'best_params':self.best_params.copy()}
                    if save_model_dir is not None:
                        model.save_model(os.path.join(save_model_dir, f"{save_model_name}_model.json"))
                        with open(os.path.join(save_model_dir, f"{save_model_name}_params.json"),'w') as f:
                            json.dump(best_params, f)
            else:
                if cv_score>scoring:
                    scoring = cv_score
                    self.best_params = self.HyperParameter.params.copy()
                    self.best_params_history[i] = {'score':scoring, 'best_params':self.best_params.copy()}
                    if save_model_dir is not None:
                        model.save_model(os.path.join(save_model_dir, f"{save_model_name}_model.json"))
                        with open(os.path.join(save_model_dir, f"{save_model_name}_params.json"),'w') as f:
                            json.dump(best_params, f)
            logger.info(f"Random search progress: {i/iter_num*100:.1f}%, best score: {scoring:.4f}", enter=False if i<iter_num else True)
        logger.info(f"XGBRegressor random search best score: {scoring:.4f}", close=True, time_mode=1)
        return self.best_params
