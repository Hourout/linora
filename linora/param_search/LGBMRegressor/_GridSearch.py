import os
import json
from multiprocessing import cpu_count

import numpy as np
import pandas as pd

from linora.utils._logger import Logger
from linora.metrics._classification import auc_roc
from linora.sample._fold import kfold, train_test_split
from linora.param_search._HyperParameters import HyperParametersGrid
from linora.param_search._config import __lightgbm_version__


class GridSearch():
    def __init__(self):
        hp = HyperParametersGrid()
        hp.Choice('learning_rate', [0.01, 0.03, 0.05, 0.07, 0.09, 0.1], 0.1, rank=5)
        hp.Choice('n_estimators', list(range(100, 850, 50)), 300, rank=1)
        hp.Choice('max_depth', [3, 4, 5, 6, 7], 4, rank=2)
        hp.Dependence('num_leaves', 'max_depth', lambda x:int(np.random.choice(range(int(2**x*0.4), int(2**x*0.6)))), default=15)
        hp.Choice('reg_alpha', [0, 0.05, 0.5, 1, 50, 100], 0, rank=4)
        hp.Choice('reg_lambda', [0.05, 0.5, 1, 10, 50, 100], 1, rank=4)
        hp.Choice('subsample', [0.6, 0.7, 0.8, 0.9], 0.8, rank=3)
        hp.Choice('colsample_bytree', [0.6, 0.7, 0.8, 0.9], 0.8, rank=3)
        hp.Choice('random_state', [27])
        hp.Choice('boosting_type', ['gbdt'])
        hp.Choice('verbosity', [0])
        hp.Choice('n_jobs', [int(np.ceil(cpu_count()*0.8))])
        self.HyperParameter = hp
        self.best_params = dict()
        self.best_params_history = dict()
        
    def search(self, feature, label, vaild_data=None, sample_weight=None, 
               metrics=auc_roc, loss='regression', 
               scoring=0.5, cv=5, cv_num=3, metrics_min=True, 
               speedy=True, speedy_param=(20000, 0.3), 
               save_model_dir=None, save_model_name='lgb'):
        """LGBMRegressor model params search use GridSearch method.

        Args:
            feature: pandas dataframe, model's feature.
            label: pandas series, model's label.
            vaild_data: A list of (X, y) tuple pairs to use as validation sets, for which metrics will be computed. 
            sample_weight: pd.Series or np.array, sample weight, shape is (n,).
            metrics: model metrics function, default is `la.metircs.auc_roc`.
            loss: LGBMRegressor param 'objective'.
            scoring: metrics error opt base line value.
            cv: cross validation fold.
            cv_num: if use speedy method, minimum cross validation fold.
            metrics_min: metrics value whether the smaller the better.
            speedy: whether use speedy method.
            speedy_param: if use speedy method, test_size will be set, 
                          test_size = 1-round(min(speedy_param[0], feature.shape[0]*speedy_param[1])/feature.shape[0], 2).
            save_model_dir: str, save model folder.
            save_model_name: str, save model name prefix, "`lgb`_model.json" and "`lgb`_params.json".
        Returns:
            a best LGBMRegressor model params dict.
        Raises:
            params error.
        """
        import warnings
        warnings.filterwarnings("ignore")
        import lightgbm as lgb
        assert lgb.__version__>=__lightgbm_version__, f'lightgbm version should be >={__lightgbm_version__}.'
        logger = Logger(name='lgb')
        logger.info(f"api is deprecated and will be removed in 1.5.0")
        logger.info(f"please use la.param_search.GridSearch")
        if speedy:
            test_size = 1-round(min(speedy_param[0], feature.shape[0]*speedy_param[1])/feature.shape[0], 2)
        self.HyperParameter.Choice('objective', [loss])

        if vaild_data is not None:
            cv_score_list = []
            
        logger.info(f"Start LGBMRegressor hyperparameter grid search.")
        nums = self.HyperParameter.cardinality()
        for i in range(1, nums+1):
            self.HyperParameter.update(self.best_params)
            model = lgb.LGBMRegressor(**self.HyperParameter.params)
            score = []
            if speedy:
                for _ in range(cv_num):
                    index_list = train_test_split(feature, label, test_size=test_size, shuffle=True, seed=np.random.choice(range(100), 1)[0])
                    weight = None if sample_weight is None else sample_weight[index_list[0]]
                    model.fit(feature.loc[index_list[0]], label[index_list[0]], sample_weight=weight)
                    cv_pred = pd.Series(model.predict(feature.loc[index_list[1]]), index=label[index_list[1]].index)
                    score.append(metrics(label[index_list[1]], cv_pred))
            else:
                index_list = kfold(feature, label, n_splits=cv, shuffle=True, seed=np.random.choice(range(100), 1)[0])
                for n, index in enumerate(index_list):
                    weight = None if sample_weight is None else sample_weight[index[0]]
                    model.fit(feature.loc[index[0]], label[index[0]], sample_weight=weight)
                    cv_pred = pd.Series(model.predict(feature.loc[index[1]]), index=label[index[1]].index)
                    score.append(metrics(label[index[1]], cv_pred))
            cv_score = np.mean(score)
            if vaild_data is not None:
                cv_score_list.append(cv_score)
                if metrics_min:
                    cv_score_list.sort()
                    if cv_score_list[int(len(cv_score_list)*0.2)]>=cv_score:
                        cv_pred = pd.Series(model.predict(vaild_data[0]), index=vaild_data[1].index)
                        cv_score = metrics(vaild_data[1], cv_pred)
                    else:
                        logger.info(f"Grid search progress: {i/nums*100:.1f}%, best score: {scoring:.4f}", enter=False if i<nums else True)
                        continue
                else:
                    cv_score_list.sort(reverse=1)
                    if cv_score_list[int(len(cv_score_list)*0.2)]<=cv_score:
                        cv_pred = pd.Series(model.predict(vaild_data[0]), index=vaild_data[1].index)
                        cv_score = metrics(vaild_data[1], cv_pred)
                    else:
                        logger.info(f"Grid search progress: {i/nums*100:.1f}%, best score: {scoring:.4f}", enter=False if i<nums else True)
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
            logger.info(f"Grid search progress: {i/nums*100:.1f}%, best score: {scoring:.4f}", enter=False if i<nums else True)
        logger.info(f"LGBMRegressor grid search best score: {scoring:.4f}", close=True, time_mode=1)
        return self.best_params
