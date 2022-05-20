import os
import json
from collections import Counter

import numpy as np
import pandas as pd

from linora.utils._config import Config
from linora.utils._logger import Logger
from linora.sample._fold import kfold, train_test_split
from linora.param_search._config import model_hp
from linora.param_search._HyperParameters import HyperParametersRandom

__all__ = ['RandomSearch']


class RandomSearch():
    def __init__(self, model=None, hp=None, name=None):
        self.params = Config()
        self.params.name = 'RS' if name is None else name
        self.params.model_init = model
        
        if model in ['XGBClassifier', 'XGBRegressor', 'LGBMClassifier', 'LGBMRegressor']:
            self.hp = model_hp(model=model, method='random')
            if hp is not None:
                self.hp.from_HyperParameters(hp)
        else:
            self.hp = hp
        self.best_params = dict()
        self.best_params_history = dict()

    def search(self, train_data, metrics, vaild_data=None,
               iter_num=100, scoring=0.5, cv=5, cv_num=3, metrics_min=True, 
               speedy=True, speedy_param=(20000, 0.3), 
               save_model_dir=None, save_model_name=None):
        """model params search use RandomSearch method.

        Args:
            train_data: A list of (X, y, sample_weight) tuple pairs to use as train sets.
            metrics: model metrics function.
            vaild_data: A list of (X, y, sample_weight) tuple pairs to use as validation sets.
            iter_num: random search count.
            scoring: metrics error opt base line value.
            cv: cross validation fold.
            cv_num: if use speedy method, minimum cross validation fold.
            metrics_min: metrics value whether the smaller the better.
            speedy: whether use speedy method.
            speedy_param: if use speedy method, test_size will be set, 
                          test_size = 1-round(min(speedy_param[0], feature.shape[0]*speedy_param[1])/feature.shape[0], 2).
            save_model_dir: str, save model folder, only work with model='XGBClassifier' or 'XGBRegressor'.
            save_model_name: str, save model name prefix, only work with model='XGBClassifier' or 'XGBRegressor'.
        Returns:
            a best model params dict.
        Raises:
            params error.
        """
        logger = Logger(name=self.params.name)
        import warnings
        warnings.filterwarnings("ignore")
        if speedy:
            test_size = 1-round(min(speedy_param[0], len(train_data[1])*speedy_param[1])/len(train_data[1]), 2)
        if self.params.model_init=='XGBClassifier':
            self._xgb_weight(train_data[1])
        
        if vaild_data is not None:
            cv_score_list = []
            
        logger.info(f"Start hyperparameter random search.")
        for i in range(1, iter_num+1):
            self.hp.update()
            self.params.model = self.params.model_init(**self.hp.params)
            score = []
            if speedy:
                for _ in range(cv_num):
                    index = train_test_split(train_data[0], train_data[1], test_size, seed=np.random.choice(range(100), 1)[0])
                    score.append(self._model_fit_predict(train_data, metrics, index, mode=1))
            else:
                index_list = kfold(train_data[0], train_data[1], n_splits=cv, seed=np.random.choice(range(100), 1)[0])
                for n, index in enumerate(index_list):
                    score.append(self._model_fit_predict(train_data, metrics, index, mode=1))
            cv_score = np.mean(score)
            if vaild_data is not None:
                cv_score_list.append(cv_score)
                cv_score_list.sort()
                threshold = cv_score_list[int(len(cv_score_list)*(0.2 if metrics_min else 0.8))]
                if (metrics_min==True and threshold>=cv_score) or (metrics_min==False and threshold<=cv_score):
                    cv_score = self._model_fit_predict(vaild_data, metrics, index=None, mode=0)
                else:
                    logger.info(f"Random search progress: {i/iter_num*100:.1f}%, best score: {scoring:.4f}", enter=False if i<iter_num else True)
                    continue
            if (metrics_min==True and cv_score<scoring) or (metrics_min==False and cv_score>scoring):
                scoring = cv_score
                self.best_params = self.hp.params.copy()
                self.best_params_history[i] = {'score':scoring, 'best_params':self.best_params.copy()}
                if self.params.model_init in ['XGBClassifier', 'XGBRegressor']:
                    if save_model_dir is not None:
                        if save_model_name is None:
                            save_model_name = 'RS'
                        model.save_model(os.path.join(save_model_dir, f"{save_model_name}_model.json"))
                        with open(os.path.join(save_model_dir, f"{save_model_name}_params.json"),'w') as f:
                            json.dump(best_params, f)
            logger.info(f"Random search progress: {i/iter_num*100:.1f}%, best score: {scoring:.4f}", enter=False if i<iter_num else True)
        logger.info(f"{self.name} random search best score: {scoring:.4f}", close=True, time_mode=1)
        return self.best_params

    def _model_fit_predict(self, data, metrics, index=None, mode=1):
        if mode:
            if len(data)==2:
                self.params.model.fit(data[0].loc[index[0]], data[1][index[0]])
            else:
                self.params.model.fit(data[0].loc[index[0]], data[1][index[0]], sample_weight=data[2][index[0]])
        if index is None:
            cv_pred = pd.Series(self.params.model.predict(data[0]), index=data[1].index)
        else:
            cv_pred = pd.Series(self.params.model.predict(data[0].loc[index[1]]), index=data[1][index[1]].index)
        if len(data)==2:
            if index is None:
                return metrics(data[1], cv_pred)
            else:
                return metrics(data[1][index[1]], cv_pred)
        else:
            if index is None:
                return metrics(data[1], cv_pred, sample_weight=data[2])
            else:
                return metrics(data[1][index[1]], cv_pred, sample_weight=data[2][index[1]])
    
    def _xgb_weight(self, label):
        weight_dict = Counter(label)
        if len(weight_dict)==2:
            weight = int(np.ceil(weight_dict[min(weight_dict)]/weight_dict[max(weight_dict)]))
        else:
            weight_dict = {j:i for i,j in weight_dict.items()}
            weight = int(np.ceil(weight_dict[max(weight_dict)]/weight_dict[min(weight_dict)]))
        self.hp.Choice('scale_pos_weight', [1, weight])