import os
import json
from collections import Counter

import numpy as np
import pandas as pd

from linora.utils._config import Config
from linora.utils._logger import Logger
from linora.sample._fold import kfold, train_test_split
from linora.param_search._config import model_hp
from linora.param_search._config import __xgboost_version__, __lightgbm_version__


__all__ = ['RandomSearch', 'GridSearch']


class BaseSearch():
    def __init__(self, model, hp=None, name=None, method='random'):
        self.params = Config()
        if name is not None:
            self.params.name = name
        elif method=='random':
            self.params.name = 'RS'
        elif method=='grid':
            self.params.name = 'GS'
        self.params.model_init = model
        self.params.model_name = ''
        self.params.method = method
        if model in ['XGBClassifier', 'XGBRegressor', 'LGBMClassifier', 'LGBMRegressor']:
            self.hp = model_hp(model=model, method=method)
            if hp is not None:
                self.hp.from_HyperParameters(hp)
            self.params.model_name = model
            if model=='XGBClassifier':
                import xgboost as xgb
                assert xgb.__version__>=__xgboost_version__, f'xgboost version should be >={__xgboost_version__}.'
                self.params.model_init = xgb.XGBClassifier
            elif model=='XGBRegressor':
                import xgboost as xgb
                assert xgb.__version__>=__xgboost_version__, f'xgboost version should be >={__xgboost_version__}.'
                self.params.model_init = xgb.XGBRegressor
            elif model=='LGBMClassifier':
                import lightgbm as lgb
                assert lgb.__version__>=__lightgbm_version__, f'lightgbm version should be >={__lightgbm_version__}.'
                self.params.model_init = lgb.LGBMClassifier
            elif model=='LGBMRegressor':
                import lightgbm as lgb
                assert lgb.__version__>=__lightgbm_version__, f'lightgbm version should be >={__lightgbm_version__}.'
                self.params.model_init = lgb.LGBMRegressor
        else:
            self.hp = hp
        self.best_params = dict()
        self.best_params_history = dict()

    def search(self, train_data, test_data, metrics, valid_data=None,
               iter_num=None, cv=3, metrics_min=True, 
               train_test_diff=0.03, train_size=1,
               save_model_dir=None, save_model_name=None):
        """model params search method.

        Args:
            train_data: A list of (X, y, sample_weight) tuple pairs to use as train sets.
            test_data: A list of (X, y, sample_weight) tuple pairs to use as test sets.
            metrics: model metrics function.
            valid_data: A list of (X, y, sample_weight) tuple pairs to use as validation sets.
            iter_num: search count.
            cv: cross validation fold.
            metrics_min: metrics value whether the smaller the better.
            train_size: train data size.
            train_test_diff: train error and test error must be less than `train_test_diff`.
            save_model_dir: str, save model folder, only work with model='XGBClassifier' or 'XGBRegressor'.
            save_model_name: str, save model name prefix, only work with model='XGBClassifier' or 'XGBRegressor'.
        Returns:
            a best model params dict.
        Raises:
            params error.
        """
        logger = Logger(name=self.params.name)
        logger.info(f"Start hyperparameter {self.params.method} search.")
        logger.info(f"train data: {int(len(train_data[0])*train_size)}")
        logger.info(f"test  data: {int(len(test_data[0]))}")
        if valid_data is not None:
            logger.info(f"valid data: {int(len(valid_data[0]))}")
            cv_score_list = []
        import warnings
        warnings.filterwarnings("ignore")
        
        if self.params.model_name=='XGBClassifier':
            self._xgb_weight(train_data[1])
        
        if self.params.method=='grid':
            if iter_num is None:
                iter_num = self.hp.cardinality()
            else:
                iter_num = min(iter_num, self.hp.cardinality())
        if iter_num is None:
            iter_num = 100
        scoring = None
        for i in range(1, iter_num+1):
            self.hp.update(self.best_params)
            self.params.model = self.params.model_init(**self.hp.params)
            score = []
            for _ in range(cv):
                index = train_test_split(train_data[0], train_data[1], 1-train_size, seed=np.random.choice(range(100), 1)[0])

                self._model_fit(train_data, index)
                train_metrics = self._model_metrics(train_data, metrics)
                test_metrics = self._model_metrics(test_data, metrics)
                if (train_metrics-test_metrics)<train_test_diff:
                    score.append(test_metrics)

            if not score:
                continue
            cv_score = np.mean(score)
            if valid_data is not None:
                cv_score_list.append(cv_score)
                cv_score_list.sort()
                threshold = cv_score_list[int(len(cv_score_list)*(0.2 if metrics_min else 0.8))]
                if (metrics_min==True and threshold>=cv_score) or (metrics_min==False and threshold<=cv_score):
                    cv_score = self._model_metrics(valid_data, metrics)
                else:
                    logger.info(f"Model {self.params.method} search progress: {i/iter_num*100:.1f}%, best score: {scoring:.4f}", enter=False if i<iter_num else True)
                    continue
            if scoring is None:
                scoring = cv_score
            if (metrics_min==True and cv_score<=scoring) or (metrics_min==False and cv_score>=scoring):
                scoring = cv_score
                self.best_params = self.hp.params.copy()
                self.best_params_history[i] = {'score':scoring, 'best_params':self.best_params.copy()}
                if self.params.model_name in ['XGBClassifier', 'XGBRegressor']:
                    if save_model_dir is not None:
                        if save_model_name is None:
                            save_model_name = self.params.name
                        model.save_model(os.path.join(save_model_dir, f"{save_model_name}_model.json"))
                        with open(os.path.join(save_model_dir, f"{save_model_name}_params.json"),'w') as f:
                            json.dump(best_params, f)
            logger.info(f"Model {self.params.method} search progress: {i/iter_num*100:.1f}%, best score: {scoring:.4f}", enter=False if i<iter_num else True)
        logger.info(f"Model {self.params.method} search best score: {scoring:.4f}", close=True, time_mode=1)
        return self.best_params

    def _model_fit(self, data, index):
        if len(data)==2:
            self.params.model.fit(data[0].loc[index[0]], data[1][index[0]])
        else:
            self.params.model.fit(data[0].loc[index[0]], data[1][index[0]], sample_weight=data[2][index[0]])
    
    def _model_metrics(self, data, metrics):
        pred = pd.Series(self.params.model.predict_proba(data[0])[:,1], index=data[1].index)
        if len(data)==2:
            return metrics(data[1], pred)
        else:
            return metrics(data[1], pred, sample_weight=data[2])
    
    def _xgb_weight(self, label):
        weight_dict = Counter(label)
        if len(weight_dict)==2:
            weight = int(np.ceil(weight_dict[min(weight_dict)]/weight_dict[max(weight_dict)]))
        else:
            weight_dict = {j:i for i,j in weight_dict.items()}
            weight = int(np.ceil(weight_dict[max(weight_dict)]/weight_dict[min(weight_dict)]))
        if self.params.method=='grid':
            self.hp.Choice('scale_pos_weight', [1, weight], weight, rank=6)
        else:
            self.hp.Choice('scale_pos_weight', [1, weight])
        
        
class RandomSearch(BaseSearch):
    """model params search use RandomSearch method.
    
    Args:
        model: a model object or one of ['XGBClassifier', 'XGBRegressor', 'LGBMClassifier', 'LGBMRegressor'].
        hp: a la.param_search.HyperParametersRandom object.
        name: logger name.
    """
    def __init__(self, model, hp=None, name=None):
        super(RandomSearch, self).__init__(model, hp=hp, name=name, method='random')


class GridSearch(BaseSearch):
    """model params search use GridSearch method.
    
    Args:
        model: a model object or one of ['XGBClassifier', 'XGBRegressor', 'LGBMClassifier', 'LGBMRegressor'].
        hp: a la.param_search.HyperParametersGrid object.
        name: logger name.
    """
    def __init__(self, model, hp=None, name=None):
        super(GridSearch, self).__init__(model, hp=hp, name=name, method='grid')