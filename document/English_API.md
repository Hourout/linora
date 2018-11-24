# Linora API Document

```python
linora.XGBClassifier.GridSearch(feature, label, loss, metrics, scoring=0.5, cv=5, cv_num=3, metrics_min=True, speedy=True, speedy_param=(20000, 0.3), gpu=False)
```

参数：
- feature：pd.DataFrame,shape=(n_sample, n_feature),feature variables;
- label：pd.Series,shape=(n_sample),target variable;
- loss：XGBoost Built in string loss or self-define loss function;
- metrics：self-define metrics function;
- scoring/：float，default 0.5,self-define metrics function baseline;
- cv：int，default 5 ,cross-validation 5 fold,only work for apply k-fold search;
- cv_num：int，default 3 ,min search time,when speedy=False,cv should larger than or equal to cv_num, or cv_num would be invalid， when speedy=True，cv would be invalid，only work for cv_num;
- metrics_min：bool，default True，whether metrics is lesser the better;
- speedy：bool，default True,whether can user fast search;
- speedy_param：tuple,default (20000, 0.3)，only work when speedy=True，first param means min sample number in fast search,second param means min samlpe rate in [0,1],and get the i
min number between them;
- gpu：bool，default False，whether use gpu;


```python
linora.XGBClassifier.RandomSearch(feature, label, loss, metrics, iter_num=1000, scoring=0.5, cv=5, cv_num=3, metrics_min=True, speedy=True, speedy_param=(20000, 0.3), gpu=False)
```

参数：
- feature：pd.DataFrame,shape=(n_sample, n_feature),feature variables;
- label：pd.Series,shape=(n_sample),target variable;
- loss：XGBoost Built in string loss or self-define loss function;
- metrics：self-define metrics function;
- iter_num：int，default 1000,the numberof times in random hyperparameter search；
- scoring/：float，default 0.5,self-define metrics function baseline;
- cv：int，default 5 ,cross-validation 5 fold,only work for apply k-fold search;
- cv_num：int，default 3 ,min search time,when speedy=False,cv should larger than or equal to cv_num, or cv_num would be invalid， when speedy=True，cv would be invalid，only work for cv_num;
- metrics_min：bool，default True，whether metrics is lesser the better;
- speedy：bool，default True,whether can user fast search;
- speedy_param：tuple,default (20000, 0.3)，only work when speedy=True，first param means min sample number in fast search,second param means min samlpe rate in [0,1],and get the i
min number between them;
- gpu：bool，default False，whether use gpu;

```python
linora.XGBRegressor.GridSearch(feature, label, loss, metrics, scoring=0.5, cv=5, cv_num=3, metrics_min=True, speedy=True, speedy_param=(20000, 0.3), gpu=False)
```

参数：
- feature：pd.DataFrame,shape=(n_sample, n_feature),feature variables;
- label：pd.Series,shape=(n_sample),target variable;
- loss：XGBoost Built in string loss or self-define loss function;
- metrics：self-define metrics function;
- scoring/：float，default 0.5,self-define metrics function baseline;
- cv：int，default 5 ,cross-validation 5 fold,only work for apply k-fold search;
- cv_num：int，default 3 ,min search time,when speedy=False,cv should larger than or equal to cv_num, or cv_num would be invalid， when speedy=True，cv would be invalid，only work for cv_num;
- metrics_min：bool，default True，whether metrics is lesser the better;
- speedy：bool，default True,whether can user fast search;
- speedy_param：tuple,default (20000, 0.3)，only work when speedy=True，first param means min sample number in fast search,second param means min samlpe rate in [0,1],and get the i
min number between them;
- gpu：bool，default False，whether use gpu;


```python
linora.XGBRegressor.RandomSearch(feature, label, loss, metrics, iter_num=1000, scoring=0.5, cv=5, cv_num=3, metrics_min=True, speedy=True, speedy_param=(20000, 0.3), gpu=False)
```

参数：
- feature：pd.DataFrame,shape=(n_sample, n_feature),feature variables;
- label：pd.Series,shape=(n_sample),target variable;
- loss：XGBoost Built in string loss or self-define loss function;
- metrics：self-define metrics function;
- iter_num：int，default 1000,the numberof times in random hyperparameter search；
- scoring/：float，default 0.5,self-define metrics function baseline;
- cv：int，default 5 ,cross-validation 5 fold,only work for apply k-fold search;
- cv_num：int，default 3 ,min search time,when speedy=False,cv should larger than or equal to cv_num, or cv_num would be invalid， when speedy=True，cv would be invalid，only work for cv_num;
- metrics_min：bool，default True，whether metrics is lesser the better;
- speedy：bool，default True,whether can user fast search;
- speedy_param：tuple,default (20000, 0.3)，only work when speedy=True，first param means min sample number in fast search,second param means min samlpe rate in [0,1],and get the i
min number between them;
- gpu：bool，default False，whether use gpu;


```python
linora.XGBRanker.GridSearch(feature, label, group, metrics, scoring=0.5, cv=5, cv_num=3, metrics_min=True, speedy=True, speedy_param=(20000, 0.3), gpu=False)
```

参数：
- feature：pd.DataFrame,shape=(n_sample, n_feature),feature variables;
- label：pd.Series,shape=(n_sample),target variable;
- loss：XGBoost Built in string loss or self-define loss function;
- metrics：self-define metrics function;
- scoring/：float，default 0.5,self-define metrics function baseline;
- cv：int，default 5 ,cross-validation 5 fold,only work for apply k-fold search;
- cv_num：int，default 3 ,min search time,when speedy=False,cv should larger than or equal to cv_num, or cv_num would be invalid， when speedy=True，cv would be invalid，only work for cv_num;
- metrics_min：bool，default True，whether metrics is lesser the better;
- speedy：bool，default True,whether can user fast search;
- speedy_param：tuple,default (20000, 0.3)，only work when speedy=True，first param means min sample number in fast search,second param means min samlpe rate in [0,1],and get the i
min number between them;
- gpu：bool，only False;

```python
linora.XGBRanker.RandomSearch(feature, label, group, metrics, iter_num=1000, scoring=0.5, cv=5, cv_num=3, metrics_min=True, speedy=True, speedy_param=(20000, 0.3), gpu=False)
```

参数：
- feature：pd.DataFrame,shape=(n_sample, n_feature),feature variables;
- label：pd.Series,shape=(n_sample),target variable;
- loss：XGBoost Built in string loss or self-define loss function;
- metrics：self-define metrics function;
- iter_num：int，default 1000,the numberof times in random hyperparameter search；
- scoring/：float，default 0.5,self-define metrics function baseline;
- cv：int，default 5 ,cross-validation 5 fold,only work for apply k-fold search;
- cv_num：int，default 3 ,min search time,when speedy=False,cv should larger than or equal to cv_num, or cv_num would be invalid， when speedy=True，cv would be invalid，only work for cv_num;
- metrics_min：bool，default True，whether metrics is lesser the better;
- speedy：bool，default True,whether can user fast search;
- speedy_param：tuple,default (20000, 0.3)，only work when speedy=True，first param means min sample number in fast search,second param means min samlpe rate in [0,1],and get the i
min number between them;
- gpu：bool，only False;

