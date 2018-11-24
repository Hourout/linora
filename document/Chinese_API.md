# Linora API Document

```python
linora.XGBClassifier.GridSearch(feature, label, loss, metrics, scoring=0.5, cv=5, cv_num=3, metrics_min=True, speedy=True, speedy_param=(20000, 0.3), gpu=False)
```

参数：
- feature：pd.DataFrame，shape=(n_sample, n_feature)，特征变量；
- label：pd.Series，shape=(n_sample)，目标变量；
- loss：XGBoost内置字符串loss或者自定义损失函数；
- metrics：自定义评估函数；
- scoring/：float，默认0.5，自定义评估函数指标基准分;
- cv：int，默认5，交叉验证折数5，仅使用k折搜索有效；
- cv_num：int，默认为3，最小搜索次数，当speedy=False时，cv应当大于等于cv_num, 不然cv_num参数无效，
          当speedy=True，cv无效，仅使用cv_num；
- metrics_min：bool，默认True，评估指标是否越小越好；
- speedy：bool，默认True，是否使用加速搜索算法；
- speedy_param：tuple，默认(20000, 0.3)，仅在speedy=True时有效，
                第一个参数代表加速搜索算法采样最小样本数，第二个参数代表最小采样比例[0,1]的样本数，取二者最小值。
- gpu：bool，默认False，是否使用gpu；


```python
linora.XGBClassifier.RandomSearch(feature, label, loss, metrics, iter_num=1000, scoring=0.5, cv=5, cv_num=3, metrics_min=True, speedy=True, speedy_param=(20000, 0.3), gpu=False)
```

参数：
- feature：pd.DataFrame，shape=(n_sample, n_feature)，特征变量；
- label：pd.Series，shape=(n_sample)，目标变量；
- loss：XGBoost内置字符串loss或者自定义损失函数；
- metrics：自定义评估函数；
- iter_num：int，默认1000，随机超参搜索次数；
- scoring/：float，默认0.5，自定义评估函数指标基准分;
- cv：int，默认5，交叉验证折数5，仅使用k折搜索有效；
- cv_num：int，默认为3，最小搜索次数，当speedy=False时，cv应当大于等于cv_num, 不然cv_num参数无效，
          当speedy=True，cv无效，仅使用cv_num；
- metrics_min：bool，默认True，评估指标是否越小越好；
- speedy：bool，默认True，是否使用加速搜索算法；
- speedy_param：tuple，默认(20000, 0.3)，仅在speedy=True时有效，
                第一个参数代表加速搜索算法采样最小样本数，第二个参数代表最小采样比例[0,1]的样本数，取二者最小值。
- gpu：bool，默认False，是否使用gpu；


```python
linora.XGBRegressor.GridSearch(feature, label, loss, metrics, scoring=0.5, cv=5, cv_num=3, metrics_min=True, speedy=True, speedy_param=(20000, 0.3), gpu=False)
```

参数：
- feature：pd.DataFrame，shape=(n_sample, n_feature)，特征变量；
- label：pd.Series，shape=(n_sample)，目标变量；
- loss：XGBoost内置字符串loss或者自定义损失函数；
- metrics：自定义评估函数；
- scoring/：float，默认0.5，自定义评估函数指标基准分;
- cv：int，默认5，交叉验证折数5，仅使用k折搜索有效；
- cv_num：int，默认为3，最小搜索次数，当speedy=False时，cv应当大于等于cv_num, 不然cv_num参数无效，
          当speedy=True，cv无效，仅使用cv_num；
- metrics_min：bool，默认True，评估指标是否越小越好；
- speedy：bool，默认True，是否使用加速搜索算法；
- speedy_param：tuple，默认(20000, 0.3)，仅在speedy=True时有效，
                第一个参数代表加速搜索算法采样最小样本数，第二个参数代表最小采样比例[0,1]的样本数，取二者最小值。
- gpu：bool，默认False，是否使用gpu；


```python
linora.XGBRegressor.RandomSearch(feature, label, loss, metrics, iter_num=1000, scoring=0.5, cv=5, cv_num=3, metrics_min=True, speedy=True, speedy_param=(20000, 0.3), gpu=False)
```

参数：
- feature：pd.DataFrame，shape=(n_sample, n_feature)，特征变量；
- label：pd.Series，shape=(n_sample)，目标变量；
- loss：XGBoost内置字符串loss或者自定义损失函数；
- metrics：自定义评估函数；
- iter_num：int，默认1000，随机超参搜索次数；
- scoring/：float，默认0.5，自定义评估函数指标基准分;
- cv：int，默认5，交叉验证折数5，仅使用k折搜索有效；
- cv_num：int，默认为3，最小搜索次数，当speedy=False时，cv应当大于等于cv_num, 不然cv_num参数无效，
          当speedy=True，cv无效，仅使用cv_num；
- metrics_min：bool，默认True，评估指标是否越小越好；
- speedy：bool，默认True，是否使用加速搜索算法；
- speedy_param：tuple，默认(20000, 0.3)，仅在speedy=True时有效，
                第一个参数代表加速搜索算法采样最小样本数，第二个参数代表最小采样比例[0,1]的样本数，取二者最小值。
- gpu：bool，默认False，是否使用gpu；


```python
linora.XGBRanker.GridSearch(feature, label, group, metrics, scoring=0.5, cv=5, cv_num=3, metrics_min=True, speedy=True, speedy_param=(20000, 0.3), gpu=False)
```

参数：
- feature：pd.DataFrame，shape=(n_sample, n_feature)，特征变量；
- label：pd.Series，shape=(n_sample)，目标变量；
- group：pd.Series，shape=(n_sample)，排序分组变量；
- metrics：自定义评估函数；
- scoring/：float，默认0.5，自定义评估函数指标基准分;
- cv：int，默认5，交叉验证折数5，仅使用k折搜索有效；
- cv_num：int，默认为3，最小搜索次数，当speedy=False时，cv应当大于等于cv_num, 不然cv_num参数无效，
          当speedy=True，cv无效，仅使用cv_num；
- metrics_min：bool，默认True，评估指标是否越小越好；
- speedy：bool，默认True，是否使用加速搜索算法；
- speedy_param：tuple，默认(20000, 0.3)，仅在speedy=True时有效，
                第一个参数代表加速搜索算法采样最小样本数，第二个参数代表最小采样比例[0,1]的样本数，取二者最小值。
- gpu：bool，默认False，是否使用gpu，仅使用False；


```python
linora.XGBRanker.RandomSearch(feature, label, loss, metrics, iter_num=1000, scoring=0.5, cv=5, cv_num=3, metrics_min=True, speedy=True, speedy_param=(20000, 0.3), gpu=False)
```

参数：
- feature：pd.DataFrame，shape=(n_sample, n_feature)，特征变量；
- label：pd.Series，shape=(n_sample)，目标变量；
- group：pd.Series，shape=(n_sample)，排序分组变量；
- metrics：自定义评估函数；
- iter_num：int，默认1000，随机超参搜索次数；
- scoring/：float，默认0.5，自定义评估函数指标基准分;
- cv：int，默认5，交叉验证折数5，仅使用k折搜索有效；
- cv_num：int，默认为3，最小搜索次数，当speedy=False时，cv应当大于等于cv_num, 不然cv_num参数无效，
          当speedy=True，cv无效，仅使用cv_num；
- metrics_min：bool，默认True，评估指标是否越小越好；
- speedy：bool，默认True，是否使用加速搜索算法；
- speedy_param：tuple，默认(20000, 0.3)，仅在speedy=True时有效，
                第一个参数代表加速搜索算法采样最小样本数，第二个参数代表最小采样比例[0,1]的样本数，取二者最小值。
- gpu：bool，默认False，是否使用gpu，仅使用False；
