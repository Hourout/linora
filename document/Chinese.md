# Linora

Linora是一个简洁高效的机器学习超三参数自动化调优的训练库，支持XGBoost、LightGBM、CatBoost以及其他具有sklearn模式的算法实例。


## [API文档](https://github.com/Hourout/linora/blob/master/document/Chinese_API.md)


## 安装
你可以通过pypi进行安装：
```
pip install linora
```
也可以通过源代码安装
```
pip install git+git://github.com/Hourout/linora.git
```


## 功能
-XGBoost
  - 支持XGBClassifier中随机搜索（RandomSearch）、坐标下降搜索（GridSearch）
  - 支持XGBRegressor中随机搜索（RandomSearch）、坐标下降搜索（GridSearch）
  - 支持XGBRanker中随机搜索（RandomSearch）、坐标下降搜索（GridSearch）
  - 支持支持cpu、gpu计算
  - 支持快速搜索、k折搜索

-LightGBM
  - 支持LGBClassifier中随机搜索（RandomSearch）、坐标下降搜索（GridSearch）
  - 支持LGBRegressor中随机搜索（RandomSearch）、坐标下降搜索（GridSearch）
  - 支持LGBRanker中随机搜索（RandomSearch）、坐标下降搜索（GridSearch）
  - 支持支持cpu、gpu计算
  - 支持快速搜索、k折搜索


## Example
- [linora.XGBRanker](https://github.com/Hourout/linora/blob/master/example/XGBRanker.ipynb)
- [linora.XGBClassifier](https://github.com/Hourout/linora/blob/master/example/XGBClassifier.ipynb)
- [linora.XGBRegressor](https://github.com/Hourout/linora/blob/master/example/XGBRegressor.ipynb)
