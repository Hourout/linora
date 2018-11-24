# Linora
 Linora is a efficent machine learning hyper parameters automated tuning Library,supporting XGBoost、LightGBM、CatBoost and other algorithm that implement by sklearn. 
 
## [API document](https://github.com/Hourout/linora/blob/master/document/English_API.md)

## Install
You can install by pypi：
```
pip install linora
```
You can also install by source code
```
pip install git+git://github.com/Hourout/linora.git
```

## Feature
-XGBoost
  - Support XGBClassifier in RandomSearch、GridSearch
  - Support XGBRegressor in RandomSearch、GridSearch
  - Support XGBRanker in RandomSearch、GridSearch
  - Support cpu、gpu
  - Support fast search、k-fold search

-LightGBM
  - Support LGBClassifier in RandomSearch、GridSearch
  - Support LGBRegressor in RandomSearch、GridSearch
  - Support LGBRanker in RandomSearch、GridSearch
  - Support cpu、gpu
  - Support fast search、k-fold search

## Example
- [linora.XGBRanker](https://github.com/Hourout/linora/blob/master/example/XGBRanker.ipynb)
- [linora.XGBClassifier](https://github.com/Hourout/linora/blob/master/example/XGBClassifier.ipynb)
- [linora.XGBRegressor](https://github.com/Hourout/linora/blob/master/example/XGBRegressor.ipynb)
