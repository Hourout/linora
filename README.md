# Linora



![PyPI version](https://img.shields.io/pypi/pyversions/linora.svg)
![Github license](https://img.shields.io/github/license/Hourout/linora.svg)
[![PyPI](https://img.shields.io/pypi/v/linora.svg)](https://pypi.python.org/pypi/linora)
![PyPI format](https://img.shields.io/pypi/format/linora.svg)

Linora is a efficent machine learning hyper parameters automated tuning Library,supporting XGBoost、LightGBM、CatBoost and other algorithm that implement by sklearn. 
 


## [API Document](https://github.com/Hourout/linora/blob/master/document/English_API.md)
## [API文档](https://github.com/Hourout/linora/blob/master/document/Chinese_API.md)
## [中文介绍](https://github.com/Hourout/linora/blob/master/document/Chinese.md)

## Installation

To install [this verson from PyPI](https://pypi.org/project/linora/), type:

```
pip install linora
```

To get the newest one from this repo (note that we are in the alpha stage, so there may be frequent updates), type:

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
