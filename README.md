![](https://github.com/Hourout/linora/blob/master/image/linora.png)


![PyPI version](https://img.shields.io/pypi/pyversions/linora.svg)
![Github license](https://img.shields.io/github/license/Hourout/linora.svg)
[![PyPI](https://img.shields.io/pypi/v/linora.svg)](https://pypi.python.org/pypi/linora)
![PyPI format](https://img.shields.io/pypi/format/linora.svg)
![contributors](https://img.shields.io/github/contributors/Hourout/linora)
![downloads](https://img.shields.io/pypi/dm/linora.svg)
[![Documentation](https://img.shields.io/badge/docs-linora-blue.svg)](https://www.yuque.com/jinqing-ps0ax/linora/htibub) 

Linora is a simple and efficient data mining and data analysis tool that allows you to do related data mining tasks without using sklearn to the maximum extent. It is perfectly compatible with pandas and runs faster and saves memory compared to sklearn.


| [API Document](https://www.yuque.com/jinqing-ps0ax/linora/htibub) | [中文介绍](https://github.com/Hourout/linora/blob/master/document/Chinese.md) |

## Installation

To install this verson from [PyPI](https://pypi.org/project/linora/), type:

```
pip install linora -U
```

To get the newest one from this repo (note there may be frequent updates), type:

```
pip install git+https://github.com/Hourout/linora.git
```

## Feature
- metrics
- metrics charts
- feature columns module
- feature selection module
- image augmentation
- text processing
- model param search
- sample splits
- sample
- parallel
- logger
- config
- progbar
- schedulers

## Example

```python
import linora as la

# plot ks curve
label = [1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1]
label_prob = [0.8, 0.4, 0.2, 0.5, 0.9, 0.2, 0.8, 0.6, 0.1, 0.3, 0.8, 0.3, 0.9, 0.2, 0.84, 
              0.2, 0.5, 0.23, 0.83, 0.71, 0.34, 0.3, 0.2, 0.7, 0.2, 0.8, 0.3, 0.59, 0.26, 0.16, 0.13, 0.8]
la.chart.ks_curve(label, label_prob)
```
![](https://github.com/Hourout/linora/blob/master/image/ks_curve.png)

## Contact
Please contact me if you have any related questions or improvements.

[WeChat](https://github.com/Hourout/linora/blob/master/image/hourout_wechat.jpg)
