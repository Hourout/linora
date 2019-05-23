![](https://github.com/Hourout/linora/blob/master/image/linora.png)


![PyPI version](https://img.shields.io/pypi/pyversions/linora.svg)
![Github license](https://img.shields.io/github/license/Hourout/linora.svg)
[![PyPI](https://img.shields.io/pypi/v/linora.svg)](https://pypi.python.org/pypi/linora)
![PyPI format](https://img.shields.io/pypi/format/linora.svg)

Linora is a simple and efficient tools for data mining and data analysis.
 


| [API Document](https://github.com/Hourout/linora/blob/master/document/English_API.md) | [API文档](https://github.com/Hourout/linora/blob/master/document/Chinese_API.md) | [中文介绍](https://github.com/Hourout/linora/blob/master/document/Chinese.md) |

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


## Example
[more example](https://github.com/Hourout/linora/blob/master/example/example.md)

```
import linora as la

# plot ks curve
label = [1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1]
label_prob = [0.8, 0.4, 0.2, 0.5, 0.9, 0.2, 0.8, 0.6, 0.1, 0.3, 0.8, 0.3, 0.9, 0.2, 0.84, 
              0.2, 0.5, 0.23, 0.83, 0.71, 0.34, 0.3, 0.2, 0.7, 0.2, 0.8, 0.3, 0.59, 0.26, 0.16, 0.13, 0.8]
la.chart.ks_curve(label, label_prob)
```
![](https://github.com/Hourout/linora/blob/master/image/ks_curve.png)
