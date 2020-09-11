# Linora

Linora是一个简洁高效的数据挖掘和数据分析工具，最大限度让你不使用sklearn也能做相关数据挖掘任务，同时完美兼容pandas，相比sklearn运行速度更快且更省内存。


## [API文档](https://github.com/Hourout/linora/blob/master/document/api.md)


## 安装
你可以通过pypi进行安装：
```
pip3 install linora
```
也可以通过源代码安装最新开发版：
```
pip3 install git+git://github.com/Hourout/linora.git
```


## 功能
| module | 描述 |
| --- | --- |
| la.metrics | 机器学习评估模块 |
| la.chart | 机器学习评估可视化模块 |
| la.feature_column | 特征工程模块 |
| la.feature_selection | 特征选择模块 |
| la.image | 图像增强模块 |
| la.text | 文本处理模块 |
| la.param_search | 模型超参数搜索模块 |
| la.sample_splits | 样本分割模块 |
| la.sample | 样本模块 |


## Example
[more example](https://github.com/Hourout/linora/blob/master/example/readme.md)

```python
import linora as la

# plot ks curve
label = [1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1]
label_prob = [0.8, 0.4, 0.2, 0.5, 0.9, 0.2, 0.8, 0.6, 0.1, 0.3, 0.8, 0.3, 0.9, 0.2, 0.84, 
              0.2, 0.5, 0.23, 0.83, 0.71, 0.34, 0.3, 0.2, 0.7, 0.2, 0.8, 0.3, 0.59, 0.26, 0.16, 0.13, 0.8]
la.chart.ks_curve(label, label_prob)
```
![](https://github.com/Hourout/linora/blob/master/image/ks_curve.png)
