# Linora

Linora是一个简洁高效的数据挖掘和数据分析工具，最大限度让你不使用sklearn也能做相关数据挖掘任务，同时完美兼容pandas，相比skearn运行速度更快且更省内存。


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

- metrics：机器学习评估模块
- metrics charts：机器学习评估可视化模块
- feature columns：特征工程模块
- feature selection：特征选择模块
- image augmentation：图像增强模块
- text processing：文本处理模块
- model param search：模型超参数搜索模块
- sample splits：样本分割模块



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
