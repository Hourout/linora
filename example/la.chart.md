

```python
import linora as la

la._hello()
```

    
    ------------------------------------------------------------------------------------
          Linora
    --------------------
          Version      : -- 0.6.0  --
          Author       : JinQing Lee
          License      : Apache-2.0
          Homepage     : https://github.com/Hourout/linora
          Description  : Simple and efficient tools for data mining and data analysis.
    ------------------------------------------------------------------------------------
    


```python
label = [1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1]
label_prob = [0.8, 0.4, 0.2, 0.5, 0.9, 0.2, 0.8, 0.6, 0.1, 0.3, 0.8, 0.3, 0.9, 0.2, 0.84, 
              0.2, 0.5, 0.23, 0.83, 0.71, 0.34, 0.3, 0.2, 0.7, 0.2, 0.8, 0.3, 0.59, 0.26, 0.16, 0.13, 0.8]
label_pred = list(map(lambda x:round(x), label_prob))
```


```python
la.chart.confusion_matrix_map(label, label_pred)
```




<script>
    require.config({
        paths: {
            'echarts':'https://assets.pyecharts.org/assets/echarts.min'
        }
    });
</script>

    <div id="46a2df4e4dab4d8faab4141578419c7a" style="width:900px; height:500px;"></div>


<script>
    require(['echarts'], function(echarts) {
        var chart_46a2df4e4dab4d8faab4141578419c7a = echarts.init(
            document.getElementById('46a2df4e4dab4d8faab4141578419c7a'), 'white', {renderer: 'canvas'});
        var option_46a2df4e4dab4d8faab4141578419c7a = {
    "color": [
        "#c23531",
        "#2f4554",
        "#61a0a8",
        "#d48265",
        "#749f83",
        "#ca8622",
        "#bda29a",
        "#6e7074",
        "#546570",
        "#c4ccd3",
        "#f05b72",
        "#ef5b9c",
        "#f47920",
        "#905a3d",
        "#fab27b",
        "#2a5caa",
        "#444693",
        "#726930",
        "#b2d235",
        "#6d8346",
        "#ac6767",
        "#1d953f",
        "#6950a1",
        "#918597"
    ],
    "series": [
        {
            "type": "heatmap",
            "name": "Confusion Matrix",
            "data": [
                [
                    0,
                    0,
                    11
                ],
                [
                    0,
                    1,
                    4
                ],
                [
                    1,
                    0,
                    8
                ],
                [
                    1,
                    1,
                    9
                ]
            ],
            "label": {
                "show": true,
                "position": "top",
                "margin": 8,
                "fontSize": 12
            }
        }
    ],
    "legend": [
        {
            "data": [
                "Confusion Matrix"
            ],
            "selected": {
                "Confusion Matrix": true
            },
            "show": true
        }
    ],
    "tooltip": {
        "show": true,
        "trigger": "item",
        "triggerOn": "mousemove|click",
        "axisPointer": {
            "type": "line"
        },
        "textStyle": {
            "fontSize": 14
        },
        "borderWidth": 0
    },
    "yAxis": [
        {
            "type": "Predict",
            "show": true,
            "scale": false,
            "nameLocation": "end",
            "nameGap": 15,
            "gridIndex": 0,
            "inverse": false,
            "offset": 0,
            "splitNumber": 5,
            "minInterval": 0,
            "splitLine": {
                "show": false,
                "lineStyle": {
                    "width": 1,
                    "opacity": 1,
                    "curveness": 0,
                    "type": "solid"
                }
            },
            "data": [
                0,
                1
            ]
        }
    ],
    "title": [
        {
            "text": "Confusion Matrix Map"
        }
    ],
    "visualMap": {
        "type": "continuous",
        "min": 0,
        "max": 11,
        "inRange": {
            "color": [
                "#50a3ba",
                "#eac763",
                "#d94e5d"
            ]
        },
        "calculable": true,
        "splitNumber": 5,
        "orient": "vertical",
        "showLabel": true
    },
    "xAxis": [
        {
            "type": "Actual",
            "show": true,
            "scale": false,
            "nameLocation": "end",
            "nameGap": 15,
            "gridIndex": 0,
            "inverse": false,
            "offset": 0,
            "splitNumber": 5,
            "minInterval": 0,
            "splitLine": {
                "show": false,
                "lineStyle": {
                    "width": 1,
                    "opacity": 1,
                    "curveness": 0,
                    "type": "solid"
                }
            },
            "data": [
                0,
                1
            ]
        }
    ]
};
        chart_46a2df4e4dab4d8faab4141578419c7a.setOption(option_46a2df4e4dab4d8faab4141578419c7a);
    });
</script>





```python
la.chart.gain_curve(label, label_prob)
```




<script>
    require.config({
        paths: {
            'echarts':'https://assets.pyecharts.org/assets/echarts.min'
        }
    });
</script>

    <div id="0cc334513e9b4bc7a792972dcf14a2ff" style="width:900px; height:500px;"></div>


<script>
    require(['echarts'], function(echarts) {
        var chart_0cc334513e9b4bc7a792972dcf14a2ff = echarts.init(
            document.getElementById('0cc334513e9b4bc7a792972dcf14a2ff'), 'white', {renderer: 'canvas'});
        var option_0cc334513e9b4bc7a792972dcf14a2ff = {
    "color": [
        "#c23531",
        "#2f4554",
        "#61a0a8",
        "#d48265",
        "#749f83",
        "#ca8622",
        "#bda29a",
        "#6e7074",
        "#546570",
        "#c4ccd3",
        "#f05b72",
        "#ef5b9c",
        "#f47920",
        "#905a3d",
        "#fab27b",
        "#2a5caa",
        "#444693",
        "#726930",
        "#b2d235",
        "#6d8346",
        "#ac6767",
        "#1d953f",
        "#6950a1",
        "#918597"
    ],
    "series": [
        {
            "type": "line",
            "name": "Gain",
            "symbolSize": 4,
            "showSymbol": true,
            "smooth": true,
            "step": false,
            "data": [
                [
                    0.03,
                    1.0
                ],
                [
                    0.06,
                    1.0
                ],
                [
                    0.09,
                    1.0
                ],
                [
                    0.12,
                    1.0
                ],
                [
                    0.16,
                    1.0
                ],
                [
                    0.19,
                    1.0
                ],
                [
                    0.22,
                    1.0
                ],
                [
                    0.25,
                    0.875
                ],
                [
                    0.28,
                    0.7778
                ],
                [
                    0.31,
                    0.7
                ],
                [
                    0.34,
                    0.7273
                ],
                [
                    0.38,
                    0.6667
                ],
                [
                    0.41,
                    0.6923
                ],
                [
                    0.44,
                    0.6429
                ],
                [
                    0.47,
                    0.6
                ],
                [
                    0.5,
                    0.5625
                ],
                [
                    0.53,
                    0.5882
                ],
                [
                    0.56,
                    0.6111
                ],
                [
                    0.59,
                    0.5789
                ],
                [
                    0.62,
                    0.55
                ],
                [
                    0.66,
                    0.5238
                ],
                [
                    0.69,
                    0.5455
                ],
                [
                    0.72,
                    0.5217
                ],
                [
                    0.75,
                    0.5417
                ],
                [
                    0.78,
                    0.56
                ],
                [
                    0.81,
                    0.5769
                ],
                [
                    0.84,
                    0.5926
                ],
                [
                    0.88,
                    0.5714
                ],
                [
                    0.91,
                    0.5517
                ],
                [
                    0.94,
                    0.5333
                ],
                [
                    0.97,
                    0.5484
                ],
                [
                    1.0,
                    0.5312
                ]
            ],
            "label": {
                "show": false,
                "position": "top",
                "margin": 8,
                "fontSize": 12
            },
            "lineStyle": {
                "width": 1,
                "opacity": 1,
                "curveness": 0,
                "type": "solid"
            },
            "areaStyle": {
                "opacity": 0
            },
            "rippleEffect": {
                "show": true,
                "brushType": "stroke",
                "scale": 2.5,
                "period": 4
            }
        }
    ],
    "legend": [
        {
            "data": [
                "Gain"
            ],
            "selected": {
                "Gain": true
            },
            "show": true
        }
    ],
    "tooltip": {
        "show": true,
        "trigger": "item",
        "triggerOn": "mousemove|click",
        "axisPointer": {
            "type": "line"
        },
        "textStyle": {
            "fontSize": 14
        },
        "borderWidth": 0
    },
    "yAxis": [
        {
            "show": true,
            "scale": false,
            "nameLocation": "end",
            "nameGap": 15,
            "gridIndex": 0,
            "inverse": false,
            "offset": 0,
            "splitNumber": 5,
            "minInterval": 0,
            "splitLine": {
                "show": false,
                "lineStyle": {
                    "width": 1,
                    "opacity": 1,
                    "curveness": 0,
                    "type": "solid"
                }
            }
        }
    ],
    "xAxis": [
        {
            "type": "value",
            "show": true,
            "scale": true,
            "nameLocation": "end",
            "nameGap": 15,
            "gridIndex": 0,
            "inverse": false,
            "offset": 0,
            "splitNumber": 5,
            "minInterval": 0,
            "splitLine": {
                "show": false,
                "lineStyle": {
                    "width": 1,
                    "opacity": 1,
                    "curveness": 0,
                    "type": "solid"
                }
            },
            "data": [
                0.03,
                0.06,
                0.09,
                0.12,
                0.16,
                0.19,
                0.22,
                0.25,
                0.28,
                0.31,
                0.34,
                0.38,
                0.41,
                0.44,
                0.47,
                0.5,
                0.53,
                0.56,
                0.59,
                0.62,
                0.66,
                0.69,
                0.72,
                0.75,
                0.78,
                0.81,
                0.84,
                0.88,
                0.91,
                0.94,
                0.97,
                1.0
            ]
        }
    ],
    "title": [
        {
            "text": "Gain Curve"
        }
    ]
};
        chart_0cc334513e9b4bc7a792972dcf14a2ff.setOption(option_0cc334513e9b4bc7a792972dcf14a2ff);
    });
</script>





```python
la.chart.gini_curve(label, label_prob)
```




<script>
    require.config({
        paths: {
            'echarts':'https://assets.pyecharts.org/assets/echarts.min'
        }
    });
</script>

    <div id="7da830233a65432caef84309e6187e89" style="width:900px; height:500px;"></div>


<script>
    require(['echarts'], function(echarts) {
        var chart_7da830233a65432caef84309e6187e89 = echarts.init(
            document.getElementById('7da830233a65432caef84309e6187e89'), 'white', {renderer: 'canvas'});
        var option_7da830233a65432caef84309e6187e89 = {
    "color": [
        "#c23531",
        "#2f4554",
        "#61a0a8",
        "#d48265",
        "#749f83",
        "#ca8622",
        "#bda29a",
        "#6e7074",
        "#546570",
        "#c4ccd3",
        "#f05b72",
        "#ef5b9c",
        "#f47920",
        "#905a3d",
        "#fab27b",
        "#2a5caa",
        "#444693",
        "#726930",
        "#b2d235",
        "#6d8346",
        "#ac6767",
        "#1d953f",
        "#6950a1",
        "#918597"
    ],
    "series": [
        {
            "type": "line",
            "name": "Gini",
            "symbolSize": 4,
            "showSymbol": true,
            "smooth": true,
            "step": false,
            "data": [
                [
                    0,
                    0
                ],
                [
                    0.03,
                    0.0
                ],
                [
                    0.06,
                    0.0588
                ],
                [
                    0.09,
                    0.0588
                ],
                [
                    0.12,
                    0.0588
                ],
                [
                    0.16,
                    0.0588
                ],
                [
                    0.19,
                    0.1176
                ],
                [
                    0.22,
                    0.1765
                ],
                [
                    0.25,
                    0.2353
                ],
                [
                    0.28,
                    0.2941
                ],
                [
                    0.31,
                    0.2941
                ],
                [
                    0.34,
                    0.3529
                ],
                [
                    0.38,
                    0.3529
                ],
                [
                    0.41,
                    0.3529
                ],
                [
                    0.44,
                    0.3529
                ],
                [
                    0.47,
                    0.4118
                ],
                [
                    0.5,
                    0.4706
                ],
                [
                    0.53,
                    0.4706
                ],
                [
                    0.56,
                    0.4706
                ],
                [
                    0.59,
                    0.4706
                ],
                [
                    0.62,
                    0.5294
                ],
                [
                    0.66,
                    0.5294
                ],
                [
                    0.69,
                    0.5882
                ],
                [
                    0.72,
                    0.5882
                ],
                [
                    0.75,
                    0.5882
                ],
                [
                    0.78,
                    0.5882
                ],
                [
                    0.81,
                    0.6471
                ],
                [
                    0.84,
                    0.7059
                ],
                [
                    0.88,
                    0.7647
                ],
                [
                    0.91,
                    0.8235
                ],
                [
                    0.94,
                    0.8824
                ],
                [
                    0.97,
                    0.9412
                ],
                [
                    1.0,
                    1.0
                ]
            ],
            "label": {
                "show": false,
                "position": "top",
                "margin": 8,
                "fontSize": 12
            },
            "lineStyle": {
                "width": 1,
                "opacity": 1,
                "curveness": 0,
                "type": "solid"
            },
            "areaStyle": {
                "opacity": 0.4
            },
            "rippleEffect": {
                "show": true,
                "brushType": "stroke",
                "scale": 2.5,
                "period": 4
            }
        },
        {
            "type": "line",
            "name": "Random",
            "symbolSize": 4,
            "showSymbol": true,
            "smooth": false,
            "step": false,
            "data": [
                [
                    0,
                    0
                ],
                [
                    0.03,
                    0.03
                ],
                [
                    0.06,
                    0.06
                ],
                [
                    0.09,
                    0.09
                ],
                [
                    0.12,
                    0.12
                ],
                [
                    0.16,
                    0.16
                ],
                [
                    0.19,
                    0.19
                ],
                [
                    0.22,
                    0.22
                ],
                [
                    0.25,
                    0.25
                ],
                [
                    0.28,
                    0.28
                ],
                [
                    0.31,
                    0.31
                ],
                [
                    0.34,
                    0.34
                ],
                [
                    0.38,
                    0.38
                ],
                [
                    0.41,
                    0.41
                ],
                [
                    0.44,
                    0.44
                ],
                [
                    0.47,
                    0.47
                ],
                [
                    0.5,
                    0.5
                ],
                [
                    0.53,
                    0.53
                ],
                [
                    0.56,
                    0.56
                ],
                [
                    0.59,
                    0.59
                ],
                [
                    0.62,
                    0.62
                ],
                [
                    0.66,
                    0.66
                ],
                [
                    0.69,
                    0.69
                ],
                [
                    0.72,
                    0.72
                ],
                [
                    0.75,
                    0.75
                ],
                [
                    0.78,
                    0.78
                ],
                [
                    0.81,
                    0.81
                ],
                [
                    0.84,
                    0.84
                ],
                [
                    0.88,
                    0.88
                ],
                [
                    0.91,
                    0.91
                ],
                [
                    0.94,
                    0.94
                ],
                [
                    0.97,
                    0.97
                ],
                [
                    1.0,
                    1.0
                ]
            ],
            "label": {
                "show": false,
                "position": "top",
                "margin": 8,
                "fontSize": 12
            },
            "lineStyle": {
                "width": 1,
                "opacity": 1,
                "curveness": 0,
                "type": "solid"
            },
            "areaStyle": {
                "opacity": 0.4
            },
            "rippleEffect": {
                "show": true,
                "brushType": "stroke",
                "scale": 2.5,
                "period": 4
            }
        }
    ],
    "legend": [
        {
            "data": [
                "Gini",
                "Random"
            ],
            "selected": {
                "Gini": true,
                "Random": true
            },
            "show": true
        }
    ],
    "tooltip": {
        "show": true,
        "trigger": "item",
        "triggerOn": "mousemove|click",
        "axisPointer": {
            "type": "line"
        },
        "textStyle": {
            "fontSize": 14
        },
        "borderWidth": 0
    },
    "yAxis": [
        {
            "show": true,
            "scale": false,
            "nameLocation": "end",
            "nameGap": 15,
            "gridIndex": 0,
            "inverse": false,
            "offset": 0,
            "splitNumber": 5,
            "minInterval": 0,
            "splitLine": {
                "show": false,
                "lineStyle": {
                    "width": 1,
                    "opacity": 1,
                    "curveness": 0,
                    "type": "solid"
                }
            }
        }
    ],
    "xAxis": [
        {
            "type": "value",
            "show": true,
            "scale": true,
            "nameLocation": "end",
            "nameGap": 15,
            "gridIndex": 0,
            "inverse": false,
            "offset": 0,
            "splitNumber": 5,
            "minInterval": 0,
            "splitLine": {
                "show": false,
                "lineStyle": {
                    "width": 1,
                    "opacity": 1,
                    "curveness": 0,
                    "type": "solid"
                }
            },
            "data": [
                0,
                0.03,
                0.06,
                0.09,
                0.12,
                0.16,
                0.19,
                0.22,
                0.25,
                0.28,
                0.31,
                0.34,
                0.38,
                0.41,
                0.44,
                0.47,
                0.5,
                0.53,
                0.56,
                0.59,
                0.62,
                0.66,
                0.69,
                0.72,
                0.75,
                0.78,
                0.81,
                0.84,
                0.88,
                0.91,
                0.94,
                0.97,
                1.0
            ]
        }
    ],
    "title": [
        {
            "text": "Gini Curve"
        }
    ]
};
        chart_7da830233a65432caef84309e6187e89.setOption(option_7da830233a65432caef84309e6187e89);
    });
</script>





```python

```
