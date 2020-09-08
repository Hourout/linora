```python
hpr = la.param_search.HyperParametersRandom()

hpr.Boolean('a', True)
hpr.Float('b', 0, 10, step=2, default=4)
hpr.Int('c', 0, 10, default=4)
hpr.Choice('d', [20, 21,22, 23])

hpr.params ==> {'a': True, 'b': 4, 'c': 4, 'd': 22}

hpr.update()

hpr.params ==> {'a': False, 'b': 2.09, 'c': 2, 'd': 23}
```
