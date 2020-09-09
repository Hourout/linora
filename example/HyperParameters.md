- la.param_search.HyperParametersRandom()

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

- la.param_search.HyperParametersGrid()

```python
hpr = la.param_search.HyperParametersGrid()
hpr.Boolean('a', True, rank=0)
hpr.Choice('d', [20, 21,22, 23], rank=0)
hpr.Float('b', 0, 0.3, rounds=1, default=0.4, rank=1)
hpr.Int('c', 0, 5, default=4, rank=1)

for i in hpr.update(0):
    print(hpr.params)

==>
{'a': True, 'b': 0.3, 'c': 5, 'd': 20}
{'a': True, 'b': 0.3, 'c': 5, 'd': 21}
{'a': True, 'b': 0.3, 'c': 5, 'd': 22}
{'a': True, 'b': 0.3, 'c': 5, 'd': 23}
{'a': False, 'b': 0.3, 'c': 5, 'd': 20}
{'a': False, 'b': 0.3, 'c': 5, 'd': 21}
{'a': False, 'b': 0.3, 'c': 5, 'd': 22}
{'a': False, 'b': 0.3, 'c': 5, 'd': 23}

for i in hpr.update(1):
    print(hpr.params)
    
==>
{'a': False, 'b': 0.0, 'c': 0, 'd': 23}
{'a': False, 'b': 0.0, 'c': 1, 'd': 23}
{'a': False, 'b': 0.0, 'c': 2, 'd': 23}
{'a': False, 'b': 0.0, 'c': 3, 'd': 23}
{'a': False, 'b': 0.0, 'c': 4, 'd': 23}
{'a': False, 'b': 0.0, 'c': 5, 'd': 23}
{'a': False, 'b': 0.1, 'c': 0, 'd': 23}
{'a': False, 'b': 0.1, 'c': 1, 'd': 23}
{'a': False, 'b': 0.1, 'c': 2, 'd': 23}
{'a': False, 'b': 0.1, 'c': 3, 'd': 23}
{'a': False, 'b': 0.1, 'c': 4, 'd': 23}
{'a': False, 'b': 0.1, 'c': 5, 'd': 23}
{'a': False, 'b': 0.2, 'c': 0, 'd': 23}
{'a': False, 'b': 0.2, 'c': 1, 'd': 23}
{'a': False, 'b': 0.2, 'c': 2, 'd': 23}
{'a': False, 'b': 0.2, 'c': 3, 'd': 23}
{'a': False, 'b': 0.2, 'c': 4, 'd': 23}
{'a': False, 'b': 0.2, 'c': 5, 'd': 23}
{'a': False, 'b': 0.3, 'c': 0, 'd': 23}
{'a': False, 'b': 0.3, 'c': 1, 'd': 23}
{'a': False, 'b': 0.3, 'c': 2, 'd': 23}
{'a': False, 'b': 0.3, 'c': 3, 'd': 23}
{'a': False, 'b': 0.3, 'c': 4, 'd': 23}
{'a': False, 'b': 0.3, 'c': 5, 'd': 23}
```
