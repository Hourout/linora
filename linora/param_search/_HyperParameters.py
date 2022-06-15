import random
import itertools
from collections import defaultdict

import numpy as np

__all__ = ['HyperParametersRandom', 'HyperParametersGrid']


class HyperParametersRandom():
    def __init__(self):
        """Container for both a hyperparameter space, and current values."""
        self._space = defaultdict()
        self.params = defaultdict()
        self.params_history = defaultdict()
        self._update_init = True
        self._update_nums = 0
        
    def Boolean(self, name, default=None):
        """Choice between True and False.

        Args:
            name: Str. Name of parameter. Must be unique.
            default: Default value to return for the parameter. If unspecified, the default value will be None.
        """
        if name not in self._space:
            self._space[name] = {'mode':'Boolean', 'default':default}
            self.params[name] = np.random.choice([True, False]) if default is None else default
        
    def Choice(self, name, values, weight=None, default=None):
        """Choice of one value among a predefined set of possible values.

        Args:
            name: Str. Name of parameter. Must be unique.
            values: List of possible values. Values must be int, float, str, or bool. All values must be of the same type.
            weight: List of possible values, The probabilities associated with each entry in value.
            default: Default value to return for the parameter. If unspecified, the default value will be None.
        """
        if name not in self._space:
            self._space[name] = {'mode':'Choice', 'values':list(values), 'weight':weight, 'default':default}
            self.params[name] = np.random.choice(self._space[name]['values'], p=weight) if default is None else default
    
    def Dependence(self, name, dependent_name, function, default=None):
        """Values generated depending on other parameters.

        Args:
            name: Str. Name of parameter. Must be unique.
            dependent_name: str, dependent params name, must already exist.
            function: A function to transform the value of `dependent_name`.
            default: Default value to return for the parameter. If unspecified, the default value will be None.
        """
        assert dependent_name in self._space, '`dependent_name` must already exist.'
        if name not in self._space:
            self._space[name] = {'mode':'Dependence', 'dependent_name':dependent_name, 'function':function, 'default':default}
            self.params[name] = function(self.params[dependent_name]) if default is None else default
    
    def Fixed(self, name, value):
        """Fixed, untunable value.
        
        Args:
            name: Str. Name of parameter. Must be unique.
            value: Fixed value.
        """
        if name not in self._space:
            self._space[name] = {'mode':'Fixed', 'default':value}
            self.params[name] = value
    
    def Float(self, name, min_value, max_value, rounds=2, default=None):
        """Floating point range, can be evenly divided.

        Args:
            name: Str. Name of parameter. Must be unique.
            min_value: Float. Lower bound of the range.
            max_value: Float. Upper bound of the range.
            rounds: Optional. Int, e.g. 2 mean round(x, 2). smallest meaningful distance between two values. 
            default: Default value to return for the parameter. If unspecified, the default value will be None.
        """
        if name not in self._space:
            self._space[name] = {'mode':'Float', 'min_value':min_value, 'max_value':max_value, 'round':rounds, 'default':default}
            self.params[name] = round(random.uniform(min_value, max_value), rounds) if default is None else default
    
    def Int(self, name, min_value, max_value, default=None):
        """Integer range.
        Note that unlinke Python's range function, max_value is included in the possible values this parameter can take on.

        Args:
            name: Str. Name of parameter. Must be unique.
            min_value: Int. Lower limit of range (included).
            max_value: Int. Upper limit of range (included).
            default: Default value to return for the parameter. If unspecified, the default value will be None.
        """
        if name not in self._space:
            self._space[name] = {'mode':'Int', 'min_value':min_value, 'max_value':max_value+1, 'default':default}
            self.params[name] = np.random.randint(min_value, max_value+1) if default is None else default
        
    def update(self, best_params=None):
        """params update"""
        if self._update_init:
            self._update_init = False
        else:
            for name, config in self._space.items():
                if config['mode']=='Boolean':
                    self.params[name] = np.random.choice([True, False])
                elif config['mode']=='Float':
                    self.params[name] = round(np.random.uniform(config['min_value'], config['max_value']), config['round'])
                elif config['mode']=='Int':
                    self.params[name] = np.random.randint(config['min_value'], config['max_value'])
                elif config['mode']=='Choice':
                    self.params[name] = np.random.choice(config['values'], p=config['weight'])
                elif config['mode']=='Dependence':
                    self.params[name] = config['function'](self.params[config['dependent_name']])
        self.params_history[self._update_nums] = self.params.copy()
        self._update_nums += 1
        if best_params is not None:
            self.best_params = best_params.copy()

    def from_HyperParameters(self, hp):
        """update HyperParametersRandom class.
        
        Args:
            hp: a HyperParametersRandom class.
        """
        for name, config in hp._space.items():
            self._space[name] = config
            self.params[name] = hp.params[name]

class HyperParametersGrid():
    def __init__(self):
        """Container for both a hyperparameter space, and current values."""
        self._space = defaultdict()
        self.params = defaultdict()
        self.params_history = defaultdict()
        self._update_init = True
        self._update_nums = 0
        self._rank_list = []
        self._rank = defaultdict(list)
        self._dependent = defaultdict(list)
        
        
    def Boolean(self, name, default=None, rank=0):
        """Choice between True and False.

        Args:
            name: Str. Name of parameter. Must be unique.
            default: Default value to return for the parameter. If unspecified, the default value will be None.
            rank: Int, default 0, Importance ordering of parameters, smaller is more important, rank should be greater than 1.
        """
        if name not in self._space:
            self._space[name] = {'mode':'Boolean', 'default':default, 'rank':rank, 'values':[True, False]}
            if rank>0:
                self._rank[rank].append(name)
            self.params[name] = np.random.choice([True, False]) if default is None else default
        
    def Choice(self, name, values, default=None, rank=0):
        """
        Choice of one value among a predefined set of possible values.

        Args:
            name: Str. Name of parameter. Must be unique.
            values: List of possible values. Values must be int, float, str, or bool. All values must be of the same type.
            default: Default value to return for the parameter. If unspecified, the default value will be None.
            rank: Int, default 0, Importance ordering of parameters, smaller is more important, rank should be greater than 1.
        """
        if name not in self._space:
            self._space[name] = {'mode':'Choice', 'default':default, 'rank':rank, 'values':list(values)}
            if rank>0:
                self._rank[rank].append(name)
            self.params[name] = random.choice(self._space[name]['values']) if default is None else default
        
    def Dependence(self, name, dependent_name, function, default=None):
        """Values generated depending on other parameters.

        Args:
            name: Str. Name of parameter. Must be unique.
            dependent_name: str, dependent params name, must already exist.
            function: A function to transform the value of `dependent_name`.
            default: Default value to return for the parameter. If unspecified, the default value will be None.
        """
        assert dependent_name in self._space, '`dependent_name` must already exist.'
        if name not in self._space:
            self._space[name] = {'mode':'Dependence', 'default':default,
                                 'dependent_name':dependent_name, 'function':function}
            self._dependent[dependent_name].append(name)
            self.params[name] = function(self.params[dependent_name]) if default is None else default
    
    def Fixed(self, name, value):
        """Fixed, untunable value.
        
        Args:
            name: Str. Name of parameter. Must be unique.
            value: Fixed value.
        """
        if name not in self._space:
            self._space[name] = {'mode':'Fixed', 'default':value}
            self.params[name] = value
        
    def Float(self, name, min_value, max_value, rounds=1, default=None, rank=0):
        """Floating point range, can be evenly divided.

        Args:
            name: Str. Name of parameter. Must be unique.
            min_value: Float. Lower bound of the range.
            max_value: Float. Upper bound of the range.
            rounds: Optional. Int, e.g. 2 mean round(x, 2). smallest meaningful distance between two values. 
            default: Default value to return for the parameter. If unspecified, the default value will be None.
            rank: Int, default 0, Importance ordering of parameters, smaller is more important, rank should be greater than 1.
        """
        if name not in self._space:
            self._space[name] = {'mode':'Float', 'default':default, 'rank':rank, 
                                 'values':np.linspace(min_value, max_value, round((max_value-min_value)*10**rounds)+1).round(rounds).tolist()}
            if rank>0:
                self._rank[rank].append(name)
            self.params[name] = round(random.uniform(min_value, max_value), rounds) if default is None else default
        
    def Int(self, name, min_value, max_value, default=None, rank=0):
        """Integer range.
        Note that unlinke Python's range function, max_value is included in the possible values this parameter can take on.

        Args:
            name: Str. Name of parameter. Must be unique.
            min_value: Int. Lower limit of range (included).
            max_value: Int. Upper limit of range (included).
            default: Default value to return for the parameter. If unspecified, the default value will be None.
            rank: Int, default 0, Importance ordering of parameters, smaller is more important, rank should be greater than 1.
        """
        if name not in self._space:
            self._space[name] = {'mode':'Int', 'default':default, 'rank':rank, 'values':list(range(min_value, max_value+1))}
            if rank>0:
                self._rank[rank].append(name)
            self.params[name] = round(random.uniform(min_value, max_value)) if default is None else default
    
    def _rank_list_func(self):
        rank = sorted(self._rank)
        for i in rank:
            for v in itertools.product(*[self._space[j]['values'] for j in self._rank[i]]):
                grid = {name:name_value for name, name_value in zip(self._rank[i], v)}
                for n in list(grid):
                    if n in self._dependent:
                        for m in self._dependent[n]:
                            grid.update({m:self._space[m]['function'](grid[n])})
                self._rank_list.append(grid.copy())
        self._cardinality = len(self._rank_list)
    
    def update(self, best_params):
        """params update
        
        Args:
            best_params: a best params dict.
        """
        if self._update_init:
            if not self._rank_list:
                self._rank_list_func()
            self._update_init = False
        else:
            if best_params is not None:
                self.best_params = best_params.copy()
                self.params = best_params.copy()
            self.params.update(self._rank_list[self._update_nums-1])
        self.params_history[self._update_nums] = self.params.copy()
        self._update_nums += 1
        
    def cardinality(self):
        """number of grid searches."""
        if not self._rank_list:
            self._rank_list_func()
        return self._cardinality
    
    def grid_space(self):
        """grid search space."""
        return self._rank_list
    
    def from_HyperParameters(self, hp):
        """update HyperParametersGrid class.
        
        Args:
            hp: a HyperParametersGrid class.
        """
        for name, config in hp._space.items():
            self._space[name] = config
            self.params[name] = hp.params[name]
        for rank, name_list in hp._rank.items():
            if rank not in self._rank:
                self._rank[rank].append(name_list)
                continue
            for name in name_list:
                for r, n in self._rank.items():
                    if rank==r:
                        if name not in self._rank[rank]:
                            self._rank[rank].append(name)
                    else:
                        if name in self._rank[rank]:
                            self._rank[rank].remove(name)
                        