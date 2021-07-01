import random
import itertools
from collections import defaultdict

import numpy as np

__all__ = ['HyperParametersRandom', 'HyperParametersGrid']
        
class HyperParametersRandom():
    def __init__(self):
        """Container for both a hyperparameter space, and current values."""
        self.params = {}
        self.space = {}
        
    def Boolean(self, name, default=None):
        """Choice between True and False.

        Args:
            name: Str. Name of parameter. Must be unique.
            default: Default value to return for the parameter. If unspecified, the default value will be None.
        """
        if name not in self.space:
            self.space[name] = {'mode':'Boolean', 'default':default}
        self.params[name] = random.choice([True, False]) if default is None else default
        
    def Float(self, name, min_value, max_value, rounds=2, default=None):
        """Floating point range, can be evenly divided.

        Args:
            name: Str. Name of parameter. Must be unique.
            min_value: Float. Lower bound of the range.
            max_value: Float. Upper bound of the range.
            rounds: Optional. Int, e.g. 2 mean round(x, 2). smallest meaningful distance between two values. 
            default: Default value to return for the parameter. If unspecified, the default value will be None.
        """
        if name not in self.space:
            self.space[name] = {'mode':'Float', 'min_value':min_value, 'max_value':max_value, 'round':rounds, 'default':default}
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
        if name not in self.space:
            self.space[name] = {'mode':'Int', 'min_value':min_value, 'max_value':max_value, 'default':default}
        self.params[name] = round(random.uniform(min_value, max_value)) if default is None else default
        
    def Choice(self, name, values, default=None):
        """Choice of one value among a predefined set of possible values.

        Args:
            name: Str. Name of parameter. Must be unique.
            values: List of possible values. Values must be int, float, str, or bool. All values must be of the same type.
            default: Default value to return for the parameter. If unspecified, the default value will be None.
        """
        if name not in self.space:
            self.space[name] = {'mode':'Choice', 'values':list(values), 'default':default}
        self.params[name] = random.choice(self.space[name]['values']) if default is None else default
    
    def update(self):
        """params update"""
        for param_name, param_config in self.space.items():
            if param_config['mode']=='Boolean':
                self.Boolean(param_name)
            if param_config['mode']=='Float':
                self.Float(param_name, param_config['min_value'], param_config['max_value'], param_config['round'])
            if param_config['mode']=='Int':
                self.Int(param_name, param_config['min_value'], param_config['max_value'])
            if param_config['mode']=='Choice':
                self.Choice(param_name, param_config['values'])


class HyperParametersGrid():
    def __init__(self):
        """Container for both a hyperparameter space, and current values."""
        self.params = {}
        self.space = {}
        self._rank = defaultdict(lambda:[])
        
    def Boolean(self, name, default=None, rank=0):
        """Choice between True and False.

        Args:
            name: Str. Name of parameter. Must be unique.
            default: Default value to return for the parameter. If unspecified, the default value will be None.
            rank: Int, default 0, Importance ordering of parameters.
        """
        if name not in self.space:
            self.space[name] = {'mode':'Boolean', 'default':default, 'values':[True, False]}
            self._rank[rank].append(name)
        self.params[name] = random.choice([True, False]) if default is None else default
        
    def Float(self, name, min_value, max_value, rounds=1, default=None, rank=0):
        """Floating point range, can be evenly divided.

        Args:
            name: Str. Name of parameter. Must be unique.
            min_value: Float. Lower bound of the range.
            max_value: Float. Upper bound of the range.
            rounds: Optional. Int, e.g. 2 mean round(x, 2). smallest meaningful distance between two values. 
            default: Default value to return for the parameter. If unspecified, the default value will be None.
            rank: Int, default 0, Importance ordering of parameters.
        """
        if name not in self.space:
            self.space[name] = {'mode':'Float', 'default':default, 'values':np.linspace(min_value, max_value, round((max_value-min_value)*10**rounds)+1).round(rounds).tolist()}
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
            rank: Int, default 0, Importance ordering of parameters.
        """
        if name not in self.space:
            self.space[name] = {'mode':'Int', 'default':default, 'values':np.linspace(min_value, max_value, max_value-min_value+1, dtype='int').tolist()}
            self._rank[rank].append(name)
        self.params[name] = round(random.uniform(min_value, max_value)) if default is None else default
        
    def Choice(self, name, values, default=None, rank=0):
        """
        Choice of one value among a predefined set of possible values.

        Args:
            name: Str. Name of parameter. Must be unique.
            values: List of possible values. Values must be int, float, str, or bool. All values must be of the same type.
            default: Default value to return for the parameter. If unspecified, the default value will be None.
            rank: Int, default 0, Importance ordering of parameters.
        """
        if name not in self.space:
            self.space[name] = {'mode':'Choice', 'default':default, 'values':list(values)}
            self._rank[rank].append(name)
        self.params[name] = random.choice(self.space[name]['values']) if default is None else default
        
    def update(self, rank):
        """params update, 
        
        Args:
            rank: Int, Importance ordering of parameters.
        return:
            a yield.
        """
        param_list = [self.space[param_name]['values'] for param_name in self._rank[rank]]
        for value in itertools.product(*param_list):
            self.params.update({name:name_value for name, name_value in zip(self._rank[rank], value)})
            yield self.params
