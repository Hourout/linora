import numpy as np

__all__ = ['HyperParametersRandom']

class HyperParametersRandom():
    def __init__(self):
        self.params = {}
        self._space = {}
        
    def Boolean(self, name, default=None):
        self.params[name] = np.random.choice([True, False]) if default is None else default
        if name not in self._space:
            self._space[name] = {'mode':'Boolean', 'default':default}
    
    def Float(self, name, min_value, max_value, step=2, default=None):
        self.params[name] = round(np.random.uniform(min_value, max_value), step) if default is None else default
        if name not in self._space:
            self._space[name] = {'mode':'Float', 'min_value':min_value, 'max_value':max_value, 'step':step, 'default':default}
    
    def Int(self, name, min_value, max_value, default=None):
        self.params[name] = round(np.random.uniform(min_value, max_value)) if default is None else default
        if name not in self._space:
            self._space[name] = {'mode':'Int', 'min_value':min_value, 'max_value':max_value, 'default':default}
        
    def Choice(self, name, values, default=None):
        self.params[name] = np.random.choice(values) if default is None else default
        if name not in self._space:
            self._space[name] = {'mode':'Choice', 'values':values, 'default':default}
    
    def update(self):
        for param_name, param_config in self._space.items():
            if param_config['mode']=='Boolean':
                self.Boolean(param_name)
            if param_config['mode']=='Float':
                self.Float(param_name, param_config['min_value'], param_config['max_value'], param_config['step'])
            if param_config['mode']=='Int':
                self.Int(param_name, param_config['min_value'], param_config['max_value'])
            if param_config['mode']=='Choice':
                self.Choice(param_name, param_config['values'])
