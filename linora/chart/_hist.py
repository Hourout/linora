import numpy as np


class Hist():
    def add_hist(self, name, xdata, **kwargs):
        """Plot a histogram.
        
        Args:
            name: data name.
            xdata: x-axis data.
            
        """
        if 'color' not in kwargs:
            kwargs['color'] = tuple([round(np.random.uniform(0, 1),1) for _ in range(3)])
        else:
            if isinstance(kwargs['color'], dict):
                kwargs['color'] = kwargs.pop('color')['mode']
        self._params.ydata[name]['kwargs'] = kwargs
        self._params.ydata[name]['xdata'] = xdata
        self._params.ydata[name]['plotmode'] = 'hist'
        return self
    
