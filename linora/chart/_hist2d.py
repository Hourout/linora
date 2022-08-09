import numpy as np


class Hist2d():
    def add_hist2d(self, name, xdata, ydata, **kwargs):
        """A scatter plot of *y* vs. *x* with varying marker size and/or color.
        
        Args:
            name: data name.
            xdata: x-axis data.
            ydata: y-axis data.
        """
        if 'color' not in kwargs:
            kwargs['color'] = self._params.color.pop(0)[1]#tuple([round(np.random.uniform(0, 1),1) for _ in range(3)])
        else:
            if isinstance(kwargs['color'], dict):
                kwargs['color'] = kwargs.pop('color')['mode']
        self._params.ydata[name]['kwargs'] = kwargs
        self._params.ydata[name]['xdata'] = xdata
        self._params.ydata[name]['ydata'] = ydata
        self._params.ydata[name]['plotmode'] = 'hist2d'
        self._params.ydata[name]['plotfunc'] = self._execute_plot_hist2d
        return self
    
    def _execute_plot_hist2d(self, fig, ax, i, j):
        ax_plot = ax.hist2d(j['xdata'], j['ydata'], **j['kwargs'])
