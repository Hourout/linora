import numpy as np

class Vlines():
    def add_vlines(self, name, xdata, ymin, ymax, **kwargs):
        """Plot vertical lines at each *x* from *ymin* to *ymax*.
        
        Args:
            name: data name.
            xdata: x-axis data, float or array-like.
            ymin, ymax : float or array-like
                Respective beginning and end of each line. If scalars are
                provided, all lines will have same length.
            colors : list of colors, default: :rc:`lines.color`
            linestyles : {'solid', 'dashed', 'dashdot', 'dotted'}, optional
            label : str, default: ''
        """
        if isinstance(xdata, (int, float)):
            xdata = [xdata]
        if 'color' not in kwargs and 'colors' not in kwargs and 'linecolor' not in kwargs:
            kwargs['colors'] = [self._params.color.pop(0)[1]]*len(xdata)
        self._params.ydata[name]['kwargs'] = kwargs
        self._params.ydata[name]['ymin'] = ymin
        self._params.ydata[name]['ymax'] = ymax
        self._params.ydata[name]['xdata'] = xdata
        self._params.ydata[name]['plotmode'] = 'vlines'
        self._params.ydata[name]['plotfunc'] = self._execute_plot_vlines
        return self
    
    def _execute_plot_vlines(self, fig, ax, i, j):
        ax_plot = ax.vlines(j['xdata'], j['ymin'], j['ymax'], **j['kwargs'])