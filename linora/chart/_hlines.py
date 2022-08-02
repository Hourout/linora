import numpy as np

class Hlines():
    def add_hlines(self, name, ydata, xmin, xmax, **kwargs):
        """Plot y versus x as lines and/or markers.
        
        Args:
            name: data name.
            ydata: y-axis data, float or array-like.
            xmin, xmax : float or array-like
                Respective beginning and end of each line. If scalars are
                provided, all lines will have same length.
            colors : list of colors, default: :rc:`lines.color`
            linestyles : {'solid', 'dashed', 'dashdot', 'dotted'}, optional
            label : str, default: ''
        """
        if 'colors' not in kwargs and 'colors' not in kwargs and 'linecolor' not in kwargs:
            kwargs['colors'] = [tuple([round(np.random.uniform(0, 1),1) for _ in range(3)])]*len(ydata)
        self._params.ydata[name]['kwargs'] = kwargs
        self._params.ydata[name]['xmin'] = xmin
        self._params.ydata[name]['xmax'] = xmax
        self._params.ydata[name]['ydata'] = ydata
        self._params.ydata[name]['plotmode'] = 'hlines'
        self._params.ydata[name]['plotfunc'] = self._execute_plot_hlines
        return self
    
    def _execute_plot_hlines(self, ax, i, j):
        ax_plot = ax.hlines(j['ydata'], j['xmin'], j['xmax'], **j['kwargs'])