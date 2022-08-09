import numpy as np


class Fillline():
    def add_fillline(self, name, xdata, ydata, ydata2=0, **kwargs):
        """A scatter plot of *y* vs. *x* with varying marker size and/or color.
        
        Args:
            name: data name.
            xdata: x-axis data.
            ydata: y-axis data.
            ydata2: y-axis data.
            where : array of bool (length N), optional
                Define *where* to exclude some horizontal regions from being filled.
                The filled regions are defined by the coordinates ``x[where]``.
                More precisely, fill between ``x[i]`` and ``x[i+1]`` if
                ``where[i] and where[i+1]``.  Note that this definition implies
                that an isolated *True* value between two *False* values in *where*
                will not result in filling.  Both sides of the *True* position
                remain unfilled due to the adjacent *False* values.

            interpolate : bool, default: False
                This option is only relevant if *where* is used and the two curves
                are crossing each other.

                Semantically, *where* is often used for *y1* > *y2* or
                similar.  By default, the nodes of the polygon defining the filled
                region will only be placed at the positions in the *x* array.
                Such a polygon cannot describe the above semantics close to the
                intersection.  The x-sections containing the intersection are
                simply clipped.

                Setting *interpolate* to *True* will calculate the actual
                intersection point and extend the filled region up to this point.

            step : {'pre', 'post', 'mid'}, optional
                Define *step* if the filling should be a step function,
                i.e. constant in between *x*.  The value determines where the
                step will occur:

                - 'pre': The y value is continued constantly to the left from
                  every *x* position, i.e. the interval ``(x[i-1], x[i]]`` has the
                  value ``y[i]``.
                - 'post': The y value is continued constantly to the right from
                  every *x* position, i.e. the interval ``[x[i], x[i+1])`` has the
                  value ``y[i]``.
                - 'mid': Steps occur half-way between the *x* positions.
    
            linestyle: line style, {'-', '--', '-.', ':'}.
                       '-' or 'solid': solid line
                       '--' or 'dashed': dashed line
                       '-.' or 'dashdot': dash-dotted line
                       ':' or 'dotted': dotted line
                       'none', 'None', ' ', or '': draw nothing
            linecolor: line color, eg. 'blue' or '0.75' or 'g' or '#FFDD44' or (1.0,0.2,0.3) or 'chartreuse'.
            linewidth: line width.
            alpha: float, default: None, The alpha blending value, between 0 (transparent) and 1 (opaque).
        """
        if 'linecolor' not in kwargs:
            kwargs['color'] = self._params.color.pop(0)[1]#tuple([round(np.random.uniform(0, 1),1) for _ in range(3)])
        elif isinstance(kwargs['linecolor'], dict):
            kwargs['color'] = kwargs.pop('linecolor')['mode']
        else:
            kwargs['color'] = kwargs.pop('linecolor')
        self._params.ydata[name]['kwargs'] = kwargs
        self._params.ydata[name]['xdata'] = xdata
        self._params.ydata[name]['ydata'] = ydata
        self._params.ydata[name]['ydata2'] = ydata2
        self._params.ydata[name]['plotmode'] = 'fillline'
        self._params.ydata[name]['plotfunc'] = self._execute_plot_fillline
        return self
    
    def _execute_plot_fillline(self, fig, ax, i, j):
        ax_plot = ax.fill_between(j['xdata'], j['ydata'], y2=j['ydata2'], **j['kwargs'])
        ax_plot.set_label(i)