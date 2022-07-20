import numpy as np


class Bar():
    def add_bar(self, name, xdata, ydata, **kwargs):
        """Make a bar plot.
        
        Args:
            name: data name.
            xdata: x-axis data.
            ydata: y-axis data.
            width : float or array-like, default: 0.8
                The width(s) of the bars.

            bottom : float or array-like, default: 0
                The y coordinate(s) of the bars bases.

            align : {'center', 'edge'}, default: 'center'
                Alignment of the bars to the *x* coordinates:

                - 'center': Center the base on the *x* positions.
                - 'edge': Align the left edges of the bars with the *x* positions.
                
            barcolor : color or list of color, optional
                The colors of the bar faces.

            edgecolor : color or list of color, optional
                The colors of the bar edges.

            linewidth : float or array-like, optional
                Width of the bar edge(s). If 0, don't draw edges.

            tick_label : str or list of str, optional
                The tick labels of the bars.
                Default: None (Use default numeric labels.)

            xerr, yerr : float or array-like of shape(N,) or shape(2, N), optional
                If not *None*, add horizontal / vertical errorbars to the bar tips.
                The values are +/- sizes relative to the data:

                - scalar: symmetric +/- values for all bars
                - shape(N,): symmetric +/- values for each bar
                - shape(2, N): Separate - and + values for each bar. First row
                  contains the lower errors, the second row contains the upper
                  errors.
                - *None*: No errorbar. (Default)

                See :doc:`/gallery/statistics/errorbar_features`
                for an example on the usage of ``xerr`` and ``yerr``.

            ecolor : color or list of color, default: 'black'
                The line color of the errorbars.

            capsize : float, default: :rc:`errorbar.capsize`
               The length of the error bar caps in points.

            error_kw : dict, optional
                Dictionary of kwargs to be passed to the `~.Axes.errorbar`
                method. Values of *ecolor* or *capsize* defined here take
                precedence over the independent kwargs.

            log : bool, default: False
                If *True*, set the y-axis to be log scale.

            linestyle: line style, {'-', '--', '-.', ':'}.
                '-' or 'solid': solid line
                '--' or 'dashed': dashed line
                '-.' or 'dashdot': dash-dotted line
                ':' or 'dotted': dotted line
                'none', 'None', ' ', or '': draw nothing
        """
        if 'barcolor' not in kwargs:
            kwargs['color'] = tuple([round(np.random.uniform(0, 1),1) for _ in range(3)])
        elif isinstance(kwargs['barcolor'], dict):
            kwargs['color'] = kwargs.pop('barcolor')['mode']
        else:
            kwargs['color'] = kwargs.pop('barcolor')
        
        self._params.ydata[name]['kwargs'] = kwargs
        self._params.ydata[name]['xdata'] = xdata
        self._params.ydata[name]['ydata'] = ydata
        self._params.ydata[name]['plotmode'] = 'bar'
        return self