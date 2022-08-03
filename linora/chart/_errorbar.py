import numpy as np


class Errorbar():
    def add_errorbar(self, name, xdata, ydata, **kwargs):
        """Plot y versus x as lines and/or markers with attached errorbars.
        
        Args:
            name: data name.
            xdata: x-axis data.
            ydata: y-axis data.
            xerr, yerr : float or array-like, shape(N,) or shape(2, N), optional
                The errorbar sizes:

                - scalar: Symmetric +/- values for all data points.
                - shape(N,): Symmetric +/-values for each data point.
                - shape(2, N): Separate - and + values for each bar. First row
                  contains the lower errors, the second row contains the upper
                  errors.
                - *None*: No errorbar.

                Note that all error arrays should have *positive* values.

                See :doc:`/gallery/statistics/errorbar_features`
                for an example on the usage of ``xerr`` and ``yerr``.

            ecolor : color, default: None
                The color of the errorbar lines.  If None, use the color of the
                line connecting the markers.

            elinewidth : float, default: None
                The linewidth of the errorbar lines. If None, the linewidth of
                the current style is used.

            capsize : float, The length of the error bar caps in points.

            capthick : float, default: None
                An alias to the keyword argument *markeredgewidth*.
                This setting is a more sensible name for the property that
                controls the thickness of the error bar cap in points. For
                backwards compatibility, if *mew* or *markeredgewidth* are given,
                then they will over-ride *capthick*. This may change in future
                releases.

            barsabove : bool, default: False
                If True, will plot the errorbars above the plot
                symbols. Default is below.

            lolims, uplims, xlolims, xuplims : bool, default: False
                These arguments can be used to indicate that a value gives only
                upper/lower limits.  In that case a caret symbol is used to
                indicate this. *lims*-arguments may be scalars, or array-likes of
                the same length as *xerr* and *yerr*.  To use limits with inverted
                axes, `~.Axes.set_xlim` or `~.Axes.set_ylim` must be called before
                :meth:`errorbar`.  Note the tricky parameter names: setting e.g.
                *lolims* to True means that the y-value is a *lower* limit of the
                True value, so, only an *upward*-pointing arrow will be drawn!

            errorevery : int or (int, int), default: 1
                draws error bars on a subset of the data. *errorevery* =N draws
                error bars on the points (x[::N], y[::N]).
                *errorevery* =(start, N) draws error bars on the points
                (x[start::N], y[start::N]). e.g. errorevery=(6, 3)
                adds error bars to the data at (x[6], x[9], x[12], x[15], ...).
                Used to avoid overlapping error bars when two series share x-axis
                values.
    
            linestyle: line style, {'-', '--', '-.', ':'}.
                '-' or 'solid': solid line
                '--' or 'dashed': dashed line
                '-.' or 'dashdot': dash-dotted line
                ':' or 'dotted': dotted line
                'none', 'None', ' ', or '': draw nothing
            linecolor: line color, eg. 'blue' or '0.75' or 'g' or '#FFDD44' or (1.0,0.2,0.3) or 'chartreuse'.
            linewidth: line width.
            
            linelink: Set the drawstyle of the plot. The drawstyle determines how the points are connected.
                {'default', 'steps', 'steps-pre', 'steps-mid', 'steps-post'}.
                'default': the points are connected with straight lines.
                'steps-pre': The step is at the beginning of the line segment.
                'steps-mid': The step is halfway between the points.
                'steps-post': The step is at the end of the line segment.
                'steps': is equal to 'steps-pre' and is maintained for backward-compatibility.
            markfill: {'full', 'left', 'right', 'bottom', 'top', 'none'}
                'full': Fill the whole marker with the markerfacecolor.
                'left', 'right', 'bottom', 'top': Fill the marker half at the given side with the markerfacecolor. 
                                                  The other half of the marker is filled with markerfacecoloralt.
                'none': No filling.
            mark: marker style string, 
                {'.': 'point', ',': 'pixel', 'o': 'circle', 'v': 'triangle_down', '^': 'triangle_up', 
                '<': 'triangle_left', '>': 'triangle_right', '1': 'tri_down', '2': 'tri_up', '3': 'tri_left', 
                '4': 'tri_right', '8': 'octagon', 's': 'square', 'p': 'pentagon', '*': 'star', 'h': 'hexagon1', 
                'H': 'hexagon2', '+': 'plus', 'x': 'x', 'D': 'diamond', 'd': 'thin_diamond', '|': 'vline', 
                '_': 'hline', 'P': 'plus_filled', 'X': 'x_filled', 0: 'tickleft', 1: 'tickright', 2: 'tickup', 
                3: 'tickdown', 4: 'caretleft', 5: 'caretright', 6: 'caretup', 7: 'caretdown', 8: 'caretleftbase', 
                9: 'caretrightbase', 10: 'caretupbase', 11: 'caretdownbase', 
                'None': 'nothing', None: 'nothing', ' ': 'nothing', '': 'nothing'}
            markedgecolor: marker edge color.
            markedgewidth: float, marker edge width.
            markfacecolor: marker face color.
            marksize: float, marker size.
            markevery: None or int or (int, int) or slice or list[int] or float or (float, float) or list[bool]
            antialiased: Set whether to use antialiased rendering.
        """
        if 'linecolor' not in kwargs:
            kwargs['color'] = tuple([round(np.random.uniform(0, 1),1) for _ in range(3)])
        elif isinstance(kwargs['linecolor'], dict):
            kwargs['color'] = kwargs.pop('linecolor')['mode']
        else:
            kwargs['color'] = kwargs.pop('linecolor')
        if 'ecolor' not in kwargs:
            kwargs['ecolor'] = 'lightgray' 
        if 'linelink' in kwargs:
            kwargs['drawstyle'] = kwargs.pop('linelink')
        if 'markfill' in kwargs:
            kwargs['fillstyle'] = kwargs.pop('markfill')
        if 'mark' in kwargs:
            kwargs['marker'] = kwargs.pop('mark')
        if 'markedgecolor' in kwargs:
            kwargs['markeredgecolor'] = kwargs.pop('markedgecolor')
        if 'markedgewidth' in kwargs:
            kwargs['markeredgewidth'] = kwargs.pop('markedgewidth')
        if 'markfacecolor' in kwargs:
            kwargs['markerfacecolor'] = kwargs.pop('markfacecolor')
        if 'marksize' in kwargs:
            kwargs['markersize'] = kwargs.pop('marksize')
        self._params.ydata[name]['kwargs'] = kwargs
        self._params.ydata[name]['xdata'] = xdata
        self._params.ydata[name]['ydata'] = ydata
        self._params.ydata[name]['plotmode'] = 'errorbar'
        self._params.ydata[name]['plotfunc'] = self._execute_plot_errorbar
        return self
    
    def _execute_plot_errorbar(self, fig, ax, i, j):
        ax_plot = ax.errorbar(j['xdata'], j['ydata'], **j['kwargs'])
        ax_plot.set_label(i)
    