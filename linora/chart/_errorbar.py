import numpy as np
import matplotlib.pyplot as plt

from linora.chart._base import Coordinate


__all__ = ['Errorbar']


class Errorbar(Coordinate):
    def __init__(self, *args, **kwargs):
        super(Errorbar, self).__init__()
        if len(args)!=0:
            if isinstance(args[0], dict):
                for i,j in args[0].items():
                    setattr(self._params, i, j)
        if kwargs:
            for i,j in kwargs.items():
                setattr(self._params, i, j)

    
    def add_data(
        self, 
        name, 
        xdata, 
        ydata, 
        yerr=None, 
        xerr=None, 
        ecolor=None,
        elinewidth=None, 
        capsize=None,
        barsabove=False,
        lolims=False,
        uplims=False,
        xlolims=False,
        xuplims=False,
        errorevery=1,
        capthick=None,
                 linestyle=None, linecolor=None, linewidth=None,
                 marker=None, markersize=None, markeredgewidth=None, markeredgecolor=None, 
                 markerfacecolor=None, markevery=None,
                 fillstyle=None, antialiased=None, drawstyle=None, 
                 dash_capstyle=None, solid_capstyle=None, dash_joinstyle=None, solid_joinstyle=None
                ):
        """A scatter plot of *y* vs. *x* with varying marker size and/or color.
        
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

capsize : float, default: :rc:`errorbar.capsize`
    The length of the error bar caps in points.

capthick : float, default: None
    An alias to the keyword argument *markeredgewidth* (a.k.a. *mew*).
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
            drawstyle: Set the drawstyle of the plot. The drawstyle determines how the points are connected.
                       {'default', 'steps', 'steps-pre', 'steps-mid', 'steps-post'}.
                       'default': the points are connected with straight lines.
                       'steps-pre': The step is at the beginning of the line segment.
                       'steps-mid': The step is halfway between the points.
                       'steps-post: The step is at the end of the line segment.
                       'steps': is equal to 'steps-pre' and is maintained for backward-compatibility.
            dash_capstyle: Define how the two endpoints (caps) of an unclosed line are drawn.
                           {'butt', 'projecting', 'round'}
                           'butt': the line is squared off at its endpoint.
                           'projecting': the line is squared off as in butt, 
                                         but the filled in area extends beyond the endpoint a distance of linewidth/2.
                           'round': like butt, but a semicircular cap is added to the end of the line, of radius linewidth/2.
            dash_joinstyle: Define how the connection between two line segments is drawn.
                            {'miter', 'round', 'bevel'}
                            'miter': the "arrow-tip" style. Each boundary of the filled-in area will extend 
                                     in a straight line parallel to the tangent vector of the centerline at 
                                     the point it meets the corner, until they meet in a sharp point.
                            'round': stokes every point within a radius of linewidth/2 of the center lines.
                            'bevel': the "squared-off" style. It can be thought of as a rounded corner where 
                                     the "circular" part of the corner has been cut off.
            fillstyle: {'full', 'left', 'right', 'bottom', 'top', 'none'}
                       'full': Fill the whole marker with the markerfacecolor.
                       'left', 'right', 'bottom', 'top': Fill the marker half at the given side with the markerfacecolor. 
                                                         The other half of the marker is filled with markerfacecoloralt.
                       'none': No filling.
            marker: marker style string, 
                    {'.': 'point', ',': 'pixel', 'o': 'circle', 'v': 'triangle_down', '^': 'triangle_up', 
                    '<': 'triangle_left', '>': 'triangle_right', '1': 'tri_down', '2': 'tri_up', '3': 'tri_left', 
                    '4': 'tri_right', '8': 'octagon', 's': 'square', 'p': 'pentagon', '*': 'star', 'h': 'hexagon1', 
                    'H': 'hexagon2', '+': 'plus', 'x': 'x', 'D': 'diamond', 'd': 'thin_diamond', '|': 'vline', 
                    '_': 'hline', 'P': 'plus_filled', 'X': 'x_filled', 0: 'tickleft', 1: 'tickright', 2: 'tickup', 
                    3: 'tickdown', 4: 'caretleft', 5: 'caretright', 6: 'caretup', 7: 'caretdown', 8: 'caretleftbase', 
                    9: 'caretrightbase', 10: 'caretupbase', 11: 'caretdownbase', 
                    'None': 'nothing', None: 'nothing', ' ': 'nothing', '': 'nothing'}
            markeredgecolor: color
            markeredgewidth: float
            markerfacecolor: color
            markersize: float
            markevery: None or int or (int, int) or slice or list[int] or float or (float, float) or list[bool]
            solid_capstyle: {'butt', 'projecting', 'round'}
            solid_joinstyle: {'miter', 'round', 'bevel'}
            antialiased: Set whether to use antialiased rendering.
        """
        self._params.ydata[name]['xdata'] = xdata
        self._params.ydata[name]['ydata'] = ydata
        self._params.ydata[name]['yerr'] = yerr
        self._params.ydata[name]['xerr'] = xerr
        self._params.ydata[name]['ecolor'] = 'lightgray' if ecolor is None else ecolor
        self._params.ydata[name]['elinewidth'] = elinewidth
        self._params.ydata[name]['capsize'] = capsize
        self._params.ydata[name]['barsabove'] = barsabove
        self._params.ydata[name]['lolims'] = lolims
        self._params.ydata[name]['uplims'] = uplims
        self._params.ydata[name]['xlolims'] = xlolims
        self._params.ydata[name]['xuplims'] = xuplims
        self._params.ydata[name]['errorevery'] = errorevery
        self._params.ydata[name]['capthick'] = capthick
        
        self._params.ydata[name]['linestyle'] = linestyle
        self._params.ydata[name]['linewidth'] = linewidth
        if linecolor is None:
            linecolor = tuple([round(np.random.uniform(0, 1),1) for _ in range(3)])
        elif isinstance(linecolor, dict):
            linecolor = linecolor['mode']
        self._params.ydata[name]['linecolor'] = linecolor
        self._params.ydata[name]['marker'] = marker
        self._params.ydata[name]['markersize'] = markersize
        self._params.ydata[name]['markeredgewidth'] = markeredgewidth
        self._params.ydata[name]['markeredgecolor'] = markeredgecolor
        self._params.ydata[name]['markerfacecolor'] = markerfacecolor
        self._params.ydata[name]['markevery'] = markevery
        self._params.ydata[name]['fillstyle'] = fillstyle
        self._params.ydata[name]['drawstyle'] = drawstyle
        self._params.ydata[name]['dash_capstyle'] = dash_capstyle
        self._params.ydata[name]['solid_capstyle'] = solid_capstyle
        self._params.ydata[name]['dash_joinstyle'] = dash_joinstyle
        self._params.ydata[name]['solid_joinstyle'] = solid_joinstyle
        self._params.ydata[name]['antialiased'] = antialiased
        return self
        
    def render(self):
        return self._execute().show()
    
    def _execute(self):
        with plt.style.context(self._params.theme):
            fig = plt.figure(figsize=self._params.figsize, 
                             dpi=self._params.dpi, 
                             facecolor=self._params.facecolor,
                             edgecolor=self._params.edgecolor, 
                             frameon=self._params.frameon, 
                             clear=self._params.clear)
            ax = fig.add_subplot()
        for i,j in self._params.ydata.items():
            ax_plot = ax.errorbar(
                j['xdata'], 
                j['ydata'], 
                yerr=j['yerr'], 
                xerr=j['xerr'], 
                ecolor=j['ecolor'],
                elinewidth=j['elinewidth'], 
                capsize=j['capsize'],
                barsabove=j['barsabove'],
                lolims=j['lolims'],
                uplims=j['uplims'],
                xlolims=j['xlolims'],
                xuplims=j['xuplims'],
                errorevery=j['errorevery'],
                capthick=j['capthick'],
                linestyle=j['linestyle'], 
                color=j['linecolor'], 
                linewidth=j['linewidth'], 
                marker=j['marker'], 
                markersize=j['markersize'], 
                markeredgewidth=j['markeredgewidth'], 
                markeredgecolor=j['markeredgecolor'], 
                markerfacecolor=j['markerfacecolor'], 
                markevery=j['markevery'],
                fillstyle=j['fillstyle'], 
                drawstyle=j['drawstyle'], 
                antialiased=j['antialiased'],
                dash_capstyle=j['dash_capstyle'], 
                solid_capstyle=j['solid_capstyle'], 
                dash_joinstyle=j['dash_joinstyle'], 
                solid_joinstyle=j['solid_joinstyle'])
            ax_plot.set_label(i)
        if self._params.xlabel is not None:
            ax.set_xlabel(self._params.xlabel, labelpad=self._params.xlabelpad, loc=self._params.xloc)
        if self._params.ylabel is not None:
            ax.set_ylabel(self._params.ylabel, labelpad=self._params.ylabelpad, loc=self._params.yloc)
        if self._params.title is not None:
            ax.set_title(self._params.title, fontdict=None, loc=self._params.titleloc, 
                         pad=self._params.titlepad, y=self._params.titley)
        if self._params.axis is not None:
            ax.axis(self._params.axis)
        if self._params.legendloc is not None:
            ax.legend(loc=self._params.legendloc)        
        return fig