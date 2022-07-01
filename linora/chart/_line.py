from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from linora.utils._config import Config
from linora.chart._base import Coordinate

__all__ = ['Line']


class Line(Coordinate):
    def __init__(self, *args, **kwargs):
        super(Line, self).__init__()
        if len(args)!=0:
            if isinstance(args[0], dict):
                for i,j in args[0].items():
                    setattr(self._params, i, j)
        if kwargs:
            for i,j in kwargs.items():
                setattr(self._params, i, j)
    
    def add_data(self, name, xdata, ydata, linestyle=None, linecolor=None, linewidth=None,
                 marker=None, markersize=None, markeredgewidth=None, markeredgecolor=None, 
                 markerfacecolor=None, markerfacecoloralt='none', markevery=None,
                 fillstyle=None, antialiased=None, drawstyle=None, 
                 dash_capstyle=None, solid_capstyle=None, dash_joinstyle=None, solid_joinstyle=None, 
                ):
        """Plot y versus x as lines and/or markers.
        
        Args:
            name: data name.
            xdata: x-axis data.
            ydata: y-axis data.
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
            markerfacecoloralt: color
            markersize: float
            markevery: None or int or (int, int) or slice or list[int] or float or (float, float) or list[bool]
            solid_capstyle: {'butt', 'projecting', 'round'}
            solid_joinstyle: {'miter', 'round', 'bevel'}
            antialiased: Set whether to use antialiased rendering.
        """
        self._params.ydata[name]['xdata'] = xdata
        self._params.ydata[name]['ydata'] = ydata
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
        self._params.ydata[name]['markerfacecoloralt'] = markerfacecoloralt
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
            ax.plot(j['xdata'], j['ydata'], label=i, 
                    linestyle=j['linestyle'], color=j['linecolor'], linewidth=j['linewidth'], 
                    marker=j['marker'], markersize=j['markersize'], markeredgewidth=j['markeredgewidth'], 
                    markeredgecolor=j['markeredgecolor'], markerfacecolor=j['markerfacecolor'], 
                    markerfacecoloralt=j['markerfacecoloralt'], markevery=j['markevery'],
                    fillstyle=j['fillstyle'], drawstyle=j['drawstyle'], antialiased=j['antialiased'],
                    dash_capstyle=j['dash_capstyle'], solid_capstyle=j['solid_capstyle'], 
                    dash_joinstyle=j['dash_joinstyle'], solid_joinstyle=j['solid_joinstyle'])
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