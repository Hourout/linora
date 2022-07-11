import numpy as np
import matplotlib.pyplot as plt

from linora.chart._base import Coordinate

__all__ = ['Fillline']


class Fillline(Coordinate):
    def __init__(self, *args, **kwargs):
        super(Fillline, self).__init__()
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
        ydata2=0,
        where=None,
        interpolate=False,
        step=None,
                 linestyle=None, fillcolor=None, linewidth=None,alpha=None
                ):
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
            fillcolor: line color, eg. 'blue' or '0.75' or 'g' or '#FFDD44' or (1.0,0.2,0.3) or 'chartreuse'.
            linewidth: line width.
            alpha:
        """
        self._params.ydata[name]['xdata'] = xdata
        self._params.ydata[name]['ydata'] = ydata
        self._params.ydata[name]['ydata2'] = ydata2
        self._params.ydata[name]['where'] = where
        self._params.ydata[name]['interpolate'] = interpolate
        self._params.ydata[name]['step'] = step
        
        self._params.ydata[name]['linestyle'] = linestyle
        self._params.ydata[name]['linewidth'] = linewidth
        if fillcolor is None:
            fillcolor = tuple([round(np.random.uniform(0, 1),1) for _ in range(3)])
        elif isinstance(fillcolor, dict):
            fillcolor = fillcolor['mode']
        self._params.ydata[name]['fillcolor'] = fillcolor
        self._params.ydata[name]['alpha'] = alpha
        return self
    
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
            ax_plot = ax.fill_between(
                j['xdata'], 
                j['ydata'], 
                y2=j['ydata2'],
                where=j['where'],
                interpolate=j['interpolate'],
                step=j['step'],
#                 linestyle=j['linestyle'], 
                color=j['fillcolor'], 
                linewidth=j['linewidth'],
                alpha=j['alpha']
            )
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
#         if self._params.legendloc is not None:
#             ax.legend(loc=self._params.legendloc)  
        return fig