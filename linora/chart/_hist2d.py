import numpy as np
import matplotlib.pyplot as plt

from linora.chart._base import Coordinate

__all__ = ['Hist2d']


class Hist2d(Coordinate):
    def __init__(self, *args, **kwargs):
        super(Hist2d, self).__init__()
        if len(args)!=0:
            if isinstance(args[0], dict):
                for i,j in args[0].items():
                    setattr(self._params, i, j)
        if kwargs:
            for i,j in kwargs.items():
                setattr(self._params, i, j)

    def add_data(self, name, xdata, ydata, **kwargs):
        """A scatter plot of *y* vs. *x* with varying marker size and/or color.
        
        Args:
            name: data name.
            xdata: x-axis data.
            
        """
        self._params.ydata[name]['x'] = xdata
        self._params.ydata[name]['y'] = ydata
        self._params.ydata[name].update(kwargs)
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
            ax_plot = ax.hist2d(**j)
#             print(ax_plot)
#             ax_plot[2].set_label(i)
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