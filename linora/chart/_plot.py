import matplotlib.pyplot as plt

from linora.chart._base import Coordinate
from linora.chart._line import Line
from linora.chart._scatter import Scatter
from linora.chart._errorbar import Errorbar
from linora.chart._fillline import Fillline
from linora.chart._hist import Hist
from linora.chart._hist2d import Hist2d


__all__ = ['Plot']


class Plot(Coordinate, Line, Scatter, Errorbar, Fillline, Hist, Hist2d):
    def __init__(self, *args, **kwargs):
        super(Plot, self).__init__()
        if len(args)!=0:
            if isinstance(args[0], dict):
                for i,j in args[0].items():
                    setattr(self._params, i, j)
        if kwargs:
            for i,j in kwargs.items():
                setattr(self._params, i, j)
                
    def _execute(self):
        fig = plt.figure(figsize=self._params.figsize, 
                         dpi=self._params.dpi, 
                         facecolor=self._params.facecolor,
                         edgecolor=self._params.edgecolor, 
                         frameon=self._params.frameon, 
                         clear=self._params.clear)
        with plt.style.context(self._params.theme):
            ax = fig.add_subplot()
        ax = self._execute_ax(ax)
        return fig
            
    def _execute_ax(self, ax):
        for i,j in self._params.ydata.items():
            if j['plotmode']=='line':
                ax_plot = ax.plot(j['xdata'], j['ydata'], **j['kwargs'])
            elif j['plotmode']=='scatter':
                ax_plot = ax.scatter(j['xdata'], j['ydata'], **j['kwargs'])
                ax_plot.set_label(i)
#                 if not self._params.set_label:
#                     if len(self._params.colorbar)>0:
#                         fig.colorbar(ax_plot)
#                         self._params.colorbar.remove(list(self._params.colorbar)[0])
            elif j['plotmode']=='errorbar':
                ax_plot = ax.errorbar(j['xdata'], j['ydata'], **j['kwargs'])
                ax_plot.set_label(i)
            elif j['plotmode']=='fillline':
                ax_plot = ax.fill_between(j['xdata'], j['ydata'], y2=j['ydata2'], **j['kwargs'])
                ax_plot.set_label(i)
            elif j['plotmode']=='hist':
                ax_plot = ax.hist(j['xdata'], **j['kwargs'])
            elif j['plotmode']=='hist2d':
                ax_plot = ax.hist(j['xdata'], j['ydata'], **j['kwargs'])
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
        return ax