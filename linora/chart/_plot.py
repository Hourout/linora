import matplotlib.pyplot as plt

from linora.chart._base import Coordinate
from linora.chart._bar import Bar
from linora.chart._errorbar import Errorbar
from linora.chart._line import Line
from linora.chart._fillline import Fillline
from linora.chart._hist import Hist
from linora.chart._hist2d import Hist2d
from linora.chart._scatter import Scatter


__all__ = ['Plot']


class Plot(Coordinate, Bar, Errorbar, Fillline, Hist, Hist2d, Line, Scatter):
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
        fig = plt.figure(**self._params.figure)
        with plt.style.context(self._params.theme):
            ax = fig.add_subplot()
        ax = self._execute_ax(ax)
        return fig
            
    def _execute_ax(self, ax):
        for i,j in self._params.ydata.items():
            if j['plotmode']=='bar':
                ax_plot = ax.bar(j['xdata'], j['ydata'], **j['kwargs'])
                ax_plot.set_label(i)
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
            elif j['plotmode']=='line':
                ax_plot = ax.plot(j['xdata'], j['ydata'], **j['kwargs'])
            elif j['plotmode']=='scatter':
                ax_plot = ax.scatter(j['xdata'], j['ydata'], **j['kwargs'])
                ax_plot.set_label(i)
#                 if not self._params.set_label:
#                     if len(self._params.colorbar)>0:
#                         fig.colorbar(ax_plot)
#                         self._params.colorbar.remove(list(self._params.colorbar)[0])
        if self._params.xlabel['xlabel'] is not None:
            ax.set_xlabel(**self._params.xlabel)
        if self._params.ylabel['ylabel'] is not None:
            ax.set_ylabel(**self._params.ylabel)
        if self._params.title['label'] is not None:
            ax.set_title(**self._params.title)
        if self._params.axis['axis'] is not None:
            ax.axis(self._params.axis['axis'])
        if self._params.axis['xtickshow']:
            ax.tick_params(axis='x', **self._params.axis['xtick'])
        if self._params.axis['ytickshow']:
            ax.tick_params(axis='y', **self._params.axis['ytick'])
        if self._params.axis['xinvert']:
            ax.invert_xaxis()
        if self._params.axis['yinvert']:
            ax.invert_yaxis()
        if self._params.legend['loc'] is not None:
            ax.legend(**self._params.legend)  
        return ax