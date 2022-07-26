import numpy as np
import matplotlib.pyplot as plt

from linora.chart._base import Coordinate
from linora.chart._bar import Bar
from linora.chart._boxplot import Boxplot
from linora.chart._circle import Circle
from linora.chart._ellipse import Ellipse
from linora.chart._errorbar import Errorbar
from linora.chart._fillline import Fillline
from linora.chart._hist import Hist
from linora.chart._hist2d import Hist2d
from linora.chart._line import Line
from linora.chart._polygon import Polygon
from linora.chart._rectangle import Rectangle
from linora.chart._regularpolygon import RegularPolygon
from linora.chart._scatter import Scatter

__all__ = ['Plot']


class Plot(Coordinate, Bar, Boxplot, Circle, Ellipse, Errorbar, Fillline, Hist, Hist2d, Line, 
           Polygon, Rectangle, RegularPolygon, Scatter):
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
            j['plotfunc'](ax, i, j)
            
        if self._params.label['xlabel']['xlabel'] is not None:
            ax.set_xlabel(**self._params.label['xlabel'])
        if self._params.label['ylabel']['ylabel'] is not None:
            ax.set_ylabel(**self._params.label['ylabel'])
        if self._params.title['label'] is not None:
            ax.set_title(**self._params.title)
        
        if self._params.axis['axis'] is not None:
            ax.axis(self._params.axis['axis'])
        if self._params.axis['xlabel'] is not None:
            if len(self._params.axis['xlabel'])==0:
                ax.set_xticks(self._params.axis['xlabel'])
            elif isinstance(self._params.axis['xlabel'][0], (list, tuple, np.ndarray)):
                ax.set_xticks(self._params.axis['xlabel'][0])
                ax.set_xticklabels(self._params.axis['xlabel'][1])
            else:
                ax.set_xticks(self._params.axis['xlabel'])
        if self._params.axis['ylabel'] is not None:
            if len(self._params.axis['ylabel'])==0:
                ax.set_yticks(self._params.axis['ylabel'])
            elif isinstance(self._params.axis['ylabel'][0], (list, tuple, np.ndarray)):
                ax.set_yticks(self._params.axis['ylabel'][0])
                ax.set_yticklabels(self._params.axis['ylabel'][1])
            else:
                ax.set_yticks(self._params.axis['ylabel'])
        ax.tick_params(axis='x', **self._params.axis['xtick'])
        ax.tick_params(axis='y', **self._params.axis['ytick'])
        if self._params.axis['xinvert']:
            ax.invert_xaxis()
        if self._params.axis['yinvert']:
            ax.invert_yaxis()
        if self._params.axis['xtickposition'] is not None:
            ax.xaxis.set_ticks_position(self._params.axis['xtickposition'])
        if self._params.axis['ytickposition'] is not None:
            ax.yaxis.set_ticks_position(self._params.axis['ytickposition'])
        
        if self._params.legend['loc'] is not None:
            ax.legend(**self._params.legend)
        else:
            t = ['ellipse', 'regularpolygon', 'rectangle', 'circle', 'polygon', 'boxplot']
            if len([1 for i,j in self._params.ydata.items() if j['plotmode'] not in t])>1:
                ax.legend(loc='best')
                
        if len(self._params.spine['color'])>0:
            for i,j in self._params.spine['color'].items():
                ax.spines[i].set_color(j)
        if len(self._params.spine['width'])>0:
            for i,j in self._params.spine['width'].items():
                ax.spines[i].set_linewidth(j)
        if len(self._params.spine['style'])>0:
            for i,j in self._params.spine['style'].items():
                ax.spines[i].set_linestyle(j)
        if len(self._params.spine['position'])>0:
            for i,j in self._params.spine['position'].items():
                ax.spines[i].set_position(j)
        if len(self._params.spine['show'])>0:
            for i,j in self._params.spine['show'].items():
                ax.spines[i].set_visible(j)
        return ax