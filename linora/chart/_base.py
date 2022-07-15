from collections import defaultdict

from linora.utils._config import Config


class Coordinate():
    def __init__(self):
        self._params = Config()
        self._params.ydata = defaultdict(defaultdict)
        
        self._params.theme = 'seaborn-whitegrid'
        
        self._grid.figure = {'figsize':(10, 6)}
        
        self._params.axis = None
        
        self._params.xlabel = {'xlabel':None}
        self._params.ylabel = {'ylabel':None}
        
        self._params.legend = {'loc':None}
        
        self._params.title = {'label':None}
        
        self._params.set_label = True
        self._params.colorbar = set()
        
    def render(self):
        return self._execute().show()
    
    def set_axis(self, axis=None, xmin=None, xmax=None, ymin=None, ymax=None):
        """Convenience method to get or set some axis properties.
        
        Args:
            axis: bool or str, If a bool, turns axis lines and labels on or off. 
                  If a string, possible values are:
                    ======== ==========================================================
                    Value    Description
                    ======== ==========================================================
                    'on'     Turn on axis lines and labels. Same as ``True``.
                    'off'    Turn off axis lines and labels. Same as ``False``.
                    'equal'  Set equal scaling (i.e., make circles circular) by
                             changing axis limits. This is the same as
                             ``ax.set_aspect('equal', adjustable='datalim')``.
                             Explicit data limits may not be respected in this case.
                    'scaled' Set equal scaling (i.e., make circles circular) by
                             changing dimensions of the plot box. This is the same as
                             ``ax.set_aspect('equal', adjustable='box', anchor='C')``.
                             Additionally, further autoscaling will be disabled.
                    'tight'  Set limits just large enough to show all data, then
                             disable further autoscaling.
                    'auto'   Automatic scaling (fill plot box with data).
                    'image'  'scaled' with axis limits equal to data limits.
                    'square' Square plot; similar to 'scaled', but initially forcing
                             ``xmax-xmin == ymax-ymin``.
                    ======== ==========================================================
                    if axis is set, other parameters are invalid.
            xmin: float, The left xlim in data coordinates.
            xmax: float, The right xlim in data coordinates.
            ymin: float, The bottom ylim in data coordinates.
            ymax: float, The top ylim in data coordinates.
        """
        if axis is not None:
            self._params.axis = axis
        elif xmin is not None or xmax is not None or ymin is not None or ymax is not None:
            self._params.axis = [xmin, xmax, ymin, ymax]
        else:
            raise ValueError("params is not None.")
        return self
        
    def set_label(self, xlabel=None, ylabel=None, xloc=None, yloc=None, xlabelpad=None, ylabelpad=None):
        """Set the label for the x-axis and y-axis.
        
        Args:
            xlabel : str, The label text.
            ylabel : str, The label text.
            xloc : {'left', 'center', 'right'}, The label position.
            yloc : {'bottom', 'center', 'top'}, The label position.
            xlabelpad : float, Spacing in points from the axes bounding box including ticks and tick labels.
            ylabelpad : float, Spacing in points from the axes bounding box including ticks and tick labels.
        """
        if xlabel is not None:
            self._params.xlabel.update({'xlabel':xlabel, 'loc':xloc, 'labelpad':xlabelpad})
        if ylabel is not None:
            self._params.ylabel.update({'ylabel':ylabel, 'loc':yloc, 'labelpad':ylabelpad})
        return self
    
    def set_legend(self, loc='best', **kwargs):
        """Place a legend on the Axes.
        
        Args:
            loc: str or pair of floats, The location of the legend.
                The strings ``'upper left', 'upper right', 'lower left', 'lower right'``
                place the legend at the corresponding corner of the axes/figure.

                The strings ``'upper center', 'lower center', 'center left', 'center right'``
                place the legend at the center of the corresponding edge of the axes/figure.
                The string ``'center'`` places the legend at the center of the axes/figure.

                The string ``'best'`` places the legend at the location, among the nine
                locations defined so far, with the minimum overlap with other drawn artists.  
                This option can be quite slow for plots with large amounts of data; 
                your plotting speed may benefit from providing a specific location.

                The location can also be a 2-tuple giving the coordinates of the lower-left
                corner of the legend in axes coordinates (in which case *bbox_to_anchor*
                will be ignored).
            fontsize: int or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
                      The font size of the legend. If the value is numeric the size will be the absolute 
                      font size in points. String values are relative to the current default font size.

            

numpoints: int, The number of marker points in the legend when creating a legend.
markerscale: float, The relative size of legend markers compared with the originally drawn ones.
markerfirst: bool, default: True
            If *True*, legend marker is placed to the left of the legend label.
            If *False*, legend marker is placed to the right of the legend label.

frameon : bool, Whether the legend should be drawn on a patch (frame).

fancybox : bool, Whether round edges should be enabled around the `~.FancyBboxPatch` which
    makes up the legend's background.

shadow : bool, Whether to draw a shadow behind the legend.

framealpha : float, The alpha transparency of the legend's background.
             If *shadow* is activated and *framealpha* is ``None``, the default value is ignored.

facecolor : "inherit" or color, default: :rc:`legend.facecolor`
    The legend's background color.
    If ``"inherit"``, use :rc:`axes.facecolor`.

edgecolor : "inherit" or color, default: :rc:`legend.edgecolor`
    The legend's background patch edge color.
    If ``"inherit"``, use take :rc:`axes.edgecolor`.

mode : {"expand", None}
    If *mode* is set to ``"expand"`` the legend will be horizontally
    expanded to fill the axes area (or *bbox_to_anchor* if defines
    the legend's size).

bbox_transform : None or `matplotlib.transforms.Transform`
    The transform for the bounding box (*bbox_to_anchor*). For a value
    of ``None`` (default) the Axes'
    :data:`~matplotlib.axes.Axes.transAxes` transform will be used.

title : str or None, The legend's title. Default is no title (``None``).
title_fontsize: int or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
                The font size of the legend's title.
borderpad : float, The fractional whitespace inside the legend border, in font-size units.
labelspacing : float, The vertical space between the legend entries, in font-size units.
handlelength : float, The length of the legend handles, in font-size units.
handletextpad : float, The pad between the legend handle and text, in font-size units.
borderaxespad : float, The pad between the axes and legend border, in font-size units.
columnspacing : float, The spacing between columns, in font-size units.

handler_map : dict or None
    The custom dictionary mapping instances or types to a legend
    handler. This *handler_map* updates the default handler map
    found at `matplotlib.legend.Legend.get_legend_handler_map`.

                      
        """
#         'legend.borderaxespad': 0.5,
# 'legend.borderpad': 0.4,
# 'legend.columnspacing': 2.0,
# 'legend.edgecolor': '0.8',
# 'legend.facecolor': 'inherit',
# 'legend.fancybox': True,
# 'legend.fontsize': 'medium',
# 'legend.framealpha': 0.8,
# 'legend.frameon': True,
# 'legend.handleheight': 0.7,
# 'legend.handlelength': 2.0,
# 'legend.handletextpad': 0.8,
# 'legend.labelspacing': 0.5,
# 'legend.loc': 'best',
# 'legend.markerscale': 1.0,
# 'legend.numpoints': 1,
# 'legend.scatterpoints': 1,
# 'legend.shadow': False,
# 'legend.title_fontsize': None,
        kwargs['loc'] = legendloc
        self._params.legend.update(kwargs)
        return self
        
    def set_title(self, title, titleloc='center', titlesize='large', titlecolor='auto', 
                  titlepad=6., titley=None, titleweight='normal'):
        """Set a title for the Axes.
        
        Args:
            title: str, Text to use for the title.
            titleloc : {'center', 'left', 'right'}, Which title to set.
            titlepad : float, The offset of the title from the top of the Axes, in points.
            titley : float, Vertical Axes loation for the title (1.0 is the top).
                     If None (the default), titley is determined automatically to avoid decorators on the Axes.
        """
        kwargs = {'label':title, 'loc':titleloc, 'pad':titlepad, 'y':titley,
                  'fontdict':{'fontsize':titlesize, 'fontweight':titleweight, 'color':titlecolor,
                              'verticalalignment': 'baseline', 'horizontalalignment': titleloc}}
        self._params.title = kwargs
        return self
        
    def set_theme(self, theme):
        """
        Args:
            theme: str, figure theme.
        """
        self._params.theme = theme
        return self
    
    def set_figure(self, width=10, height=6, dpi=None, facecolor=None, edgecolor=None, frameon=True, clear=False):
        """Add figure config.
        
        Args:
            width: float, figure size width in inches.
            height: float, figure size height in inches.
            dpi: float, The resolution of the figure in dots-per-inch.
            facecolor: color, The background color.
            edgecolor: color, The border color.
            frameon: bool, default: True, If False, suppress drawing the figure frame.
            clear: bool, default: False, If True and the figure already exists, then it is cleared.
        """
        kwargs = {'figsize':(width, height), 'dpi':dpi, 'facecolor':facecolor, 
                  'edgecolor':edgecolor, 'frameon':frameon, 'clear':clear}
        self._params.figure.update(kwargs)
        return self
    

    
    