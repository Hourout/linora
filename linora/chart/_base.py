from collections import defaultdict

from linora.utils._config import Config


class Coordinate():
    def __init__(self):
        self._params = Config()
        self._params.ydata = defaultdict(defaultdict)
        
        self._params.theme = 'ggplot'
        
        self._params.figure = {'figsize':(10, 6)}
        
        self._params.axis = {'axis':None, 'xinvert':False, 'yinvert':False, 'xtick':{}, 'ytick':{},
                             'xlabel':None, 'ylabel':None, 'xtickposition':None, 'ytickposition':None}
        
        self._params.label = {'xlabel':{'xlabel':None}, 'ylabel':{'ylabel':None}}
        
        self._params.legend = {'loc':None}
        self._params.spine = {'show':{}, 'color':{}, 'width':{}, 'style':{}, 'position':{}}
        self._params.title = {'label':None}
        
        self._params.set_label = True
        self._params.colorbar = set()
        
    def render(self, image_path=None, if_show=True):
        if if_show:
            return self._execute().show()
    
    def set_axis(self, axis=None, xmin=None, xmax=None, ymin=None, ymax=None,
                 xlabel=None, ylabel=None,
                 invert=False, xinvert=False, yinvert=False, 
                 tickshow=True, xtickshow=True, ytickshow=True, 
                 tickwhich=None, xtickwhich=None, ytickwhich=None, 
                 tickcolor=None, xtickcolor=None, ytickcolor=None, 
                 tickheight=None, xtickheight=None, ytickheight=None, 
                 tickwidth=None, xtickwidth=None, ytickwidth=None, 
                 tickpad=None, xtickpad=None, ytickpad=None, 
                 tickloc=None, xtickloc=None, ytickloc=None,
                 xtickposition=None, ytickposition=None,
                 labelsize=None, xlabelsize=None, ylabelsize=None, 
                 labelcolor=None, xlabelcolor=None, ylabelcolor=None, 
                 labelrotate=None, xlabelrotate=None, ylabelrotate=None, 
                 gridcolor=None, xgridcolor=None, ygridcolor=None, 
                 gridalpha=None, xgridalpha=None, ygridalpha=None, 
                 gridwidth=None, xgridwidth=None, ygridwidth=None, 
                 gridstyle=None, xgridstyle=None, ygridstyle=None, 
                 tickbottom=None, ticktop=None, tickleft=None, tickright=None,
                 labelbottom=None, labeltop=None, labelleft=None, labelright=None
                ):
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
            xlabel: list or [list, list] of x-axis tick label.
            ylabel: list or [list, list] of y-axis tick label.
            invert: invert xy-axis.
            xinvert: invert x-axis.
            yinvert: invert y-axis.
            tickshow: show xy-axis tick.
            xtickshow: show x-axis tick.
            ytickshow: show y-axis tick.
            tickwhich: {'major', 'minor', 'both'}, default: 'major', 
                The group of xy-axis ticks to which the parameters are applied.
            xtickwhich: {'major', 'minor', 'both'}, default: 'major', 
                The group of x-axis ticks to which the parameters are applied.
            ytickwhich: {'major', 'minor', 'both'}, default: 'major', 
                The group of y-axis ticks to which the parameters are applied.
            tickcolor: xy-axis tick color.
            xtickcolor: x-axis tick color.
            ytickcolor: y-axis tick color.
            tickheight: xy-axis tick height.
            xtickheight: x-axis tick height.
            ytickheight: y-axis tick height.
            tickwidth: xy-axis tick width.
            xtickwidth: x-axis tick width.
            ytickwidth: y-axis tick width.
            tickpad: xy-axis distance in points between tick and label.
            xtickpad: x-axis distance in points between tick and label.
            ytickpad: y-axis distance in points between tick and label.
            tickloc: {'in', 'out', 'inout'}, xy-axis puts ticks position.
            xtickloc: {'in', 'out', 'inout'}, x-axis puts ticks position.
            ytickloc: {'in', 'out', 'inout'}, y-axis puts ticks position.
            xtickposition: {'top', 'bottom', 'both', 'default', 'none'}, ticks position.
            ytickposition: {'left', 'right', 'both', 'default', 'none'}, ticks position.
            labelsize: xy-axis label font size.
            xlabelsize: x-axis label font size.
            ylabelsize: y-axis label font size.
            labelcolor: xy-axis label font color.
            xlabelcolor: x-axis label font color.
            ylabelcolor: y-axis label font color.
            labelrotate: xy-axis label font rotate.
            xlabelrotate: x-axis label font rotate.
            ylabelrotate: y-axis label font rotate.
            gridcolor: xy-axis grid line color.
            xgridcolor: x-axis grid line color.
            ygridcolor: y-axis grid line color.
            gridalpha: xy-axis grid line alpha.
            xgridalpha: x-axis grid line alpha.
            ygridalpha: y-axis grid line alpha.
            gridwidth: xy-axis grid line width.
            xgridwidth: x-axis grid line width.
            ygridwidth: y-axis grid line width.
            gridstyle: xy-axis grid line style.
            xgridstyle: x-axis grid line style.
            ygridstyle: y-axis grid line style.
            tickbottom: Whether to draw bottom ticks.
            ticktop: Whether to draw top ticks.
            tickleft: Whether to draw the left ticks.
            tickright: Whether to draw the right ticks.
            labelbottom: Whether to draw bottom tick labels.
            labeltop: Whether to draw top tick labels.
            labelleft: Whether to draw left tick labels.
            labelright: Whether to draw right tick labels.
        """
        if axis is not None:
            self._params.axis['axis'] = axis
        elif xmin is not None or xmax is not None or ymin is not None or ymax is not None:
            self._params.axis['axis'] = [xmin, xmax, ymin, ymax]
        self._params.axis['xinvert'] = invert or xinvert
        self._params.axis['yinvert'] = invert or yinvert
        if xlabel is not None:
            self._params.axis['xlabel'] = xlabel
        if ylabel is not None:
            self._params.axis['ylabel'] = ylabel
        if xtickposition is not None:
            self._params.axis['xtickposition'] = xtickposition
        if ytickposition is not None:
            self._params.axis['ytickposition'] = ytickposition
        
        self._set_axis(xtickwhich, ytickwhich, tickwhich, 'which')
        self._set_axis(xtickcolor, ytickcolor, tickcolor, 'color')
        self._set_axis(xtickheight, ytickheight, tickheight, 'length')
        self._set_axis(xtickwidth, ytickwidth, tickwidth, 'width')
        self._set_axis(xtickloc, ytickloc, tickloc, 'direction')
        self._set_axis(xtickpad, ytickpad, tickpad, 'pad')
        self._set_axis(xlabelsize, ylabelsize, labelsize, 'labelsize')
        self._set_axis(xlabelcolor, ylabelcolor, labelcolor, 'labelcolor')
        self._set_axis(xlabelrotate, ylabelrotate, labelrotate, 'labelrotation')
        self._set_axis(xgridcolor, ygridcolor, gridcolor, 'grid_color')
        self._set_axis(xgridalpha, ygridalpha, gridalpha, 'grid_alpha')
        self._set_axis(xgridwidth, ygridwidth, gridwidth, 'grid_linewidth')
        self._set_axis(xgridstyle, ygridstyle, gridstyle, 'grid_linestyle')
            
        if tickbottom is not None:
            self._params.axis['xtick']['bottom'] = tickbottom
        if ticktop is not None:
            self._params.axis['xtick']['top'] = ticktop
        if tickleft is not None:
            self._params.axis['ytick']['left'] = tickleft
        if tickright is not None:
            self._params.axis['ytick']['right'] = tickright
        if labelbottom is not None:
            self._params.axis['xtick']['labelbottom'] = labelbottom
        if labeltop is not None:
            self._params.axis['xtick']['labeltop'] = labeltop
        if labelleft is not None:
            self._params.axis['ytick']['labelleft'] = labelleft
        if labelright is not None:
            self._params.axis['ytick']['labelright'] = labelright

        if not (tickshow and xtickshow):
            self._params.axis['xtick'] = {'length':0, 'labelsize':0, 'which':'both'}
        if not (tickshow and ytickshow):
            self._params.axis['ytick'] = {'length':0, 'labelsize':0, 'which':'both'}
        return self
    
    def _set_axis(self, x, y, xy, s):
        if x is not None:
            self._params.axis['xtick'][s] = x['mode'] if isinstance(x, dict) else x
        if y is not None:
            self._params.axis['ytick'][s] = y['mode'] if isinstance(y, dict) else y
        if xy is not None:
            self._params.axis['xtick'][s] = xy['mode'] if isinstance(xy, dict) else xy
            self._params.axis['ytick'][s] = xy['mode'] if isinstance(xy, dict) else xy
        
    def set_label(self, xlabel=None, ylabel=None, 
                  loc=None, xloc=None, yloc=None, 
                  pad=None, xpad=None, ypad=None,
                  fontsize=None, xfontsize=None, yfontsize=None, 
                  fontcolor=None, xfontcolor=None, yfontcolor=None, 
                  fontfamily=None, xfontfamily=None, yfontfamily=None, 
                  fontstyle=None, xfontstyle=None, yfontstyle=None, 
                 ):
        """Set the label for the x-axis and y-axis.
        
        Args:
            xlabel: str, The label text.
            ylabel: str, The label text.
            loc: {'left', 'center', 'right'}, The xy label position.
            xloc: {'left', 'center', 'right'}, The xlabel position.
            yloc: {'bottom', 'center', 'top'}, The ylabel position.
            pad: float, Spacing in points from the axes bounding box including ticks and tick xy labels.
            xpad : float, Spacing in points from the axes bounding box including ticks and tick x labels.
            ypad : float, Spacing in points from the axes bounding box including ticks and tick y labels.
            fontsize: xy label font size.
            xfontsize: x label font size.
            yfontsize: y label font size.
            fontcolor: xy label font color.
            xfontcolor: x label font color.
            yfontcolor: y label font color.
            fontfamily:
            xfontfamily:
            yfontfamily:
            fontstyle: xy label font style.
            xfontstyle: x label font style.
            yfontstyle: y label font style.
        """
        self._set_label(xlabel, y=None, xy=None, xkey='xlabel')
        self._set_label(x=None, y=ylabel, xy=None, ykey='ylabel')
        self._set_label(x=xloc, y=yloc, xy=loc, xkey='loc', ykey='loc')
        self._set_label(x=xpad, y=ypad, xy=pad, xkey='labelpad', ykey='labelpad')
        self._set_label(x=xfontsize, y=yfontsize, xy=fontsize, xkey='fontsize', ykey='fontsize')
        self._set_label(x=xfontcolor, y=yfontcolor, xy=fontcolor, xkey='color', ykey='color')
        self._set_label(x=xfontfamily, y=yfontfamily, xy=fontfamily, xkey='fontfamily', ykey='fontfamily')
        self._set_label(x=xfontstyle, y=yfontstyle, xy=fontstyle, xkey='fontstyle', ykey='fontstyle')
        return self
    
    def _set_label(self, x, y, xy, xkey=None, ykey=None):
        if x is not None:
            self._params.label['xlabel'][xkey] = x['mode'] if isinstance(x, dict) else x
        if y is not None:
            self._params.label['ylabel'][ykey] = y['mode'] if isinstance(y, dict) else y
        if xy is not None:
            self._params.label['xlabel'][xkey] = xy['mode'] if isinstance(xy, dict) else xy
            self._params.label['ylabel'][ykey] = xy['mode'] if isinstance(xy, dict) else xy
    
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
        
    def set_spine(self, spineshow=None, leftshow=None, rightshow=None, topshow=None, bottomshow=None,
                  spinecolor=None, leftcolor=None, rightcolor=None, topcolor=None, bottomcolor=None,
                  spinewidth=None, leftwidth=None, rightwidth=None, topwidth=None, bottomwidth=None,
                  spinestyle=None, leftstyle=None, rightstyle=None, topstyle=None, bottomstyle=None,
                  leftloc=None, rightloc=None, toploc=None, bottomloc=None,
                 ):
        """Set a wireframes for the Axes.
        spineshow: bool, whether to show all wireframes.
        leftshow: bool, whether to show left wireframes.
        rightshow: bool, whether to show right wireframes.
        topshow: bool, whether to show top wireframes.
        bottomshow: bool, whether to show bottom wireframes.
        spinecolor: all wireframes color.
        leftcolor: left wireframes color.
        rightcolor: right wireframes color.
        topcolor: top wireframes color.
        bottomcolor: bottom wireframes color.
        spinewidth: all wireframes width.
        leftwidth: left wireframes width.
        rightwidth: right wireframes width.
        topwidth: top wireframes width.
        bottomwidth: bottom wireframes width.
        spinestyle: all wireframes style.
        leftstyle: left wireframes style.
        rightstyle: right wireframes style.
        topstyle: top wireframes style.
        bottomstyle: bottom wireframes style.
        leftloc: 2 tuple of (position type, amount), left wireframes position.
            The position types are :
            * 'outward': place the spine out from the data area by the specified number of points.
            * 'axes': place the spine at the specified Axes coordinate (0 to 1).
            * 'data': place the spine at the specified data coordinate.
        rightloc: 2 tuple of (position type, amount), right wireframes position.
        toploc: 2 tuple of (position type, amount), top wireframes position.
        bottomloc: 2 tuple of (position type, amount), bottom wireframes position.
        """
        self._set_spine(spineshow, leftshow, rightshow, topshow, bottomshow, 'show')
        self._set_spine(spinecolor, leftcolor, rightcolor, topcolor, bottomcolor, 'color')
        self._set_spine(spinewidth, leftwidth, rightwidth, topwidth, bottomwidth, 'width')
        self._set_spine(spinestyle, leftstyle, rightstyle, topstyle, bottomstyle, 'style')
        self._set_spine(None, leftloc, rightloc, toploc, bottomloc, 'position')
        return self
    
    def _set_spine(self, spine, left, right, top, bottom, key):
        if left is not None:
            self._params.spine[key]['left'] = left['mode'] if isinstance(left, dict) else left
        if right is not None:
            self._params.spine[key]['right'] = right['mode'] if isinstance(right, dict) else right
        if top is not None:
            self._params.spine[key]['top'] = top['mode'] if isinstance(top, dict) else top
        if bottom is not None:
            self._params.spine[key]['bottom'] = bottom['mode'] if isinstance(bottom, dict) else bottom
        if spine is not None:
            self._params.spine[key]['left'] = spine['mode'] if isinstance(spine, dict) else spine
            self._params.spine[key]['right'] = spine['mode'] if isinstance(spine, dict) else spine
            self._params.spine[key]['top'] = spine['mode'] if isinstance(spine, dict) else spine
            self._params.spine[key]['bottom'] = spine['mode'] if isinstance(spine, dict) else spine
        
    
    def set_title(self, title, loc='center', fontsize='large', fontcolor='#000000', 
                  titlepad=6., titley=None, fontweight='normal'):
        """Set a title for the Axes.
        
        Args:
            title: str, Text to use for the title.
            loc : {'center', 'left', 'right'}, Which title to set.
            titlepad : float, The offset of the title from the top of the Axes, in points.
            titley : float, Vertical Axes loation for the title (1.0 is the top).
                     If None (the default), titley is determined automatically to avoid decorators on the Axes.
        """
        if isinstance(fontcolor, dict):
            fontcolor = fontcolor['mode']
        kwargs = {'label':title, 'loc':loc, 'pad':titlepad, 'y':titley,
                  'fontdict':{'fontsize':fontsize, 'fontweight':fontweight, 'color':fontcolor,
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
    

    
    