import json
from collections import defaultdict

from linora.utils._config import Config


class Coordinate():
    def __init__(self):
        self._params = Config()
        self._params.ydata = defaultdict(defaultdict)
        self._params.annotate = dict()
        self._params.figure = {'figsize':(12.8, 7.2)}
        
        self._params.axis = {'normal':{'axis':None, 'xinvert':False, 'yinvert':False, 'xtick':{}, 'ytick':{},
                             'xlabel':None, 'ylabel':None, 'xtickposition':None, 'ytickposition':None}}
        
        self._params.label = {'normal':{'xlabel':{'xlabel':None}, 'ylabel':{'ylabel':None}}}
        
        self._params.legend = dict()
        self._params.spine = {'show':{}, 'alpha':{}, 'color':{}, 'width':{}, 'style':{}, 'position':{}}
        self._params.text = dict()
        self._params.theme = 'ggplot'
        self._params.title = {'label':None}
        self._params.twin = dict()
        
        self._params.set_label = True
        self._params.colorbar = set()
        
    def get_config(self, json_path=None):
        config = {
            'mode': 'plot',
            'annotate': self._params.annotate,
            'ydata': self._params.ydata,
            'theme': self._params.theme,
            'figure': self._params.figure,
            'axis': self._params.axis,
            'label': self._params.label,
            'legend': self._params.legend,
            'spine': self._params.spine,
            'text': self._params.text,
            'title': self._params.title,
            'twin': self._params.twin,
            'set_label': self._params.set_label,
            'colorbar': self._params.colorbar,
        }
        if json_path is not None:
            with open(json_path, 'w') as f:
                json.dump(config, f)
        return config
    
    def set_config(self, config):
        if isinstance(config, str):
            assert config.endswith('.json'), f'{config} not a json file.'
            with open(config) as f:
                config = json.load(f)
        assert isinstance(config, dict), f'{config} not a dict.'
        assert config['mode']=='plot', 'config info not match.'
        self._params.ydata = config['ydata']
        self._params.annotate = config['annotate']
        self._params.theme = config['theme']
        self._params.figure = config['figure']
        self._params.axis = config['axis']
        self._params.label = config['label']
        self._params.legend = config['legend']
        self._params.spine = config['spine']
        self._params.text = config['text']
        self._params.title = config['title']
        self._params.twin = config['twin']
        self._params.set_label = config['set_label']
        self._params.colorbar = config['colorbar']
        return self
            
    def render(self, image_path=None, if_show=True, **kwargs):
        """show and save plot."""
        fig = self._execute()
        if image_path is not None:
            fig.savefig(image_path, **kwargs)
        if if_show:
            return fig.show()
    
    def set_annotate(self, text, xy, xytext=None, xycoords=None, textcoords=None, 
                     arrowprops=None, annotation_clip=None, **kwargs):
        """Annotate the point *xy* with text *text*.
        
        Args:
            text : str, The text of the annotation.

            xy : (float, float)
                The point *(x, y)* to annotate. The coordinate system is determined
                by *xycoords*.

            xytext : (float, float), default: *xy*
                The position *(x, y)* to place the text at. The coordinate system
                is determined by *textcoords*.

            xycoords : str or `.Artist` or `.Transform` or callable or (float, float), default: 'data'

                The coordinate system that *xy* is given in. The following types
                of values are supported:

                - One of the following strings:

                  ==================== ============================================
                  Value                Description
                  ==================== ============================================
                  'figure points'      Points from the lower left of the figure
                  'figure pixels'      Pixels from the lower left of the figure
                  'figure fraction'    Fraction of figure from lower left
                  'subfigure points'   Points from the lower left of the subfigure
                  'subfigure pixels'   Pixels from the lower left of the subfigure
                  'subfigure fraction' Fraction of subfigure from lower left
                  'axes points'        Points from lower left corner of axes
                  'axes pixels'        Pixels from lower left corner of axes
                  'axes fraction'      Fraction of axes from lower left
                  'data'               Use the coordinate system of the object
                                       being annotated (default)
                  'polar'              *(theta, r)* if not native 'data'
                                       coordinates
                  ==================== ============================================

                  Note that 'subfigure pixels' and 'figure pixels' are the same
                  for the parent figure, so users who want code that is usable in
                  a subfigure can use 'subfigure pixels'.

                - An `.Artist`: *xy* is interpreted as a fraction of the artist's
                  `~matplotlib.transforms.Bbox`. E.g. *(0, 0)* would be the lower
                  left corner of the bounding box and *(0.5, 1)* would be the
                  center top of the bounding box.

                - A `.Transform` to transform *xy* to screen coordinates.

                - A function with one of the following signatures::

                    def transform(renderer) -> Bbox
                    def transform(renderer) -> Transform

                  where *renderer* is a `.RendererBase` subclass.

                  The result of the function is interpreted like the `.Artist` and
                  `.Transform` cases above.

                - A tuple *(xcoords, ycoords)* specifying separate coordinate
                  systems for *x* and *y*. *xcoords* and *ycoords* must each be
                  of one of the above described types.

                See :ref:`plotting-guide-annotation` for more details.

            textcoords : str or `.Artist` or `.Transform` or callable or (float, float), default: value of *xycoords*
                The coordinate system that *xytext* is given in.

                All *xycoords* values are valid as well as the following
                strings:

                =================   =========================================
                Value               Description
                =================   =========================================
                'offset points'     Offset (in points) from the *xy* value
                'offset pixels'     Offset (in pixels) from the *xy* value
                =================   =========================================

            arrowprops : dict, optional
                The properties used to draw a `.FancyArrowPatch` arrow between the
                positions *xy* and *xytext*. Note that the edge of the arrow
                pointing to *xytext* will be centered on the text itself and may
                not point directly to the coordinates given in *xytext*.

                If *arrowprops* does not contain the key 'arrowstyle' the
                allowed keys are:

                ==========   ======================================================
                Key          Description
                ==========   ======================================================
                width        The width of the arrow in points
                headwidth    The width of the base of the arrow head in points
                headlength   The length of the arrow head in points
                shrink       Fraction of total length to shrink from both ends
                ?            Any key to :class:`matplotlib.patches.FancyArrowPatch`
                ==========   ======================================================

                If *arrowprops* contains the key 'arrowstyle' the
                above keys are forbidden.  The allowed values of
                ``'arrowstyle'`` are:

                ============   =============================================
                Name           Attrs
                ============   =============================================
                ``'-'``        None
                ``'->'``       head_length=0.4,head_width=0.2
                ``'-['``       widthB=1.0,lengthB=0.2,angleB=None
                ``'|-|'``      widthA=1.0,widthB=1.0
                ``'-|>'``      head_length=0.4,head_width=0.2
                ``'<-'``       head_length=0.4,head_width=0.2
                ``'<->'``      head_length=0.4,head_width=0.2
                ``'<|-'``      head_length=0.4,head_width=0.2
                ``'<|-|>'``    head_length=0.4,head_width=0.2
                ``'fancy'``    head_length=0.4,head_width=0.4,tail_width=0.4
                ``'simple'``   head_length=0.5,head_width=0.5,tail_width=0.2
                ``'wedge'``    tail_width=0.3,shrink_factor=0.5
                ============   =============================================

                Valid keys for `~matplotlib.patches.FancyArrowPatch` are:

                ===============  ==================================================
                Key              Description
                ===============  ==================================================
                arrowstyle       the arrow style
                connectionstyle  the connection style
                relpos           default is (0.5, 0.5)
                patchA           default is bounding box of the text
                patchB           default is None
                shrinkA          default is 2 points
                shrinkB          default is 2 points
                mutation_scale   default is text size (in points)
                mutation_aspect  default is 1.
                ?                any key for :class:`matplotlib.patches.PathPatch`
                ===============  ==================================================

                Defaults to None, i.e. no arrow is drawn.

            annotation_clip : bool or None, default: None
                Whether to draw the annotation when the annotation point *xy* is
                outside the axes area.

                - If *True*, the annotation will only be drawn when *xy* is
                  within the axes.
                - If *False*, the annotation will always be drawn.
                - If *None*, the annotation will only be drawn when *xy* is
                  within the axes and *xycoords* is 'data'.
        """
        if xytext is not None:
            kwargs['xytext'] = xytext
        if xycoords is not None:
            kwargs['xycoords'] = xycoords
        if textcoords is not None:
            kwargs['textcoords'] = textcoords
        if arrowprops is not None:
            kwargs['arrowprops'] = arrowprops
        if annotation_clip is not None:
            kwargs['annotation_clip'] = annotation_clip
        kwargs['text'] = text
        kwargs['xy'] = xy
        self._params.annotate[len(self._params.annotate)] = kwargs
        return self
        
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
                 labelbottom=None, labeltop=None, labelleft=None, labelright=None,
                 twin=None,
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
            twin: twin Axes, 'x' or 'y'.
        """
        if twin is not None:
            self._params.axis[twin] = {'axis':None, 'xinvert':False, 'yinvert':False, 'xtick':{}, 'ytick':{},
                                       'xlabel':None, 'ylabel':None, 'xtickposition':None, 'ytickposition':None}
        mode = twin if twin is not None else 'normal'
        if axis is not None:
            self._params.axis[mode]['axis'] = axis
        elif xmin is not None or xmax is not None or ymin is not None or ymax is not None:
            self._params.axis[mode]['axis'] = [xmin, xmax, ymin, ymax]
        self._params.axis[mode]['xinvert'] = invert or xinvert
        self._params.axis[mode]['yinvert'] = invert or yinvert
        if xlabel is not None:
            self._params.axis[mode]['xlabel'] = xlabel
        if ylabel is not None:
            self._params.axis[mode]['ylabel'] = ylabel
        if xtickposition is not None:
            self._params.axis[mode]['xtickposition'] = xtickposition
        if ytickposition is not None:
            self._params.axis[mode]['ytickposition'] = ytickposition
        
        self._set_axis(xtickwhich, ytickwhich, tickwhich, 'which', mode)
        self._set_axis(xtickcolor, ytickcolor, tickcolor, 'color', mode)
        self._set_axis(xtickheight, ytickheight, tickheight, 'length', mode)
        self._set_axis(xtickwidth, ytickwidth, tickwidth, 'width', mode)
        self._set_axis(xtickloc, ytickloc, tickloc, 'direction', mode)
        self._set_axis(xtickpad, ytickpad, tickpad, 'pad', mode)
        self._set_axis(xlabelsize, ylabelsize, labelsize, 'labelsize', mode)
        self._set_axis(xlabelcolor, ylabelcolor, labelcolor, 'labelcolor', mode)
        self._set_axis(xlabelrotate, ylabelrotate, labelrotate, 'labelrotation', mode)
        self._set_axis(xgridcolor, ygridcolor, gridcolor, 'grid_color', mode)
        self._set_axis(xgridalpha, ygridalpha, gridalpha, 'grid_alpha', mode)
        self._set_axis(xgridwidth, ygridwidth, gridwidth, 'grid_linewidth', mode)
        self._set_axis(xgridstyle, ygridstyle, gridstyle, 'grid_linestyle', mode)
            
        if tickbottom is not None:
            self._params.axis[mode]['xtick']['bottom'] = tickbottom
        if ticktop is not None:
            self._params.axis[mode]['xtick']['top'] = ticktop
        if tickleft is not None:
            self._params.axis[mode]['ytick']['left'] = tickleft
        if tickright is not None:
            self._params.axis[mode]['ytick']['right'] = tickright
        if labelbottom is not None:
            self._params.axis[mode]['xtick']['labelbottom'] = labelbottom
        if labeltop is not None:
            self._params.axis[mode]['xtick']['labeltop'] = labeltop
        if labelleft is not None:
            self._params.axis[mode]['ytick']['labelleft'] = labelleft
        if labelright is not None:
            self._params.axis[mode]['ytick']['labelright'] = labelright

        if not (tickshow and xtickshow):
            self._params.axis[mode]['xtick'] = {'length':0, 'labelsize':0, 'which':'both'}
        if not (tickshow and ytickshow):
            self._params.axis[mode]['ytick'] = {'length':0, 'labelsize':0, 'which':'both'}
        return self
    
    def _set_axis(self, x, y, xy, s, mode):
        if x is not None:
            self._params.axis[mode]['xtick'][s] = x['mode'] if isinstance(x, dict) else x
        if y is not None:
            self._params.axis[mode]['ytick'][s] = y['mode'] if isinstance(y, dict) else y
        if xy is not None:
            self._params.axis[mode]['xtick'][s] = xy['mode'] if isinstance(xy, dict) else xy
            self._params.axis[mode]['ytick'][s] = xy['mode'] if isinstance(xy, dict) else xy
        
    def set_figure(self, width=12.8, height=7.2, dpi=None, facecolor=None, edgecolor=None, frameon=True, clear=False):
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
    
    def set_label(self, xlabel=None, ylabel=None, 
                  loc=None, xloc=None, yloc=None, 
                  pad=None, xpad=None, ypad=None,
                  fontsize=None, xfontsize=None, yfontsize=None, 
                  fontcolor=None, xfontcolor=None, yfontcolor=None, 
                  fontfamily=None, xfontfamily=None, yfontfamily=None, 
                  fontstyle=None, xfontstyle=None, yfontstyle=None, 
                  twin=None,
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
            twin: twin Axes, 'x' or 'y'.
        """
        if twin is not None:
            self._params.label[twin] = {'xlabel':{'xlabel':None}, 'ylabel':{'ylabel':None}}
        mode = twin if twin is not None else 'normal'
        self._set_label(xlabel, y=None, xy=None, xkey='xlabel', mode=mode)
        self._set_label(x=None, y=ylabel, xy=None, ykey='ylabel', mode=mode)
        self._set_label(x=xloc, y=yloc, xy=loc, xkey='loc', ykey='loc', mode=mode)
        self._set_label(x=xpad, y=ypad, xy=pad, xkey='labelpad', ykey='labelpad', mode=mode)
        self._set_label(x=xfontsize, y=yfontsize, xy=fontsize, xkey='fontsize', ykey='fontsize', mode=mode)
        self._set_label(x=xfontcolor, y=yfontcolor, xy=fontcolor, xkey='color', ykey='color', mode=mode)
        self._set_label(x=xfontfamily, y=yfontfamily, xy=fontfamily, xkey='fontfamily', ykey='fontfamily', mode=mode)
        self._set_label(x=xfontstyle, y=yfontstyle, xy=fontstyle, xkey='fontstyle', ykey='fontstyle', mode=mode)
        return self
    
    def _set_label(self, x, y, xy, xkey=None, ykey=None, mode='normal'):
        if x is not None:
            self._params.label[mode]['xlabel'][xkey] = x['mode'] if isinstance(x, dict) else x
        if y is not None:
            self._params.label[mode]['ylabel'][ykey] = y['mode'] if isinstance(y, dict) else y
        if xy is not None:
            self._params.label[mode]['xlabel'][xkey] = xy['mode'] if isinstance(xy, dict) else xy
            self._params.label[mode]['ylabel'][ykey] = xy['mode'] if isinstance(xy, dict) else xy
    
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
        kwargs['loc'] = loc
        self._params.legend.update(kwargs)
        return self
        
    def set_spine(self, spineshow=None, leftshow=None, rightshow=None, topshow=None, bottomshow=None,
                  spinecolor=None, leftcolor=None, rightcolor=None, topcolor=None, bottomcolor=None,
                  spinewidth=None, leftwidth=None, rightwidth=None, topwidth=None, bottomwidth=None,
                  spinestyle=None, leftstyle=None, rightstyle=None, topstyle=None, bottomstyle=None,
                  spinealpha=None, leftalpha=None, rightalpha=None, topalpha=None, bottomalpha=None,
                  leftloc=None, rightloc=None, toploc=None, bottomloc=None,
                 ):
        """Set a wireframes for the Axes.
        
        Args:
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
            spinealpha: all wireframes alpha.
            leftalpha: left wireframes alpha.
            rightalpha: right wireframes alpha.
            topalpha: top wireframes alpha.
            bottomalpha: bottom wireframes alpha.
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
        self._set_spine(spinealpha, leftalpha, rightalpha, topalpha, bottomalpha, 'alpha')
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
        
    def set_text(self, x, y, text, **kwargs):
        """Add text to the Axes.
        
        Args:
            x, y : float, The position to place the text. 
            text : str, The text.

        """
        kwargs['x'] = x
        kwargs['y'] = y
        kwargs['s'] = text
        self._params.text[len(self._params.text)] = kwargs
        return self
        
    def set_theme(self, theme):
        """Set a theme for the Axes.
        
        Args:
            theme: str, figure theme.
        """
        self._params.theme = theme
        return self
    
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
                              'verticalalignment': 'baseline', 'horizontalalignment': loc}}
        self._params.title = kwargs
        return self
        
    def set_twin(self, name, axis='x'):
        """Create a twin Axes sharing the x-y axis.
        
        Args:
            name: data name.
            axis: 'x' or 'y'.
        """
        self._params.twin[name] = axis
        return self
    
