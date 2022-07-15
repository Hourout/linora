import numpy as np


class Scatter():
    def add_scatter(self, name, xdata, ydata, **kwargs):
        """A scatter plot of *y* vs. *x* with varying marker size and/or color.
        
        Args:
            name: data name.
            xdata: x-axis data.
            ydata: y-axis data.
            pointsize: float or array-like, shape (n, ), The marker size in points**2.
            pointcolor: array-like or list of colors or color, optional
                The marker colors. Possible values:

                - A scalar or sequence of n numbers to be mapped to colors using *cmap* and *norm*.
                - A 2D array in which the rows are RGB or RGBA.
                - A sequence of colors of length n.
                - A single color format string.

                Note that *c* should not be a single numeric RGB or RGBA sequence
                because that is indistinguishable from an array of values to be
                colormapped. If you want to specify the same RGB or RGBA value for
                all points, use a 2D array with a single row.  Otherwise, value-
                matching will have precedence in case of a size matching with *x* and *y*.

                If you wish to specify a single color for all points prefer the *color* keyword argument.

                Defaults to `None`. In that case the marker color is determined
                by the value of *color*, *facecolor* or *facecolors*. In case
                those are not specified or `None`, the marker color is determined
                by the next color of the ``Axes``' current "shape and fill" color cycle.
            marker: marker style string, 
                {'.': 'point', ',': 'pixel', 'o': 'circle', 'v': 'triangle_down', '^': 'triangle_up', 
                '<': 'triangle_left', '>': 'triangle_right', '1': 'tri_down', '2': 'tri_up', '3': 'tri_left', 
                '4': 'tri_right', '8': 'octagon', 's': 'square', 'p': 'pentagon', '*': 'star', 'h': 'hexagon1', 
                'H': 'hexagon2', '+': 'plus', 'x': 'x', 'D': 'diamond', 'd': 'thin_diamond', '|': 'vline', 
                '_': 'hline', 'P': 'plus_filled', 'X': 'x_filled', 0: 'tickleft', 1: 'tickright', 2: 'tickup', 
                3: 'tickdown', 4: 'caretleft', 5: 'caretright', 6: 'caretup', 7: 'caretdown', 8: 'caretleftbase', 
                9: 'caretrightbase', 10: 'caretupbase', 11: 'caretdownbase', 
                'None': 'nothing', None: 'nothing', ' ': 'nothing', '': 'nothing'}
            cmap: {'viridis', 'jet'}. *cmap* is only used if *pointcolor* is an array of floats.
            norm: If *c* is an array of floats, *norm* is used to scale the color data, 
                  *c*, in the range 0 to 1, in order to map into the colormap *cmap*.
            vmin, vmax : float, default: None
                        *vmin* and *vmax* are used in conjunction with the default norm to
                        map the color array *c* to the colormap *cmap*. If None, the
                        respective min and max of the color array is used.
                        It is deprecated to use *vmin*/*vmax* when *norm* is given.

            alpha : float, default: None, The alpha blending value, between 0 (transparent) and 1 (opaque).
            linewidths : float or array-like, The linewidth of the marker edges. 
                         Note: The default *edgecolors* is 'face'. You may want to change this as well.

            edgecolors : {'face', 'none', *None*} or color or sequence of color
                The edge color of the marker. Possible values:

                - 'face': The edge color will always be the same as the face color.
                - 'none': No patch boundary will be drawn.
                - A color or sequence of colors.

                For non-filled markers, *edgecolors* is ignored. Instead, the color
                is determined like with 'face', i.e. from *c*, *colors*, or *facecolors*.

            plotnonfinite : bool, default: False
                Whether to plot points with nonfinite *c* (i.e. ``inf``, ``-inf``
                or ``nan``). If ``True`` the points are drawn with the *bad*
                colormap color (see `.Colormap.set_bad`).

        """
#         if 'pointcolor' in kwargs or not self._params.set_label:
#             self._params.set_label = False
#         self._params.colorbar.add('viridis' if 'cmap' not in kwargs else kwargs['cmap'])
        
        if 'pointsize' in kwargs:
            kwargs['s'] = kwargs.pop('pointsize')
        if 'pointcolor' not in kwargs:
            kwargs['c'] = [tuple([round(np.random.uniform(0, 1),1) for _ in range(3)])]*len(ydata)
        else:
            if isinstance(kwargs['pointcolor'], dict):
                kwargs['c'] = kwargs.pop('pointcolor')['mode']
            else:
                kwargs['c'] = kwargs.pop('pointcolor')
        self._params.ydata[name]['kwargs'] = kwargs
        self._params.ydata[name]['xdata'] = xdata
        self._params.ydata[name]['ydata'] = ydata
        self._params.ydata[name]['plotmode'] = 'scatter'
        return self
    
    