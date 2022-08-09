from matplotlib import patches


class Arrow():
    def add_arrow(self, name, x, y, dx, dy, width=1.0, **kwargs):
        """An arrow patch.
        
        Draws an arrow from (*x*, *y*) to (*x* + *dx*, *y* + *dy*).
        The width of the arrow is scaled by *width*.
        
        Args:
            x : float, x coordinate of the arrow tail.
            y : float, y coordinate of the arrow tail.
            dx : float, Arrow length in the x direction.
            dy : float, Arrow length in the y direction.
            width : float, default: 1
                Scale factor for the width of the arrow. With a default value of 1,
                the tail width is 0.2 and head width is 0.6.
            alpha: scalar or None
            animated: bool
            antialiased or aa: unknown
            capstyle: `.CapStyle` or {'butt', 'projecting', 'round'}
            clip_box: `.Bbox`
            clip_on: bool
            clip_path: Patch or (Path, Transform) or None
            color: color
            contains: unknown
            edgecolor or ec: color or None or 'auto'
            facecolor or fc: color or None
            fill: bool
            gid: str
            hatch: {'/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*'}
            in_layout: bool
            joinstyle: `.JoinStyle` or {'miter', 'round', 'bevel'}
            label: object
            linestyle or ls: {'-', '--', '-.', ':', '', (offset, on-off-seq), ...}
            linewidth or lw: float or None
            picker: None or bool or float or callable
            rasterized: bool
            sketch_params: (scale: float, length: float, randomness: float)
            snap: bool or None
        """
        if 'color' not in kwargs:
            kwargs['color'] = self._params.color.pop(0)[1]
        kwargs['dx'] = dx
        kwargs['dy'] = dy
        kwargs['width'] = width
        self._params.ydata[name]['kwargs'] = kwargs
        self._params.ydata[name]['xdata'] = x
        self._params.ydata[name]['ydata'] = y
        self._params.ydata[name]['transform'] = 'ax'
        self._params.ydata[name]['plotmode'] = 'arrow'
        self._params.ydata[name]['plotfunc'] = self._execute_plot_arrow
        return self
    
    def _execute_plot_arrow(self, fig, ax, i, j):
        poly = patches.Arrow(j['xdata'], j['ydata'], **j['kwargs'])
        if j['transform']=='ax':
            ax.add_patch(poly)
        else:
            fig.add_artist(poly)