from matplotlib import patches


class Polygon():
    def add_polygon(self, name, xy, closed=True, **kwargs):
        """Make a polygon plot.
        
        Args:
            xy: a numpy array with shape Nx2.
            closed: If True, the polygon will be closed so the 
                starting and ending points are the same.
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
        kwargs['closed'] = closed
        self._params.ydata[name]['kwargs'] = kwargs
        self._params.ydata[name]['data'] = xy
        self._params.ydata[name]['transform'] = 'ax'
        self._params.ydata[name]['plotmode'] = 'polygon'
        self._params.ydata[name]['plotfunc'] = self._execute_plot_polygon
        return self
    
    def _execute_plot_polygon(self, fig, ax, i, j):
        poly = patches.Polygon(j['data'], **j['kwargs'])
        if j['transform']=='ax':
            ax.add_patch(poly)
        else:
            fig.add_artist(poly)