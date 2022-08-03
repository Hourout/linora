from matplotlib import patches


class Ellipse():
    def add_ellipse(self, name, xy, width, height, angle=0, **kwargs):
        """Make a ellipse plot.
        
        Args:
            xy: (float, float) xy coordinates of ellipse centre.
            width : float, Total length (diameter) of horizontal axis.
            height : float, Total length (diameter) of vertical axis.
            angle : float, Rotation in degrees anti-clockwise.
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
        kwargs['width'] = width
        kwargs['height'] = height
        kwargs['angle'] = angle
        self._params.ydata[name]['kwargs'] = kwargs
        self._params.ydata[name]['data'] = xy
        self._params.ydata[name]['transform'] = 'ax'
        self._params.ydata[name]['plotmode'] = 'ellipse'
        self._params.ydata[name]['plotfunc'] = self._execute_plot_ellipse
        return self
    
    def _execute_plot_ellipse(self, fig, ax, i, j):
        poly = patches.Ellipse(j['data'], **j['kwargs'])
        if j['transform']=='ax'
            ax.add_patch(poly)
        else:
            fig.add_artist(poly)