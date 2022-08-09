from matplotlib import patches


class Wedge():
    def add_wedge(self, name, center, r, theta1, theta2, width=None, **kwargs):
        """Wedge shaped patch.
        
        A wedge centered at *x*, *y* center with radius *r* that
        sweeps *theta1* to *theta2* (in degrees).  If *width* is given,
        then a partial wedge is drawn from inner radius *r* - *width*
        to outer radius *r*.
        
        Args:
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
        kwargs['center'] = center
        kwargs['r'] = r
        kwargs['theta1'] = theta1
        kwargs['theta2'] = theta2
        kwargs['width'] = width
        self._params.ydata[name]['kwargs'] = kwargs
        self._params.ydata[name]['transform'] = 'ax'
        self._params.ydata[name]['plotmode'] = 'wedge'
        self._params.ydata[name]['plotfunc'] = self._execute_plot_wedge
        return self
    
    def _execute_plot_wedge(self, fig, ax, i, j):
        poly = patches.Wedge(**j['kwargs'])
        if j['transform']=='ax':
            ax.add_patch(poly)
        else:
            fig.add_artist(poly)