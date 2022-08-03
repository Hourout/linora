import numpy as np
from matplotlib import patches


class Circle():
    def add_circle(self, name, xy, radius=5, **kwargs):
        """Make a circle plot.
        
        Args:
            xy: Create a true circle at center (x, y) with given *radius*.
            radius: circle radius.
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
        kwargs['radius'] = radius
        self._params.ydata[name]['kwargs'] = kwargs
        self._params.ydata[name]['data'] = xy
        self._params.ydata[name]['transform'] = 'ax'
        self._params.ydata[name]['plotmode'] = 'circle'
        self._params.ydata[name]['plotfunc'] = self._execute_plot_circle
        return self
    
    def _execute_plot_circle(self, fig, ax, i, j):
        poly = patches.Circle(j['data'], **j['kwargs'])
        if j['transform']=='ax':
            ax.add_patch(poly)
        else:
            fig.add_artist(poly)