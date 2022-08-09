from matplotlib import patches


class Rectangle():
    def add_rectangle(self, name, xy, width, height, angle=0.0, **kwargs):
        """Make a rectangle plot.
        
        The rectangle extends from ``xy[0]`` to ``xy[0] + width`` in x-direction
        and from ``xy[1]`` to ``xy[1] + height`` in y-direction. ::

          :                +------------------+
          :                |                  |
          :              height               |
          :                |                  |
          :               (xy)---- width -----+

        Args:
            xy: (float, float) The anchor point.
            width : float, Rectangle width.
            height : float, Rectangle height.
            angle : float, Rotation in degrees anti-clockwise about *xy*.
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
        kwargs['width'] = width
        kwargs['height'] = height
        kwargs['angle'] = angle
        self._params.ydata[name]['kwargs'] = kwargs
        self._params.ydata[name]['data'] = xy
        self._params.ydata[name]['transform'] = 'ax'
        self._params.ydata[name]['plotmode'] = 'rectangle'
        self._params.ydata[name]['plotfunc'] = self._execute_plot_rectangle
        return self
    
    def _execute_plot_rectangle(self, fig, ax, i, j):
        poly = patches.Rectangle(j['data'], **j['kwargs'])
        if j['transform']=='ax':
            ax.add_patch(poly)
        else:
            fig.add_artist(poly)