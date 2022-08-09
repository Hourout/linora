from matplotlib import patches


class RegularPolygon():
    def add_regularpolygon(self, name, xy, num_vertices, radius=5, orientation=0, **kwargs):
        """Make a regular polygon plot.
        
        Args:
            xy: (float, float) The anchor point.
            num_vertices : int, The number of vertices.
            radius : float, The distance from the center to each of the vertices.
            orientation : float, The polygon rotation angle (in radians).
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
        kwargs['numVertices'] = num_vertices
        kwargs['radius'] = radius
        kwargs['orientation'] = orientation
        self._params.ydata[name]['kwargs'] = kwargs
        self._params.ydata[name]['data'] = xy
        self._params.ydata[name]['transform'] = 'ax'
        self._params.ydata[name]['plotmode'] = 'regularpolygon'
        self._params.ydata[name]['plotfunc'] = self._execute_plot_regularpolygon
        return self
    
    def _execute_plot_regularpolygon(self, fig, ax, i, j):
        poly = patches.RegularPolygon(j['data'], **j['kwargs'])
        if j['transform']=='ax':
            ax.add_patch(poly)
        else:
            fig.add_artist(poly)