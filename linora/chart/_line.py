import numpy as np


class Line():
    def add_line(self, name, xdata, ydata, **kwargs):
        """Plot y versus x as lines and/or markers.
        
        Args:
            name: data name.
            xdata: x-axis data.
            ydata: y-axis data.
            linestyle: line style, {'-', '--', '-.', ':'}.
                '-' or 'solid': solid line
                '--' or 'dashed': dashed line
                '-.' or 'dashdot': dash-dotted line
                ':' or 'dotted': dotted line
                'none', 'None', ' ', or '': draw nothing
            color: line color, eg. 'blue' or '0.75' or 'g' or '#FFDD44' or (1.0,0.2,0.3) or 'chartreuse'.
            linewidth: line width.
            drawstyle: Set the drawstyle of the plot. The drawstyle determines how the points are connected.
                {'default', 'steps', 'steps-pre', 'steps-mid', 'steps-post'}.
                'default': the points are connected with straight lines.
                'steps-pre': The step is at the beginning of the line segment.
                'steps-mid': The step is halfway between the points.
                'steps-post: The step is at the end of the line segment.
                'steps': is equal to 'steps-pre' and is maintained for backward-compatibility.
            dash_capstyle: Define how the two endpoints (caps) of an unclosed line are drawn.
                {'butt', 'projecting', 'round'}
                'butt': the line is squared off at its endpoint.
                'projecting': the line is squared off as in butt, 
                             but the filled in area extends beyond the endpoint a distance of linewidth/2.
                'round': like butt, but a semicircular cap is added to the end of the line, of radius linewidth/2.
            dash_joinstyle: Define how the connection between two line segments is drawn.
                {'miter', 'round', 'bevel'}
                'miter': the "arrow-tip" style. Each boundary of the filled-in area will extend 
                         in a straight line parallel to the tangent vector of the centerline at 
                         the point it meets the corner, until they meet in a sharp point.
                'round': stokes every point within a radius of linewidth/2 of the center lines.
                'bevel': the "squared-off" style. It can be thought of as a rounded corner where 
                         the "circular" part of the corner has been cut off.
            fillstyle: {'full', 'left', 'right', 'bottom', 'top', 'none'}
                'full': Fill the whole marker with the markerfacecolor.
                'left', 'right', 'bottom', 'top': Fill the marker half at the given side with the markerfacecolor. 
                                                  The other half of the marker is filled with markerfacecoloralt.
                'none': No filling.
            marker: marker style string, 
                {'.': 'point', ',': 'pixel', 'o': 'circle', 'v': 'triangle_down', '^': 'triangle_up', 
                '<': 'triangle_left', '>': 'triangle_right', '1': 'tri_down', '2': 'tri_up', '3': 'tri_left', 
                '4': 'tri_right', '8': 'octagon', 's': 'square', 'p': 'pentagon', '*': 'star', 'h': 'hexagon1', 
                'H': 'hexagon2', '+': 'plus', 'x': 'x', 'D': 'diamond', 'd': 'thin_diamond', '|': 'vline', 
                '_': 'hline', 'P': 'plus_filled', 'X': 'x_filled', 0: 'tickleft', 1: 'tickright', 2: 'tickup', 
                3: 'tickdown', 4: 'caretleft', 5: 'caretright', 6: 'caretup', 7: 'caretdown', 8: 'caretleftbase', 
                9: 'caretrightbase', 10: 'caretupbase', 11: 'caretdownbase', 
                'None': 'nothing', None: 'nothing', ' ': 'nothing', '': 'nothing'}
            markeredgecolor: marker edge color.
            markeredgewidth: float, marker edge width.
            markerfacecolor: marker face color.
            markerfacecoloralt: marker face coloralt.
            markersize: float, marker size.
            markevery: None or int or (int, int) or slice or list[int] or float or (float, float) or list[bool]
            solid_capstyle: same with `dash_capstyle`.
            solid_joinstyle: same with `dash_joinstyle`.
            antialiased: Set whether to use antialiased rendering.
        """
        kwargs['label'] = name
        if 'color' not in kwargs:
            kwargs['color'] = tuple([round(np.random.uniform(0, 1),1) for _ in range(3)])
        else:
            if isinstance(kwargs['color'], dict):
                kwargs['color'] = kwargs['color']['mode']
        self._params.ydata[name]['kwargs'] = kwargs
        self._params.ydata[name]['xdata'] = xdata
        self._params.ydata[name]['ydata'] = ydata
        self._params.ydata[name]['plotmode'] = 'line'
        return self
    