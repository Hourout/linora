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
            linecolor: line color, eg. 'blue' or '0.75' or 'g' or '#FFDD44' or (1.0,0.2,0.3) or 'chartreuse'.
            linewidth: line width.
            linelink: Set the drawstyle of the plot. The drawstyle determines how the points are connected.
                {'default', 'steps', 'steps-pre', 'steps-mid', 'steps-post'}.
                'default': the points are connected with straight lines.
                'steps-pre': The step is at the beginning of the line segment.
                'steps-mid': The step is halfway between the points.
                'steps-post': The step is at the end of the line segment.
                'steps': is equal to 'steps-pre' and is maintained for backward-compatibility.
            markfill: {'full', 'left', 'right', 'bottom', 'top', 'none'}
                'full': Fill the whole marker with the markerfacecolor.
                'left', 'right', 'bottom', 'top': Fill the marker half at the given side with the markerfacecolor. 
                                                  The other half of the marker is filled with markerfacecoloralt.
                'none': No filling.
            mark: marker style string, 
                {'.': 'point', ',': 'pixel', 'o': 'circle', 'v': 'triangle_down', '^': 'triangle_up', 
                '<': 'triangle_left', '>': 'triangle_right', '1': 'tri_down', '2': 'tri_up', '3': 'tri_left', 
                '4': 'tri_right', '8': 'octagon', 's': 'square', 'p': 'pentagon', '*': 'star', 'h': 'hexagon1', 
                'H': 'hexagon2', '+': 'plus', 'x': 'x', 'D': 'diamond', 'd': 'thin_diamond', '|': 'vline', 
                '_': 'hline', 'P': 'plus_filled', 'X': 'x_filled', 0: 'tickleft', 1: 'tickright', 2: 'tickup', 
                3: 'tickdown', 4: 'caretleft', 5: 'caretright', 6: 'caretup', 7: 'caretdown', 8: 'caretleftbase', 
                9: 'caretrightbase', 10: 'caretupbase', 11: 'caretdownbase', 
                'None': 'nothing', None: 'nothing', ' ': 'nothing', '': 'nothing'}
            markedgecolor: marker edge color.
            markedgewidth: float, marker edge width.
            markfacecolor: marker face color.
            markfacecoloralt: marker face coloralt.
            marksize: float, marker size.
            markevery: None or int or (int, int) or slice or list[int] or float or (float, float) or list[bool]
            antialiased: Set whether to use antialiased rendering.
        """
        kwargs['label'] = name
        if 'linecolor' not in kwargs:
            kwargs['color'] = self._params.color.pop(0)[1]#tuple([round(np.random.uniform(0, 1),1) for _ in range(3)])
        elif isinstance(kwargs['linecolor'], dict):
            kwargs['color'] = kwargs.pop('linecolor')['mode']
        else:
            kwargs['color'] = kwargs.pop('linecolor')
        if 'linelink' in kwargs:
            kwargs['drawstyle'] = kwargs.pop('linelink')
        if 'markfill' in kwargs:
            kwargs['fillstyle'] = kwargs.pop('markfill')
        if 'mark' in kwargs:
            kwargs['marker'] = kwargs.pop('mark')
        if 'markedgecolor' in kwargs:
            kwargs['markeredgecolor'] = kwargs.pop('markedgecolor')
        if 'markedgewidth' in kwargs:
            kwargs['markeredgewidth'] = kwargs.pop('markedgewidth')
        if 'markfacecolor' in kwargs:
            kwargs['markerfacecolor'] = kwargs.pop('markfacecolor')
        if 'markfacecoloralt' in kwargs:
            kwargs['markerfacecoloralt'] = kwargs.pop('markfacecoloralt')
        if 'marksize' in kwargs:
            kwargs['markersize'] = kwargs.pop('marksize')
        self._params.ydata[name]['kwargs'] = kwargs
        self._params.ydata[name]['xdata'] = xdata
        self._params.ydata[name]['ydata'] = ydata
        self._params.ydata[name]['plotmode'] = 'line'
        self._params.ydata[name]['plotfunc'] = self._execute_plot_line
        return self
    
    def _execute_plot_line(self, fig, ax, i, j):
        ax_plot = ax.plot(j['xdata'], j['ydata'], **j['kwargs'])
    