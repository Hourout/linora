class Bar():
    def add_bar(self, name, xdata, ydata, **kwargs):
        """Make a bar plot.
        
        Args:
            name: data name.
            xdata: x-axis data.
            ydata: y-axis data.
            width : float or array-like, default: 0.8
                The width(s) of the bars.

            vertical: bool, default: True
            
            bottom : float or array-like, default: 0
                The y coordinate(s) of the bars bases.

            align : {'center', 'edge'}, default: 'center'
                Alignment of the bars to the *x* coordinates:

                - 'center': Center the base on the *x* positions.
                - 'edge': Align the left edges of the bars with the *x* positions.
                
            barcolor : color or list of color, optional
                The colors of the bar faces.

            edgecolor : color or list of color, optional
                The colors of the bar edges.

            linewidth : float or array-like, optional
                Width of the bar edge(s). If 0, don't draw edges.

            tick_label : str or list of str, optional
                The tick labels of the bars.
                Default: None (Use default numeric labels.)

            xerr, yerr : float or array-like of shape(N,) or shape(2, N), optional
                If not *None*, add horizontal / vertical errorbars to the bar tips.
                The values are +/- sizes relative to the data:

                - scalar: symmetric +/- values for all bars
                - shape(N,): symmetric +/- values for each bar
                - shape(2, N): Separate - and + values for each bar. First row
                  contains the lower errors, the second row contains the upper
                  errors.
                - *None*: No errorbar. (Default)

                See :doc:`/gallery/statistics/errorbar_features`
                for an example on the usage of ``xerr`` and ``yerr``.

            ecolor : color or list of color, default: 'black'
                The line color of the errorbars.

            capsize : float, default: :rc:`errorbar.capsize`
               The length of the error bar caps in points.

            error_kw : dict, optional
                Dictionary of kwargs to be passed to the `~.Axes.errorbar`
                method. Values of *ecolor* or *capsize* defined here take
                precedence over the independent kwargs.

            log : bool, default: False
                If *True*, set the y-axis to be log scale.

            linestyle: line style, {'-', '--', '-.', ':'}.
                '-' or 'solid': solid line
                '--' or 'dashed': dashed line
                '-.' or 'dashdot': dash-dotted line
                ':' or 'dotted': dotted line
                'none', 'None', ' ', or '': draw nothing
            
            labels: array-like, optional
                A list of label texts, that should be displayed. 
                If not given, the label texts will be the data values formatted with fmt.

            label_type: {'edge', 'center'}, default: 'edge'
                The label type. Possible values:
                'edge': label placed at the end-point of the bar segment, 
                        and the value displayed will be the position of that end-point.
                'center': label placed in the center of the bar segment, 
                and the value displayed will be the length of that segment. 

            padding: float, default: 0
                Distance of label from the end of the bar, in points.
        """
        if 'barcolor' not in kwargs:
            kwargs['color'] = self._params.color.pop(0)[1]
        elif isinstance(kwargs['barcolor'], dict):
            kwargs['color'] = kwargs.pop('barcolor')['mode']
        else:
            kwargs['color'] = kwargs.pop('barcolor')
        
        barlabel = {}
        if 'labels' in kwargs:
            barlabel['labels'] = kwargs.pop('labels')
        if 'label_type'in kwargs:
            barlabel['label_type'] = kwargs.pop('label_type')
        if 'padding' in kwargs:
            barlabel['padding'] = kwargs.pop('padding')
        
        self._params.ydata[name]['vertical'] = kwargs.pop('vertical') if 'vertical' in kwargs else True
        if not self._params.ydata[name]['vertical']:
            if 'width' in kwargs:
                kwargs['height'] = kwargs.pop('width')
            if 'bottom' in kwargs:
                kwargs['left'] = kwargs.pop('bottom')
        self._params.ydata[name]['barlabel'] = barlabel
        self._params.ydata[name]['kwargs'] = kwargs
        self._params.ydata[name]['xdata'] = xdata
        self._params.ydata[name]['ydata'] = ydata
        self._params.ydata[name]['plotmode'] = 'bar'
        self._params.ydata[name]['plotfunc'] = self._execute_plot_bar
        return self
    
    def _execute_plot_bar(self, fig, ax, i, j):
        if j['vertical']:
            ax_plot = ax.bar(j['xdata'], j['ydata'], **j['kwargs'])
        else:
            ax_plot = ax.barh(j['xdata'], j['ydata'], **j['kwargs'])
        ax_plot.set_label(i)
        if len(j['barlabel'])>0:
            if 'label_type' in j['barlabel']:
                if isinstance(j['barlabel']['label_type'], str):
                    label_type = [j['barlabel']['label_type']]
                else:
                    label_type = j['barlabel']['label_type']
            else:
                label_type = ['edge']
            t = j['barlabel'].copy()
            for i in label_type:
                t['label_type'] = i
                ax.bar_label(ax_plot, **t)