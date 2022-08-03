class Boxplot():
    def add_boxplot(self, name, xdata, **kwargs):
        """Make a box plot.
        
        Args:
            name: data name.
            xdata: x-axis data.
            notch : bool, default: False
                Whether to draw a notched box plot (`True`), or a rectangular box
                plot (`False`).  The notches represent the confidence interval (CI)
                around the median.  The documentation for *bootstrap* describes how
                the locations of the notches are computed by default, but their
                locations may also be overridden by setting the *conf_intervals*
                parameter.

            sym : str, optional
                The default symbol for flier points.  An empty string ('') hides
                the fliers.  If `None`, then the fliers default to 'b+'.  More
                control is provided by the *flierprops* parameter.

            vert : bool, default: True
                If `True`, draws vertical boxes.
                If `False`, draw horizontal boxes.

            whis : float or (float, float), default: 1.5
                The position of the whiskers.

                If a float, the lower whisker is at the lowest datum above
                ``Q1 - whis*(Q3-Q1)``, and the upper whisker at the highest datum
                below ``Q3 + whis*(Q3-Q1)``, where Q1 and Q3 are the first and
                third quartiles.  The default value of ``whis = 1.5`` corresponds
                to Tukey's original definition of boxplots.

                If a pair of floats, they indicate the percentiles at which to
                draw the whiskers (e.g., (5, 95)).  In particular, setting this to
                (0, 100) results in whiskers covering the whole range of the data.

                In the edge case where ``Q1 == Q3``, *whis* is automatically set
                to (0, 100) (cover the whole range of the data) if *autorange* is
                True.

                Beyond the whiskers, data are considered outliers and are plotted
                as individual points.

            bootstrap : int, optional
                Specifies whether to bootstrap the confidence intervals
                around the median for notched boxplots. If *bootstrap* is
                None, no bootstrapping is performed, and notches are
                calculated using a Gaussian-based asymptotic approximation
                (see McGill, R., Tukey, J.W., and Larsen, W.A., 1978, and
                Kendall and Stuart, 1967). Otherwise, bootstrap specifies
                the number of times to bootstrap the median to determine its
                95% confidence intervals. Values between 1000 and 10000 are
                recommended.

            usermedians : 1D array-like, optional
                A 1D array-like of length ``len(x)``.  Each entry that is not
                `None` forces the value of the median for the corresponding
                dataset.  For entries that are `None`, the medians are computed
                by Matplotlib as normal.

            conf_intervals : array-like, optional
                A 2D array-like of shape ``(len(x), 2)``.  Each entry that is not
                None forces the location of the corresponding notch (which is
                only drawn if *notch* is `True`).  For entries that are `None`,
                the notches are computed by the method specified by the other
                parameters (e.g., *bootstrap*).

            positions : array-like, optional
                The positions of the boxes. The ticks and limits are
                automatically set to match the positions. Defaults to
                ``range(1, N+1)`` where N is the number of boxes to be drawn.

            widths : float or array-like
                The widths of the boxes.  The default is 0.5, or ``0.15*(distance
                between extreme positions)``, if that is smaller.

            patch_artist : bool, default: False
                If `False` produces boxes with the Line2D artist. Otherwise,
                boxes and drawn with Patch artists.

            labels : sequence, optional
                Labels for each dataset (one per dataset).

            manage_ticks : bool, default: True
                If True, the tick locations and labels will be adjusted to match
                the boxplot positions.

            autorange : bool, default: False
                When `True` and the data are distributed such that the 25th and
                75th percentiles are equal, *whis* is set to (0, 100) such
                that the whisker ends are at the minimum and maximum of the data.

            meanline : bool, default: False
                If `True` (and *showmeans* is `True`), will try to render the
                mean as a line spanning the full width of the box according to
                *meanprops* (see below).  Not recommended if *shownotches* is also
                True.  Otherwise, means will be shown as points.

            zorder : float, default: ``Line2D.zorder = 2``
                The zorder of the boxplot.
        """
        self._params.ydata[name]['kwargs'] = kwargs
        self._params.ydata[name]['xdata'] = xdata
        self._params.ydata[name]['plotmode'] = 'boxplot'
        self._params.ydata[name]['plotfunc'] = self._execute_plot_boxplot
        return self
    
    def _execute_plot_boxplot(self, fig, ax, i, j):
        axplot = ax.boxplot(j['xdata'], **j['kwargs'])