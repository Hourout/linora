class Csd():
    def add_csd(self, name, xdata, ydata, **kwargs):
        """Plot the cross-spectral density.

        The cross spectral density :math:`P_{xy}` by Welch's average
        periodogram method.  The vectors *x* and *y* are divided into
        *NFFT* length segments.  Each segment is detrended by function
        *detrend* and windowed by function *window*.  *noverlap* gives
        the length of the overlap between segments.  The product of
        the direct FFTs of *x* and *y* are averaged over each segment
        to compute :math:`P_{xy}`, with a scaling to correct for power
        loss due to windowing.

        If len(*x*) < *NFFT* or len(*y*) < *NFFT*, they will be zero
        padded to *NFFT*.

        Args:
            name: data name.
            xdata: x-axis data.
            ydata: y-axis data.
            Fs : float, default: 2
                The sampling frequency (samples per time unit).  It is used to calculate
                the Fourier frequencies, *freqs*, in cycles per time unit.

            window : callable or ndarray, default: `.window_hanning`
                A function or a vector of length *NFFT*.  To create window vectors see
                `.window_hanning`, `.window_none`, `numpy.blackman`, `numpy.hamming`,
                `numpy.bartlett`, `scipy.signal`, `scipy.signal.get_window`, etc.  If a
                function is passed as the argument, it must take a data segment as an
                argument and return the windowed version of the segment.

            sides : {'default', 'onesided', 'twosided'}, optional
                Which sides of the spectrum to return. 'default' is one-sided for real
                data and two-sided for complex data. 'onesided' forces the return of a
                one-sided spectrum, while 'twosided' forces two-sided.

            pad_to : int, optional
                The number of points to which the data segment is padded when performing
                the FFT.  This can be different from *NFFT*, which specifies the number
                of data points used.  While not increasing the actual resolution of the
                spectrum (the minimum distance between resolvable peaks), this can give
                more points in the plot, allowing for more detail. This corresponds to
                the *n* parameter in the call to fft(). The default is None, which sets
                *pad_to* equal to *NFFT*

            NFFT : int, default: 256
                The number of data points used in each block for the FFT.  A power 2 is
                most efficient.  This should *NOT* be used to get zero padding, or the
                scaling of the result will be incorrect; use *pad_to* for this instead.

            detrend : {'none', 'mean', 'linear'} or callable, default: 'none'
                The function applied to each segment before fft-ing, designed to remove
                the mean or linear trend.  Unlike in MATLAB, where the *detrend* parameter
                is a vector, in Matplotlib is it a function.  The :mod:`~matplotlib.mlab`
                module defines `.detrend_none`, `.detrend_mean`, and `.detrend_linear`,
                but you can use a custom function as well.  You can also use a string to
                choose one of the functions: 'none' calls `.detrend_none`. 'mean' calls
                `.detrend_mean`. 'linear' calls `.detrend_linear`.

            scale_by_freq : bool, default: True
                Whether the resulting density values should be scaled by the scaling
                frequency, which gives density in units of Hz^-1.  This allows for
                integration over the returned frequency values.  The default is True for
                MATLAB compatibility.

            noverlap : int, default: 0 (no overlap)
                The number of points of overlap between segments.

            Fc : int, default: 0
                The center frequency of *x*, which offsets the x extents of the
                plot to reflect the frequency range used when a signal is acquired
                and then filtered and downsampled to baseband.

            return_line : bool, default: False
                Whether to include the line object plotted in the returned values.
        """
        if 'color' not in kwargs:
            kwargs['color'] = self._params.color.pop(0)[1]
        elif isinstance(kwargs['color'], dict):
            kwargs['color'] = kwargs.pop('color')['mode']
        kwargs['label'] = name
        
        self._params.ydata[name]['kwargs'] = kwargs
        self._params.ydata[name]['xdata'] = xdata
        self._params.ydata[name]['ydata'] = ydata
        self._params.ydata[name]['plotmode'] = 'csd'
        self._params.ydata[name]['plotfunc'] = self._execute_plot_csd
        return self
    
    def _execute_plot_csd(self, fig, ax, i, j):
        ax_plot = ax.csd(j['xdata'], j['ydata'], **j['kwargs'])
