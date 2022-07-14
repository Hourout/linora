import matplotlib.pyplot as plt

from linora.utils._config import Config

__all__ = ['Grid']


class Grid():
    def __init__(self, nrows, ncols, left=None, bottom=None, right=None, top=None,
                 wspace=None, hspace=None, width_ratios=None, height_ratios=None):
        """A grid layout to place subplots within a figure.
        
        Args:
            nrows, ncols : int
                The number of rows and columns of the grid.

            left, right, top, bottom : float, optional
                Extent of the subplots as a fraction of figure width or height.
                Left cannot be larger than right, and bottom cannot be larger than
                top. If not given, the values will be inferred from a figure or
                rcParams at draw time.

            wspace : float, optional
                The amount of width reserved for space between subplots,
                expressed as a fraction of the average axis width.
                If not given, the values will be inferred from a figure or
                rcParams when necessary.

            hspace : float, optional
                The amount of height reserved for space between subplots,
                expressed as a fraction of the average axis height.
                If not given, the values will be inferred from a figure or
                rcParams when necessary.

            width_ratios : array-like of length *ncols*, optional
                Defines the relative widths of the columns. Each column gets a
                relative width of ``width_ratios[i] / sum(width_ratios)``.
                If not given, all columns will have the same width.

            height_ratios : array-like of length *nrows*, optional
                Defines the relative heights of the rows. Each column gets a
                relative height of ``height_ratios[i] / sum(height_ratios)``.
                If not given, all rows will have the same height.
        """
        self._grid = Config()
        self._grid.figure = {'figsize':(10, 6)}
        self._grid.grid = {'nrows':nrows, 'ncols':ncols, 'left':left, 'bottom':bottom, 
                           'right':right, 'top':top, 'wspace':wspace, 'hspace':hspace, 
                           'width_ratios':width_ratios, 'height_ratios':height_ratios}
        self._grid.grid_id = dict()
        
    def add_plot(self, grid_id, plot):
        self._grid.grid_id[len(self._grid.grid_id)] = {'grid_id':grid_id, 'plot':plot}        
        return self
        
    def render(self):
        return self._execute().show()
    
    def set_figure(width=10, height=6, dpi=None, facecolor=None, edgecolor=None, frameon=True, clear=False):
        """Create a new figure, or activate an existing figure.
        
        Args:
            width: float, figure size width in inches.
            height: float, figure size height in inches.
            dpi: float, The resolution of the figure in dots-per-inch.
            facecolor: color, The background color.
            edgecolor: color, The border color.
            frameon: bool, default: True, If False, suppress drawing the figure frame.
            clear: bool, default: False, If True and the figure already exists, then it is cleared.
        """
        kwargs = {'figsize':(width, height), 'dpi':dpi, 'facecolor':facecolor, 
                  'edgecolor':edgecolor, 'frameon':frameon, 'clear':clear}
        self._grid.figure.update(kwargs)
        return self
    
    def _execute(self):
        fig = plt.figure(**self._grid.figure)
        grid = plt.GridSpec(**self._grid.grid)
        for i, grid_id in self._grid.grid_id.items():
            with plt.style.context(grid_id['plot']._params.theme):
                ax = fig.add_subplot(grid_id['grid_id'])
                ax = grid_id['plot']._execute_ax(ax)
        return fig