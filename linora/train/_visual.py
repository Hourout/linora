from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

from linora.chart._config import Options
from linora.utils._config import Config

__all__ = ['Visual']


class Visual():
    """
    Args:
        ncols : int, default 2, The number of sub graphs that the width of metrics
                 visualiztion image to accommodate at most;
        iter_num : int, default None, Pre-specify the maximum value of x-axis in each
                  sub-picture to indicate the maximum number of batch or epoch training;
        mode : int, default 1, 1 means the x-axis name is 'batch', 0 means the x-axis name is 'epoch';
        wait_num : int, default 1, Indicates how many batches or epochs are drawn
                  each time a graph is drawn;
        figsize : tuple, default None，Represents the customize image size;
        valid_fmt : str, default "val_{}",The string preceding the underscore is used to
                   instruction the training and validation is displayed together in the
                   same sub graph. The training indicator is not required to have a prefix.
                   The validation indicator prefix is 'val' in the "val_{}";
    """
    def __init__(self, ncols=2, iter_num=None, mode=1, wait_num=1, figsize=None, valid_fmt="test_{}"):
        self._params = Config()
        self._params.ncols = ncols
        self._params.iter_num = iter_num
        self._params.mode = mode
        self._params.wait_num = wait_num
        self._params.figsize = figsize
        self._params.valid_fmt = valid_fmt
        self._params.logs = defaultdict(list)
        self._params.xlabel = {0:'epoch', 1:'batch'}
        self._params.polt_num = 0
        self._params.frames = []
        key = np.random.choice(list(Options.color), size=len(Options.color), replace=False)
        t = {i:np.random.choice(Options.color[i], size=len(Options.color[i]), replace=False) for i in key}
        self._params.color = sorted([[r+k*7, j, i] for r, i in enumerate(key) for k, j in enumerate(t[i])])

    def update(self, log):
        """update log
        
        Args:
            log: dict, name and value of loss or metrics;
        """
        self._params.metrics = list(filter(lambda x: self._params.valid_fmt.split('_')[0] not in x.lower(), log))
        if self._params.figsize is None:
            self._params.figsize = (self._params.ncols*6, ((len(self._params.metrics)+1)//self._params.ncols+1)*4)
        for metric in log:
            self._params.logs[metric] += [log[metric]]
        self._params.polt_num += 1
            
    def draw(self):
        """plot metrics."""
        if self._params.polt_num%self._params.wait_num==0:
            clear_output(wait=True)
            with plt.style.context('ggplot'):
                self._params.figure = plt.figure(figsize=self._params.figsize)
                for metric_id, metric in enumerate(self._params.metrics):
                    plt.subplot((len(self._params.metrics)+1)//self._params.ncols+1, self._params.ncols, metric_id+1)
                    if self._params.iter_num is not None:
                        plt.xlim(1, self._params.iter_num)
                    plt.plot(range(1, len(self._params.logs[metric])+1), self._params.logs[metric], label="train",
                             color=self._params.color[metric_id*2][1])
                    if self._params.valid_fmt.format(metric) in self._params.logs:
                        plt.plot(range(1, len(self._params.logs[metric])+1),
                                 self._params.logs[self._params.valid_fmt.format(metric)],
                                 label=self._params.valid_fmt.split('_')[0], color=self._params.color[metric_id*2+1][1])
                    plt.title(metric)
                    plt.xlabel(self._params.xlabel[self._params.mode])
                    plt.legend(loc='best')
            plt.tight_layout()
            plt.show()
    
    def save(self, image_path, **kwargs):
        """save plot.
        
        Args:
            image_path: str, train end save last image.
        """
        self._params.figure.savefig(image_path, **kwargs)
