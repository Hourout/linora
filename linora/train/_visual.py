import json
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

from linora.chart._config import Options
from linora.utils._config import Config

__all__ = ['Visual']


class Visual():
    """Real-time dynamic training visualization callback.
    
    Args:
        ncols : int, default 2, The number of sub graphs that the width of metrics
                 visualiztion image to accommodate at most;
        iter_num : int, default None, Pre-specify the maximum value of x-axis in each
                  sub-picture to indicate the maximum number of batch or epoch training;
        mode : int, default 1, 1 means the x-axis name is 'batch', 0 means the x-axis name is 'epoch';
        wait_num : int, default 5, Indicates how many batches or epochs are drawn
                  each time a graph is drawn;
        figsize : tuple, default None，Represents the customize image size;
        valid_fmt : str, default "val_{}",The string preceding the underscore is used to
                   instruction the training and validation is displayed together in the
                   same sub graph. The training indicator is not required to have a prefix.
                   The validation indicator prefix is 'val' in the "val_{}";
        avg_num: int, default 1, mertics move aveage.
    """
    def __init__(self, ncols=2, iter_num=None, mode=1, wait_num=5, figsize=None, valid_fmt="test_{}", avg_num=1):
        self._params = Config()
        self._params.ncols = ncols
        self._params.iter_num = iter_num
        self._params.mode = mode
        self._params.wait_num = wait_num
        self._params.figsize = figsize
        self._params.valid_fmt = valid_fmt
        self._params.xlabel = {0:'epoch', 1:'batch'}
        self._params.polt_num = 0
        self._params.figure = None
        self._params.avg_num = avg_num if avg_num>=0 else 1
        self.history = defaultdict(lambda: defaultdict(list))
        key = np.random.choice(list(Options.color), size=len(Options.color), replace=False)
        t = {i:np.random.choice(Options.color[i], size=len(Options.color[i]), replace=False) for i in key}
        self._params.color = sorted([[r+k*7, j, i] for r, i in enumerate(key) for k, j in enumerate(t[i])])
        self._params.name = 'Visual'

    def _update(self, batch, log):
        """update log
        
        Args:
            batch: Integer, index of batch.
            log: dict, name and value of loss or metrics;
        """
        self._params.metrics = list(filter(lambda x: self._params.valid_fmt.split('_')[0] not in x.lower(), log))
        for metric in log:
            self.history[metric]['values'] += [log[metric]]
            self.history[metric]['index'] += [batch]
            self.history[metric]['move_avg'] += [np.mean(self.history[metric]['values'][-self._params.avg_num:]).item()]
        self._params.polt_num += 1
        
    def draw(self):
        """plot metrics."""
        if self._params.polt_num%self._params.wait_num==0:
            clear_output(wait=True)
            self._excute()
            plt.show()
    
    def save_image(self, image_path, **kwargs):
        """save plot.
        
        Args:
            image_path: str, train end save last image.
        """
        if self._params.figure is None:
            self._excute()
        self._params.figure.savefig(image_path, **kwargs)

    def save_logs(self, filename):
        """save logs.
        
        Args:
            filename: Filename of the json file, e.g. 'run/log.json'.
        """
        with open(filename, 'w') as f:
            json.dump(self.history, f)
    
    def _excute(self):
        with plt.style.context('ggplot'):
            if self._params.figsize is None:
                figsize = (self._params.ncols*6, ((len(self._params.metrics)+1)//self._params.ncols+1)*4)
                self._params.figure = plt.figure(figsize=figsize)
            else:
                self._params.figure = plt.figure(figsize=self._params.figsize)
            for metric_id, metric in enumerate(self._params.metrics):
                plt.subplot((len(self._params.metrics)+1)//self._params.ncols+1, self._params.ncols, metric_id+1)
                if self._params.iter_num is not None:
                    plt.xlim(1, self._params.iter_num)
                plt.plot(self.history[metric]['index'], self.history[metric]['move_avg'], label="train",
                         color=self._params.color[metric_id*2][1])
                if self._params.valid_fmt.format(metric) in self.history:
                    plt.plot(self.history[self._params.valid_fmt.format(metric)]['index'],
                             self.history[self._params.valid_fmt.format(metric)]['move_avg'],
                             label=self._params.valid_fmt.split('_')[0], color=self._params.color[metric_id*2+1][1])
                plt.title(metric)
                plt.xlabel(self._params.xlabel[self._params.mode])
                plt.legend(loc='best')
        plt.tight_layout()
        