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
        batch_total : int, default None, Pre-specify the maximum value of x-axis in each
                  sub-picture to indicate the maximum number of batch or epoch training;
        batch_wait : int, default 5, Indicates how many batches or epochs are drawn
                  each time a graph is drawn;
        batch_max_draw: How many data to draw.
        avg_num: int, default 1, mertics move aveage.
        mode : int, default 1, 1 means the x-axis name is 'batch', 0 means the x-axis name is 'epoch';
        figsize : tuple, default None，Represents the customize image size;
        valid_fmt : str, default "val_{}",The string preceding the underscore is used to
                   instruction the training and validation is displayed together in the
                   same sub graph. The training indicator is not required to have a prefix.
                   The validation indicator prefix is 'val' in the "val_{}";
    """
    def __init__(self, ncols=2, batch_total=None, batch_wait=20, batch_max_draw=100, avg_num=5,
                 mode=1, figsize=None, valid_fmt="test_{}"):
        self._params = Config()
        self._params.ncols = ncols
        self._params.mode = mode
        self._params.batch_total = batch_total
        self._params.batch_wait = batch_wait
        self._params.batch_max_draw = batch_max_draw+1
        self._params.figsize = figsize
        self._params.valid_fmt = valid_fmt
        self._params.xlabel = {0:'epoch', 1:'batch'}
        self._params.figure = None
        self._params.avg_num = avg_num if avg_num>=0 else 1
        self._params.metrics = defaultdict()
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
        for i in list(filter(lambda x: self._params.valid_fmt.split('_')[0] not in x.lower(), log)):
            if i not in self._params.metrics:
                self._params.metrics[i] = {'batch_total':self._params.batch_total, 
                                           'batch_max_draw':self._params.batch_max_draw, 'avg_num':self._params.avg_num}
        for metric in log:
            self.history[metric]['values'] += [log[metric]]
            self.history[metric]['index'] += [batch]            
        self._params.batch = batch
        
    def set_draw(self, monitor, batch_total=None, batch_max_draw=100, avg_num=5):
        """Metric configuration to monitor.
        
        Args:
            monitor: Metric to monitor.
            batch_total : int, default None, Pre-specify the maximum value of x-axis in each
                      sub-picture to indicate the maximum number of batch or epoch training;
            batch_max_draw: How many data to draw.
            avg_num: int, default 1, mertics move aveage.
        """
        self._params.metrics[monitor] = {'batch_total':batch_total, 
                                         'batch_max_draw':batch_max_draw, 'avg_num':avg_num if avg_num>=0 else 1}
        
    def draw(self):
        """plot metrics."""
        if self._params.batch%self._params.batch_wait==0:
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
            for metric_id, (metric, params) in enumerate(self._params.metrics.items()):
                if self._params.batch>params['avg_num']:
                    value = np.convolve(self.history[metric]['values'], np.ones(params['avg_num']), mode='valid')/params['avg_num']
                    index = np.array(self.history[metric]['index'][-len(value)-2:])
                else:
                    value = np.array(self.history[metric]['values'])
                    index = np.array(self.history[metric]['index'])
                if len(value)>params['batch_max_draw']:
                    idx = np.linspace(0., len(value)-1, params['batch_max_draw'], dtype=int)[1:]
                else:
                    idx = np.arange(len(value))
                plt.subplot((len(self._params.metrics)+1)//self._params.ncols+1, self._params.ncols, metric_id+1)
                if params['batch_total'] is not None:
                    plt.xlim(1, params['batch_total'])
                plt.plot(index[idx], value[idx], label="train", color=self._params.color[metric_id*2][1])
                if self._params.valid_fmt.format(metric) in self.history:
                    if self._params.batch>params['avg_num']:
                        value = np.convolve(self.history[self._params.valid_fmt.format(metric)]['values'], np.ones(params['avg_num']), mode='valid')/params['avg_num']
                        index = np.array(self.history[self._params.valid_fmt.format(metric)]['index'][-len(value)-2:])
                    else:
                        value = np.array(self.history[self._params.valid_fmt.format(metric)]['values'])
                        index = np.array(self.history[self._params.valid_fmt.format(metric)]['index'])
                    if len(value)>self._params.batch_max_draw:
                        idx = np.linspace(0., len(value)-1, params['batch_max_draw'], dtype=int)[1:]
                    else:
                        idx = np.arange(len(value))
                    plt.plot(index[idx], value[idx], label=self._params.valid_fmt.split('_')[0], color=self._params.color[metric_id*2+1][1])
                plt.title(metric)
                plt.xlabel(self._params.xlabel[self._params.mode])
                plt.legend(loc='best')
                plt.axis('tight')
        plt.tight_layout()
        