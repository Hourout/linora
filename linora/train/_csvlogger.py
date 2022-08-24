import time
from collections import defaultdict

from linora.gfile._gfile import exists, remove
from linora.utils._config import Config

__all__ = ['CSVLogger']


class CSVLogger():
    """Callback that streams epoch results to a CSV file.
    
    Args:
        filename: Filename of the CSV file, e.g. 'run/log.csv'.
        sep: String used to separate elements in the CSV file.
        append: Boolean. True: append if file exists (useful for continuing training). False: overwrite existing file.
        wait_num: int, default 10, How many batches to store at intervals.
    """
    def __init__(self, filename, sep=',', append=False, wait_num=10):
        self._params = Config()
        self._params.filename = filename
        self._params.sep = sep
        self._params.append = append
        self._params.wait_num = wait_num
        self._params.history = defaultdict()
        if exists(filename):
            if not append:
                remove(filename)
        self._params.polt_num = 0
        self._params.name = 'CSVLogger'
        
    def _update(self, batch, log):
        """update log.
        
        Args:
            batch: Integer, index of batch.
            log: dict, name and value of loss or metrics;
        """
        self._params.history[self._params.polt_num] = [time.time(), batch, log]
        if self._params.polt_num%self._params.wait_num==0:
            with open(self._params.filename, 'a+') as f:
                for i,j in self._params.history.items():
                    msg = self._params.sep.join([str(r)+':'+str(j[2][r]) for r in j[2]])
                    f.write(f'time:{j[0]}{self._params.sep}batch:{j[1]}{self._params.sep}{msg}\n')
            self._params.history = defaultdict()
        self._params.polt_num += 1
        