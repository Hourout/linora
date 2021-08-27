import time
import datetime
import threading
from collections import defaultdict

__all__ = ['ThreadLoom']

class Params:
    pass
    
class ThreadLoom():
    """ThreadLoom class: executes runners using threading."""
    def __init__(self, max_runner, mode=1):
        """
        max_runner: int, The total number of runners that are allowed to be running at any given time.
        mode: 1 is setDaemon(True) and 0 is join().
        """
        self.params = Params()
        self.params.max_runner = max_runner
        self.params.mode = mode
        self.params.runners = list()
        self.params.started = list()
        self.params.time_pause = 0.1
        self.params.tracker_dict = defaultdict()
        self.params.runner_dict = defaultdict()

    def add_function(self, func, args=None, kwargs=None, key=None):
        """ Adds function in the Loom
        Args:
            func (reference): reference to the function
            args (list): function args
            kwargs (dict): function kwargs
            key (str): key to store the function output in dictionary
        """
        if args is None:
            args = list()
        if kwargs is None:
            kwargs = dict()
        if key is None:
            key = len(self.params.runners)
        self.params.runners.append((func, args, kwargs, key))
        
    def add_work(self, works):
        """ Adds work to the loom
        Args:
            works (list): list of works [(func, args, kwargs, key), (func2, args2, kwargs2), ...]
        """
        for work in works:
            if len(work) > 4 or len(work) == 0:
                raise ValueError('Need 1 to 4 values to unpack')
            args = work[1] if len(work) > 1 else None
            kwargs = work[2] if len(work) > 2 else None
            key = work[3] if len(work) == 4 else None
            self.add_function(work[0], args, kwargs, key)

    def execute(self):
        """ Executes runners and returns output dictionary containing runner output, error,
        started time, finished time and execution time of the given runner. runner key or the
        order in which function was added is the key to get the tracker dictionary.
        Returns:
            dict: output dict
            Examples:
                {
                    "runner1 key/order" : {
                            "output": runner output,
                            "error": runner errors,
                            "started_time": datetime.now() time stamp when runner was started
                            "finished_time": datetime.now() time stamp when runner was completed,
                            "execution_time: total execution time in seconds",
                            "got_error": boolean
                        },
                    "runner2 key/order" : {
                            "output": runner output,
                            "error": runner errors,
                            "started_time": datetime.now() time stamp when runner was started
                            "finished_time": datetime.now() time stamp when runner completed,
                            "execution_time: total execution time in seconds",
                            "got_error": boolean
                        }
                }
        """
        while self.params.runners:
            runner = self.params.runners.pop(0)
            self._start(runner)
            self.params.started.append(runner)
            while self._get_active_runner_count() >= self.params.max_runner:
                time.sleep(self.params.time_pause)
        while self._get_active_runner_count():
            time.sleep(self.params.time_pause)
        output = self.params.tracker_dict
        self.params.tracker_dict = defaultdict()
        self.params.runner_dict = defaultdict()
        return output
    
    def _get_active_runner_count(self):
        """ Returns the total number of runners running at the present time """
        count = 0
        for runner in self.params.started:
            if (self.params.runner_dict[runner[3]] and self.params.runner_dict[runner[3]].is_alive()
                or not self.params.tracker_dict[runner[3]]):
                    count += 1
        if count == 0:
            self.params.started = list()
        return count
    
    def _run(self, runner):
        """ Runs function runner """
        output, error, got_error = None, None, False
        started = datetime.datetime.now()
        try:
            output = runner[0](*runner[1], **runner[2])
        except Exception as e:
            got_error = True
            error = str(e)
        finally:
            finished = datetime.datetime.now()
            self.params.tracker_dict[runner[3]] = {
                "output": output,
                "started_time": started,
                "finished_time": finished,
                "execution_time": (finished - started).total_seconds(),
                "got_error": got_error,
                "error": error}
    
    def _start(self, runner):
        """ Starts runner process """
        self.params.runner_dict[runner[3]] = threading.Thread(target=self._run, args=(runner,))
        if self.params.mode:
            self.params.runner_dict[runner[3]].setDaemon(True)
        self.params.runner_dict[runner[3]].start()
        if not self.params.mode:
            self.params.runner_dict[runner[3]].join()
    
