import datetime
import multiprocessing
from collections import defaultdict

from linora.utils._config import Config

__all__ = ['Schedulers']

class Schedulers():
    """Time job task manager."""
    def __init__(self, logger=None, verbose=0, config_file=None):
        """
        Args:
            logger: Logger object, linora.utils.Logger() class.
            verbose: Verbosity mode, 0 (silent), 1 (verbose).
            config_file: job task config file, if .py file.
                         example: .py file name is schedulers_config.py, contain a dict, 
                         config = {'hhh':{'mode':'every_minute', 'time':50, 'function':function, 'args':[], 'kwargs':{}}}
        """
        self.config = dict()
        self.params = Config()
        self.params.verbose = verbose
        if logger is None:
            self.params.verbose = 0
        self.params.logger = logger
        self.params.config_file = config_file
        manager = multiprocessing.Manager()
        self.params.tracker_dict = manager.dict()
        self.params.runner_dict = defaultdict()
    
    def every_minute(self, time, function, args=None, kwargs=None, name=None):
        """Run task manager every minute.
        
        Args:
            time: int or str, Within range 0~59, 27 means the task will start at the 27th second every minute. 
            function: task function.
            args: list, function args.
            kwargs: dict, function kwargs.
            name: task name, if None, is function name.
        """
        if args is None:
            args = list()
        if kwargs is None:
            kwargs = dict()
        if name is None:
            name = function.__name__+(f'_{len(self.config)+1}' if function.__name__ in self.config else '')
        self.config[name] = {'mode':'every_minute', 'time':int(time), 'function':function, 'args':args, 
                             'kwargs':kwargs, 'execute_num':0, 'runner':(function, args, kwargs, name),
                             'time_init':datetime.datetime.now()}
        self.params.tracker_dict[name] = dict()
    
    def every_hour(self, time, function, args=None, kwargs=None, name=None):
        """Run task manager every hour.
        
        Args:
            time: str, '30:27' means the task will start at the 30th minute and 27 seconds every hour.
            function: task function.
            args: list, function args.
            kwargs: dict, function kwargs.
            name: task name, if None, is function name.
        """
        if args is None:
            args = list()
        if kwargs is None:
            kwargs = dict()
        if name is None:
            name = function.__name__+(f'_{len(self.config)+1}' if function.__name__ in self.config else '')
        self.config[name] = {'mode':'every_hour', 'time':time, 'function':function, 'args':args, 
                             'kwargs':kwargs, 'execute_num':0, 'runner':(function, args, kwargs, name),
                             'time_init':datetime.datetime.now()}
        self.params.tracker_dict[name] = dict()
    
    def every_day(self, time, function, args=None, kwargs=None, name=None):
        """Run task manager every day.
        
        Args:
            time: str, '08:30:27' means that the task will start at the 30th minute and 27 seconds of the 8th hour every day.
            function: task function.
            args: list, function args.
            kwargs: dict, function kwargs.
            name: task name, if None, is function name.
        """
        if args is None:
            args = list()
        if kwargs is None:
            kwargs = dict()
        if name is None:
            name = function.__name__+(f'_{len(self.config)+1}' if function.__name__ in self.config else '')
        self.config[name] = {'mode':'every_day', 'time':time, 'function':function, 'args':args, 
                             'kwargs':kwargs, 'execute_num':0, 'runner':(function, args, kwargs, name),
                             'time_init':datetime.datetime.now()}
        self.params.tracker_dict[name] = dict()
    
    def every_week(self, time, function, args=None, kwargs=None, name=None):
        """Run task manager every week.
        
        Args:
            time: str, '1,3,7:08:30:27' means that every Monday, Wednesday, Sunday, 
                  the 8th hour, 30 minutes and 27 seconds to start the task.
            function: task function.
            args: list, function args.
            kwargs: dict, function kwargs.
            name: task name, if None, is function name.
        """
        if args is None:
            args = list()
        if kwargs is None:
            kwargs = dict()
        if name is None:
            name = function.__name__+(f'_{len(self.config)+1}' if function.__name__ in self.config else '')
        self.config[name] = {'mode':'every_week', 'time':time, 'function':function, 'args':args, 
                             'kwargs':kwargs, 'execute_num':0, 'runner':(function, args, kwargs, name),
                             'time_init':datetime.datetime.now()}
        self.params.tracker_dict[name] = dict()
    
    def every_month(self, time, function, args=None, kwargs=None, name=None):
        """Run task manager every month.
        
        Args:
            time: str, '1,13,27:08:30:27' means that the task will start at the 30th minute and 27 seconds 
                  of the 8th hour on the 1st, 13th, and 27th of each month.
            function: task function.
            args: list, function args.
            kwargs: dict, function kwargs.
            name: task name, if None, is function name.
        """
        if args is None:
            args = list()
        if kwargs is None:
            kwargs = dict()
        if name is None:
            name = function.__name__+(f'_{len(self.config)+1}' if function.__name__ in self.config else '')
        self.config[name] = {'mode':'every_month', 'time':time, 'function':function, 'args':args, 
                             'kwargs':kwargs, 'execute_num':0, 'runner':(function, args, kwargs, name),
                             'time_init':datetime.datetime.now()}
        self.params.tracker_dict[name] = dict()
    
    def _reset_time(self, name, time_now=None):
        if time_now is None:
            time_now = datetime.datetime.now()
        if self.config[name]['mode']=='every_minute':
            seconds = 60+self.config[name]['time']-time_now.second
        elif self.config[name]['mode']=='every_hour':
            split = self.config[name]['time'].split(':')
            seconds = int(datetime.datetime(time_now.year, time_now.month, time_now.day, time_now.hour, 
                                            int(split[0]), int(split[1]), time_now.microsecond
                                           ).timestamp()-time_now.timestamp())
            if seconds<40:
                seconds = 3600+seconds
        elif self.config[name]['mode']=='every_day':
            split = self.config[name]['time'].split(':')
            seconds = int(datetime.datetime(time_now.year, time_now.month, time_now.day, int(split[0]), 
                                            int(split[1]), int(split[2]), time_now.microsecond
                                           ).timestamp()-time_now.timestamp())
            if seconds<40:
                seconds = 86400+seconds
        elif self.config[name]['mode']=='every_week':
            split = self.config[name]['time'].split(':')
            seconds = [int(i)-time_now.weekday()-1 for i in split[0].split(',')]
            seconds = [(datetime.datetime(time_now.year, time_now.month, time_now.day, int(split[1]), 
                                      int(split[2]), int(split[3]), time_now.microsecond
                                     )+datetime.timedelta(days=7+i if i<0 else i)
                    ).timestamp()-time_now.timestamp() 
                    for i in seconds]
            if max(seconds)<40:
                seconds= 604800+min(seconds)
            else:
                seconds = [i for i in sorted(seconds) if i>=40][0]
        elif self.config[name]['mode']=='every_month':
            split = self.config[name]['time'].split(':')

            seconds = [datetime.datetime(time_now.year, time_now.month, int(i), int(split[1]), 
                                      int(split[2]), int(split[3]), time_now.microsecond
                                     ) for i in split[0].split(',')]
            if time_now.month<12:
                seconds += [datetime.datetime(time_now.year, time_now.month+1, int(i), int(split[1]), 
                                      int(split[2]), int(split[3]), time_now.microsecond
                                     ) for i in split[0].split(',')]
            else:
                seconds += [datetime.datetime(time_now.year+1, 1, int(i), int(split[1]), 
                                      int(split[2]), int(split[3]), time_now.microsecond
                                     ) for i in split[0].split(',')]
            seconds = [i.timestamp()-time_now.timestamp() for i in seconds]
            seconds = [i for i in sorted(seconds) if i>=40][0]

        self.config[name]['time_next'] = time_now+datetime.timedelta(seconds=seconds)
        self.config[name]['time_record'] = time_now
    
    def run(self):
        time_now = datetime.datetime.now()
        if self.params.config_file is not None:
            config = Config(file_py=self.params.config_file)
            for name in config.config:
                self.config[name] = config.config[name]
                self.config[name]['execute_num'] = 0
                self.config[name]['runner'] = (self.config[name]['function'], self.config[name]['args'], 
                                               self.config[name]['kwargs'], name)
                self.config[name]['time_init'] = time_now
                
        for name in self.config:
            self._reset_time(name, time_now)
            if self.params.verbose:
                self.params.logger.info(f'New task {name} has been added.', write_file=True)
        
        while True:
            time_now = datetime.datetime.now()
            
            for name in self.config:
                if self.config[name]['time_next']>time_now:
                    self.config[name]['time_record'] = time_now
                else:
                    self._start(self.config[name]['runner'])                    
                    self._reset_time(name, time_now)
                    self.config[name]['execute_num'] += 1
            
            try:
                if self.params.config_file is not None:
                    config = Config(file_py=self.params.config_file)
                    for name in config.config:
                        if name not in self.config:
                            self.config[name] = config.config[name]
                            self.config[name]['execute_num'] = 0
                            self.config[name]['time_init'] = time_now
                            self._reset_time(name, time_now)
                            if self.params.verbose:
                                self.params.logger.info(f'New task {name} has been added.', write_file=True)
                        for i,j in config.config[name].items():
                            self.config[name][i] = j
                        self.config[name]['runner'] = (self.config[name]['function'], self.config[name]['args'], 
                                                       self.config[name]['kwargs'], name)
            except Exception as msg:
                if self.params.verbose:
                    self.params.logger.info(str(msg), write_file=True)
    
    def _run(self, runner):
        """Runs function runner """
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
        msg = (f'task {runner[3]} started_time: {str(started)[:19]}, finished_time: {str(finished)[:19]}, '+
               f'execution_time: {round((finished - started).total_seconds(),4)}s, got_error: {got_error}, error: {error}.')
        if self.params.verbose:
            self.params.logger.info(msg, write_file=True)
        elif self.params.logger is not None:
            self.params.logger.write(msg)
            
    def _start(self, runner):
        """Starts runner process """
        self.params.runner_dict[runner[3]] = multiprocessing.Process(target=self._run, args=(runner,))
        self.params.runner_dict[runner[3]].daemon = True
        self.params.runner_dict[runner[3]].start()
