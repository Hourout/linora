import os
import sys
import time


__all__ = ['Logger']

class Params:
    log_colors = {
        'bebug': 'purple',
        'info': 'green',
        'warning': 'yellow',
        'error': 'red',
        'critical': 'bold_red',
        'train': 'cyan',
        'test': 'blue'
    }
    log_colors_code = {
        'purple': '\033[35m{}\033[0m',
        'green':'\033[32m{}\033[0m',
        'yellow': '\033[33m{}\033[0m',
        'red': '\033[31m{}\033[0m',
        'black': '\033[30m{}\033[0m',
        'cyan': '\033[36m{}\033[0m',
        'blue': '\033[34m{}\033[0m'
    }
    log_level = {
        'debug':10,
        'info':20,
        'train':21,
        'test':22,
        'warning':30,
        'error':40,
        'critical':50
    }
    log_level_default = None
    log_name = None
    log_file = ''
    file = None
    write_file_mode = 0
    overwrite=False
    last_msg = ''
    
class Logger():
    def __init__(self, name="", level="INFO", log_file='', write_file_mode=0, overwrite=False, stream='stderr'):
        """Logs are printed on the console and stored in files.
        
        Args:
            name: log subject name.
            level: log printing level.
            log_file: log file path.
            write_file_mode: 1 is fast mode, 0 is slow mode, default 0.
            overwrite: whether overwrite log file.
            stream: stderr or stdout, a file-like object (stream), defaults to the current sys.stderr.
        """
        self.params = Params()
        self.params.log_name = 'root' if name=="" else name
        if str.lower(level) in self.params.log_level:
            self.params.log_level_default = str.lower(level)
        else:
            raise ValueError("`level` must in param dict `Logger.params.log_level`. ")
        
        self.params.write_file_mode = write_file_mode
        self.params.overwrite = overwrite
        self.params.stream = stream
        self.params.time = time.time()
        self.params.time_start = self.params.time
        self.update_log_file(log_file)
            
    def log(self, level, msg, write_file, enter, time_mode, close):
        """Logs are printed on the console and stored in files.
        
        Args:
            level: log printing level.
            msg: log printing message.
            write_file: whether to control log write to the log file.
            enter: whether to wrap print.
            time_mode: 1 is total time, 0 is interval time.
            close: whether to end the entire log notification.
        """
        if self.params.log_level[str.lower(level)]<self.params.log_level[self.params.log_level_default]:
            return 
        
        time_now = time.time()
        end = time_now-self.params.time_start if time_mode else time_now-self.params.time
        self.params.time = time_now
        if end < 60:
            msg = f'[{end:.2f} sec]: ' + msg
        elif end < 3600:
            msg = "[%d min %.2f s]: " % divmod(end, 60) + msg
        else:
            msg = f"[{end // 3600:.0f} hour %d min %.0f s]: " % divmod(end % 3600, 60) + msg
        msg = (f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time_now))}]" +
               f" [{self.params.log_name}] [{str.upper(level)}] " + msg)
        if write_file:
            self.write(msg)

        if len(self.params.last_msg)>len(msg):
            msg = msg+' '*(len(self.params.last_msg)-len(msg))
        self.params.last_msg = msg
        try:
            msg = self.params.log_colors_code[self.params.log_colors[str.lower(level)]].format(msg)
        except:
            msg = f'\033[30m{msg}\033[0m'
        print(msg, end='\n' if enter else '\r', file=sys.stderr if self.params.stream=='stderr' else sys.stdout)
        
        if close:
            self.close()
        
    def write(self, msg):
        """Logs write to the log file.
        
        Args:
            msg: log message.
        """
        try:
            self.params.file.write(msg + '\n')
        except:
            try:
                with open(self.params.log_file, 'a+') as f:
                    f.write(msg + '\n')
            except:
                pass
    
    def debug(self, msg, write_file=False, enter=True, time_mode=0, close=False):
        """Logs are printed on the console and stored in files.
        
        Args:
            msg: log printing message.
            write_file: whether to control log write to the log file.
            enter: whether to wrap print.
            time_mode: 1 is total time, 0 is interval time.
            close: whether to end the entire log notification.
        """
        self.log("DEBUG", msg, write_file, enter, time_mode, close)

    def info(self, msg, write_file=False, enter=True, time_mode=0, close=False):
        """Logs are printed on the console and stored in files.
        
        Args:
            msg: log printing message.
            write_file: whether to control log write to the log file.
            enter: whether to wrap print.
            time_mode: 1 is total time, 0 is interval time.
            close: whether to end the entire log notification.
        """
        self.log("INFO", msg, write_file, enter, time_mode, close)

    def warning(self, msg, write_file=False, enter=True, time_mode=0, close=False):
        """Logs are printed on the console and stored in files.
        
        Args:
            msg: log printing message.
            write_file: whether to control log write to the log file.
            enter: whether to wrap print.
            time_mode: 1 is total time, 0 is interval time.
            close: whether to end the entire log notification.
        """
        self.log("WARNING", msg, write_file, enter, time_mode, close)

    def error(self, msg, write_file=False, enter=True, time_mode=0, close=False):
        """Logs are printed on the console and stored in files.
        
        Args:
            msg: log printing message.
            write_file: whether to control log write to the log file.
            enter: whether to wrap print.
            time_mode: 1 is total time, 0 is interval time.
            close: whether to end the entire log notification.
        """
        self.log("ERROR", msg, write_file, enter, time_mode, close)

    def critical(self, msg, write_file=False, enter=True, time_mode=0, close=False):
        """Logs are printed on the console and stored in files.
        
        Args:
            msg: log printing message.
            write_file: whether to control log write to the log file.
            enter: whether to wrap print.
            time_mode: 1 is total time, 0 is interval time.
            close: whether to end the entire log notification.
        """
        self.log("CRITICAL", msg, write_file, enter, time_mode, close)
        
    def train(self, msg, write_file=False, enter=True, time_mode=0, close=False):
        """Logs are printed on the console and stored in files.
        
        Args:
            msg: log printing message.
            write_file: whether to control log write to the log file.
            enter: whether to wrap print.
            time_mode: 1 is total time, 0 is interval time.
            close: whether to end the entire log notification.
        """
        self.log("TRAIN", msg, write_file, enter, time_mode, close)
    
    def test(self, msg, write_file=False, enter=True, time_mode=0, close=False):
        """Logs are printed on the console and stored in files.
        
        Args:
            msg: log printing message.
            write_file: whether to control log write to the log file.
            enter: whether to wrap print.
            time_mode: 1 is total time, 0 is interval time.
            close: whether to end the entire log notification.
        """
        self.log("TEST", msg, write_file, enter, time_mode, close)

    def update_log_file(self, log_file):
        """Update log file path.
        
        Args:
            log_file: log file path.
        """
        real_path = os.path.realpath(os.path.dirname(log_file))
        if log_file=='':
            pass
        elif real_path != os.path.abspath(log_file):
            if not os.path.exists(real_path):
                os.makedirs(real_path)
            self.params.log_file = log_file
            if self.params.write_file_mode:
                self.close()
                mode = 'w' if self.params.overwrite else 'a+'
                self.params.file = open(self.params.log_file, mode)
        else:
            raise ValueError("`log_file` must is file name.")
    
    def close(self):
        """Close log ssesion."""
        if self.params.file is not None:
            self.params.file.close()
