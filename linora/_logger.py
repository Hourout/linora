import os
import time
import logging
import colorlog

__all__ = ['Logger']

class Params:
    log_colors = {
            'DEBUG': 'purple',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
            'TRAIN': 'cyan',
            'TEST': 'blue'
        }
    log_level = {
            'DEBUG':10,
            'INFO':20,
            'TRAIN':21,
            'TEST':22,
            'WARNING':30,
            'ERROR':40,
            'CRITICAL':50
        }
    log_level_default = None
    log_name = None
    log_file = ''
    file = None
    write_stream = None
    write_file = None
    message_stream = None
    write_file_mode = 0
    overwrite=False
    time = time.time()
    
class Logger():
    def __init__(self, name="", level="INFO", log_file='', write_file_mode=0, overwrite=False,
                 message_stream='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'):
        """Logs are printed on the console and stored in files.
        
        Args:
            name: log subject name.
            level: log printing level.
            log_file: log file path.
            write_file_mode: 1 is fast mode, 0 is slow mode, default 0.
            overwrite: whether overwrite log file.
            message_stream: printing to the console formatter.
        """
        self.params = Params()
        self.params.log_name = name
        if level in self.params.log_level:
            self.params.log_level_default = level
        else:
            raise ValueError("`level` must in param dict `log_level`. ")
        
        self.params.message_stream = '%(log_color)s' + message_stream
        self.params.write_file_mode = write_file_mode
        self.params.overwrite = overwrite
        self.update_log_file(log_file)
        
        logging.addLevelName(self.params.log_level['TRAIN'], 'TRAIN')
        logging.addLevelName(self.params.log_level['TEST'], 'TEST')
        self.params.logger = logging.getLogger(self.params.log_name)
        self.params.logger.setLevel(self.params.log_level[self.params.log_level_default])
        formatter_sh = colorlog.ColoredFormatter(
            self.params.message_stream, datefmt='%Y-%m-%d %H:%M:%S', log_colors=self.params.log_colors)
        self.params.sh = logging.StreamHandler()
        self.params.sh.setLevel(self.params.log_level[self.params.log_level_default])
        self.params.sh.setFormatter(formatter_sh)
            
    def log(self, level, msg, write_file):
        """Logs are printed on the console and stored in files.
        
        Args:
            level: log printing level.
            msg: log printing message.
            write_file: whether to control log write to the log file.
        """
        if write_file:
            self.write(msg)
        
        end = time.time()-self.params.time
        if end < 60:
            msg = f'[{end:.2f} sec]: ' + msg
        elif end < 3600:
            msg = "[%d min %.2f s]: " % divmod(end, 60) + msg
        else:
            msg = f"[{end // 3600:.0f} hour %d min %.0f s]: " % divmod(end % 3600, 60) + msg

        self.params.logger.addHandler(self.params.sh)
        self.params.logger.log(self.params.log_level[level], msg)
        self.params.logger.removeHandler(self.params.sh)
        self.params.time = time.time()
    
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
    
    def debug(self, msg, write_file=False):
        """Logs are printed on the console and stored in files.
        
        Args:
            msg: log printing message.
            write_file: whether to control log write to the log file.
        """
        self.log("DEBUG", msg, write_file)

    def info(self, msg, write_file=False):
        """Logs are printed on the console and stored in files.
        
        Args:
            msg: log printing message.
            write_file: whether to control log write to the log file.
        """
        self.log("INFO", msg, write_file)

    def warning(self, msg, write_file=False):
        """Logs are printed on the console and stored in files.
        
        Args:
            msg: log printing message.
            write_file: whether to control log write to the log file.
        """
        self.log("WARNING", msg, write_file)

    def error(self, msg, write_file=False):
        """Logs are printed on the console and stored in files.
        
        Args:
            msg: log printing message.
            write_file: whether to control log write to the log file.
        """
        self.log("ERROR", msg, write_file)

    def critical(self, msg, write_file=False):
        """Logs are printed on the console and stored in files.
        
        Args:
            msg: log printing message.
            write_file: whether to control log write to the log file.
        """
        self.log("CRITICAL", msg, write_file)
        
    def train(self, msg, write_file=False):
        """Logs are printed on the console and stored in files.
        
        Args:
            msg: log printing message.
            write_file: whether to control log write to the log file.
        """
        self.log("TRAIN", msg, write_file)
    
    def test(self, msg, write_file=False):
        """Logs are printed on the console and stored in files.
        
        Args:
            msg: log printing message.
            write_file: whether to control log write to the log file.
        """
        self.log("TEST", msg, write_file)

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
