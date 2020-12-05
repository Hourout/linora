import os
import logging
import colorlog

__all__ = ['Logger']

class Logger():
    def __init__(self, name="", level="INFO", log_file='', write_stream=True, write_file=True,
                 message_stream='[%(asctime)s] [%(name)s] [%(levelname)8s]: %(message)s',
                 message_file='[%(asctime)s] [%(name)s] [%(levelname)8s]: %(message)s',
                ):
        self.log_colors = {
            'DEBUG': 'purple',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
            'TRAIN': 'cyan',
            'TEST': 'blue'
        }
        self.log_level = {
            'DEBUG':10,
            'INFO':20,
            'TRAIN':21,
            'TEST':22,
            'WARNING':30,
            'ERROR':40,
            'CRITICAL':50
        }
        self.log_name = name
        if level in self.log_level:
            self.log_level_default = level
        else:
            raise ValueError("`level` must in param dict `log_level`. ")
        
        if log_file=='':
            self.log_file = log_file
        elif os.path.realpath(os.path.dirname(log_file))!=os.path.abspath(log_file):
            self.log_file = log_file
        else:
            raise ValueError("`log_file` must in file name. ")
        
        self.write_stream = write_stream
        self.write_file = write_file
        self.message_stream = '%(log_color)s'+message_stream
        self.message_file = message_file
        
    def _mylog(self, level, msg, write_stream, write_file):
        write_stream = write_stream if self.write_stream else self.write_stream
        write_file = write_file if self.write_file else self.write_file
        
        if not write_stream and not write_file:
            return
        
        if write_file:
            real_path = os.path.realpath(os.path.dirname(self.log_file))
            if self.log_file=='':
                pass
            elif real_path != os.path.abspath(self.log_file):
                if not os.path.exists(real_path):
                    os.makedirs(real_path)
            else:
                raise ValueError("`log_file` must in file name. ")
        
        logging.addLevelName(self.log_level['TRAIN'], 'TRAIN')
        logging.addLevelName(self.log_level['TEST'], 'TEST')
        logger = logging.getLogger(self.log_name)
        logger.setLevel(self.log_level[self.log_level_default])
        
        if write_stream:
            formatter_sh = colorlog.ColoredFormatter(self.message_stream, log_colors=self.log_colors)
            sh = logging.StreamHandler()
            sh.setLevel(self.log_level[self.log_level_default])
            sh.setFormatter(formatter_sh)
            logger.addHandler(sh)

        if self.log_file!='' and  write_file:
            formatter_fh = logging.Formatter(self.message_file)
            fh = logging.FileHandler(self.log_file, encoding="utf-8")
            fh.setLevel(self.log_level[self.log_level_default])
            fh.setFormatter(formatter_fh)
            logger.addHandler(fh)
        
        logger.log(self.log_level[level], msg)
        
        if write_stream:
            logger.removeHandler(sh)
        if self.log_file!='' and  write_file:
            logger.removeHandler(fh)
            fh.close()

    def debug(self, msg, write_stream=True, write_file=True):
        self._mylog("DEBUG", msg, write_stream, write_file)

    def info(self, msg, write_stream=True, write_file=True):
        self._mylog("INFO", msg, write_stream, write_file)

    def warning(self, msg, write_stream=True, write_file=True):
        self._mylog("WARNING", msg, write_stream, write_file)

    def error(self, msg, write_stream=True, write_file=True):
        self._mylog("ERROR", msg, write_stream, write_file)

    def critical(self, msg, write_stream=True, write_file=True):
        self._mylog("CRITICAL", msg, write_stream, write_file)
        
    def train(self, msg, write_stream=True, write_file=True):
        self._mylog("TRAIN", msg, write_stream, write_file)
    
    def test(self, msg, write_stream=True, write_file=True):
        self._mylog("TEST", msg, write_stream, write_file)
