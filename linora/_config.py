import os
import imp
import json


__all__ = ['Config']

def Config(file_py=None, file_dict=None, file_json=None, **kwargs):
    """params management.
    
    Args:
        file_py: a python file about param config.
        file_dict: a dict.
        file_json: a json file about param config.
        **kwargs: One or more parameter information added separately.
    Returns:
        a param config class.
    """
    class config:
        if file_dict is not None:
            for i, j in file_dict.items():
                locals()[i] = j
        if file_json is not None:
            with open(file_json) as _f:
                data = json.load(_f)
            for i, j in data.items():
                locals()[i] = j
            try:
                del data, _f
            except:
                pass
        if file_py is not None:
            filters = ['__builtins__', '__cached__', '__doc__', '__file__', 
                       '__loader__', '__name__', '__package__', '__spec__']
            temp = imp.load_source(os.path.split(file_py)[-1].split('.')[0], file_py)
            for i in dir(temp):
                if i not in filters:
                    locals()[i] = getattr(temp, i)
            try:
                del temp, filters
            except:
                pass
        for i, j in kwargs.items():
            locals()[i] = j
        try:
            del i,j
        except:
            pass
    return config()
