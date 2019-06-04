from linora import chart
from linora import metrics
from linora import param_search
from linora import feature_column
from linora import feature_selection
from linora import sample
from linora import sample_splits
from linora import text
from linora import image

__version__ = '0.7.0'
__author__ = 'JinQing Lee'

def _hello():
    print("""
------------------------------------------------------------------------------------
      Linora
--------------------
      Version      : --  {}  --
      Author       : --  {}  --
      License      : Apache-2.0
      Homepage     : https://github.com/Hourout/linora
      Description  : Simple and efficient tools for data mining and data analysis.
------------------------------------------------------------------------------------""".format(__version__, __author__))
