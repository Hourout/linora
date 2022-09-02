from linora import chart
from linora import metrics
from linora import parallel
from linora import param_search
from linora import feature_column
from linora import feature_selection
from linora import sample
from linora import text
from linora import image
from linora import utils
from linora import gfile
from linora import data
from linora import train

__version__ = '1.5.0'
__author__ = 'JinQing Lee'

def _version():
    print(f"""
------------------------------------------------------------------------------------
      Linora
--------------------
      Version      : --  {__version__}  --
      Author       : --  {__author__}  --
      License      : Apache-2.0
      Homepage     : https://github.com/Hourout/linora
      Description  : Simple and efficient tools for data mining and data analysis.
------------------------------------------------------------------------------------""")
