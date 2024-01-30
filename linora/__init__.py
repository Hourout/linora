from linora import audio
from linora import chart
from linora import credit
from linora import data
from linora import feature_column
from linora import feature_selection
from linora import gfile
from linora import image
from linora import metrics
from linora import parallel
from linora import param_search
from linora import sample
from linora import server
from linora import text
from linora import train
from linora import utils
from linora import vedio

__version__ = '2.0.0rc1'
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
      Docpage      : https://www.yuque.com/jinqing-ps0ax/linora
      Description  : Simple and efficient tools for data mining and data analysis.
      Language     : python 3.7｜3.8|3.9|3.10|3.11
      Email        : hourout@163.com
      Requires     : ['pandas>=1.3.5', 
                      'Pillow>=9.5.0',
                      'joblib>=1.3.2',
                      'requests>=2.28.0',
                      'rarfile',
                      'av']
      Time         : 2018-2024
------------------------------------------------------------------------------------""")
