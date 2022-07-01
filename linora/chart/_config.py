import matplotlib.pyplot as plt
from linora.utils._config import Config

__all__ = ['Options']


Options = Config()
Options.theme = Config(**{i.replace('-', '_'):i for i in plt.style.available})