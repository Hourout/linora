import matplotlib.pyplot as plt
from linora.utils._config import Config

__all__ = ['Options']


Options = Config()
Options.theme = Config(**{i.replace('-', '_'):i for i in plt.style.available})

Options.linestyle = Config(**{
    'solid': '-', 
    'dashed': '--', 
    'dashdot': '-.', 
    'dotted': ':'})
Options.drawstyle = Config(**{
    'steps': 'steps',
    'steps_pre': 'steps-pre',
    'steps_mid': 'steps-mid',
    'steps_post': 'steps-post'})
Options.dash_capstyle = Config(**{'butt': 'butt', 'projecting': 'projecting', 'round': 'round'})
Options.dash_joinstyle = Config(**{'miter': 'miter', 'round': 'round', 'bevel': 'bevel'})
Options.fillstyle = Config(**{'full': 'full', 'left': 'left', 'right': 'right', 'bottom': 'bottom', 'top': 'top'})
Options.marker = Config(**{
    'point': '.',
    'pixel': ',',
    'circle': 'o',
    'triangle_down': 'v',
    'triangle_up': '^',
    'triangle_left': '<',
    'triangle_right': '>',
    'tri_down': '1',
    'tri_up': '2',
    'tri_left': '3',
    'tri_right': '4',
    'octagon': '8',
    'square': 's',
    'pentagon': 'p',
    'star': '*',
    'hexagon1': 'h',
    'hexagon2': 'H',
    'plus': '+',
    'x': 'x',
    'diamond': 'D',
    'thin_diamond': 'd',
    'vline': '|',
    'hline': '_',
    'plus_filled': 'P',
    'x_filled': 'X',
    'tickleft': 0,
    'tickright': 1,
    'tickup': 2,
    'tickdown': 3,
    'caretleft': 4,
    'caretright': 5,
    'caretup': 6,
    'caretdown': 7,
    'caretleftbase': 8,
    'caretrightbase': 9,
    'caretupbase': 10,
    'caretdownbase': 11})
Options.solid_capstyle = Config(**{'butt': 'butt', 'projecting': 'projecting', 'round': 'round'})
Options.solid_joinstyle = Config(**{'miter': 'miter', 'round': 'round', 'bevel': 'bevel'})