import matplotlib.pyplot as plt

from linora.utils._config import Config

__all__ = ['Options']

_marker = Config(**{
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
_xloc = Config(**{'left':'left', 'center':'center', 'right':'right'})
_fontsize = Config(**{
    'small_xx':'xx_small', 
    'small_x':'x-small', 
    'small':'small', 
    'medium':'medium', 
    'large':'large', 
    'large_x':'x-large', 
    'large_xx':'xx-large'})
_fontweight = Config(**{
    'book': 'book', 
    'normal': 'normal',
    'bold': 'bold',
    'demi': 'demi',
    'semibold': 'semibold',
    'roman': 'roman',
    'black': 'black',
    'extra bold': 'extra bold',
    'light': 'light',
    'regular': 'regular',
    'demibold': 'demibold',
    'medium': 'medium',
    'ultralight': 'ultralight',
    'heavy': 'heavy'})
_dash_capstyle = Config(**{'butt': 'butt', 'projecting': 'projecting', 'round': 'round'})
_dash_joinstyle = Config(**{'miter': 'miter', 'round': 'round', 'bevel': 'bevel'})

Options = Config()
Options.theme = Config(**{i.replace('-', '_'):i for i in plt.style.available})
Options.axis = Config(**{'on':'on', 'off':'off', 'equal':'equal', 'scaled':'scaled', 
                         'tight':'tight', 'auto':'auto', 'image':'image', 'square':'square'})
Options.label = Config()
Options.label.xloc = _xloc
Options.label.yloc = {'bottom':'bottom', 'center':'center', 'top':'top'}

Options.title = Config()
Options.title.titleloc = _xloc
Options.title.titlesize = _fontsize
# Options.title.titlecolor = 'auto', 
# Options.title.titlepad = 
# Options.title.titley = None, 
Options.title.titleweight = _fontweight

Options.line = Config()
Options.line.linestyle = Config(**{'solid':'-', 'dashed':'--', 'dashdot':'-.', 'dotted':':'})
Options.line.drawstyle = Config(**{'steps':'steps', 'steps_pre':'steps-pre', 'steps_mid':'steps-mid', 'steps_post':'steps-post'})
Options.line.dash_capstyle = _dash_capstyle
Options.line.dash_joinstyle = _dash_joinstyle
Options.line.fillstyle = Config(**{'full': 'full', 'left': 'left', 'right': 'right', 'bottom': 'bottom', 'top': 'top'})
Options.line.marker = _marker
Options.line.solid_capstyle = _dash_capstyle
Options.line.solid_joinstyle = _dash_joinstyle

