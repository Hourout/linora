import matplotlib.pyplot as plt

from linora.utils._config import Config

__all__ = ['Options']

Options = Config()
Options.cmap = Config(**{'viridis':'viridis', 'jet':'jet', 'cubehelix':'cubehelix', 'rdbu':'RdBu', 'binary':'binary'})
Options.dash_capstyle = Config(**{'butt': 'butt', 'projecting': 'projecting', 'round': 'round'})
Options.dash_joinstyle = Config(**{'miter': 'miter', 'round': 'round', 'bevel': 'bevel'})
Options.linelink = Config(**{'steps':'steps', 'steps_pre':'steps-pre', 'steps_mid':'steps-mid', 'steps_post':'steps-post'})
Options.linestyle = Config(**{'solid':'-', 'dashed':'--', 'dashdot':'-.', 'dotted':':'})
Options.fontsize = Config(**{
    'small_xx':'xx_small', 
    'small_x':'x-small', 
    'small':'small', 
    'medium':'medium', 
    'large':'large', 
    'large_x':'x-large', 
    'large_xx':'xx-large'})
Options.fontweight = Config(**{
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
Options.mark = Config(**{
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
Options.markfill = Config(**{'full': 'full', 'left': 'left', 'right': 'right', 'bottom': 'bottom', 'top': 'top'})
Options.solid_capstyle = Options.dash_capstyle
Options.solid_joinstyle = Options.dash_joinstyle
Options.xloc = Config(**{'left':'left', 'center':'center', 'right':'right'})


Options.theme = Config(**{i.replace('-', '_'):i for i in plt.style.available})
Options.axis = Config(**{'on':'on', 'off':'off', 'equal':'equal', 'scaled':'scaled', 
                         'tight':'tight', 'auto':'auto', 'image':'image', 'square':'square'})

Options.color = {'green':['#9eccab', '#8cc269', '#68b88e', '#5dbe8a', '#5bae23', '#41b349', '#20894d',
                          '#1a6840', '#69a794', '#2c9678', '#12aa9c', '#57c3c2'],
                 'pink':['#eeb8c3', '#f0ala8', '#ec8aa4', '#ec7696', '#ef82a0', '#de7897', '#e77c8e',
                         '#ed9db2', '#f07c82', '#ea517f', '#eb507e', '#ce5777',]
                }

