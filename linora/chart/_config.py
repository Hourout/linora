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
                 'pink':['#eeb8c3', '#f0a1a8', '#ec8aa4', '#ec7696', '#ef82a0', '#de7897', '#e77c8e',
                         '#ed9db2', '#f07c82', '#ea517f', '#eb507e', '#ce5777',
                         '#eeaa9c', '#f17666', '#ed4845', '#f25a47', '#f03f24', '#ed3333', '#f43e06', 
                        '#eb261a', '#e60012', '#4d2517', '#9e2a22', '#77171c', ],
                 'blue':['#b4d9ef', '#78bbdd', '#6f94cd', '#63bbd0', '#158bb8', '#2775b6', '#0d67bf', 
                         '#1661ab', '#11659a', '#4656b7', '#2e317c', '#144a74', ],
                 'purple':['#c8adc4', '#c08eaf', '#c06f98', '#ad6598', '#806d9e', '#983680', '#815c94', 
                           '#813c85', '#9b1e64', '#7e2065', '#8b2671', '#7e1671', ],
                 'yellow':['#f7de98', '#e1d384', '#f8d86a', '#fed71a', '#e2d849', '#fccb16', '#feba07', 
                           '#edc300', '#d3a237', '#ffa60f', '#c57917', '#ff9900', ],
                 'grey':['#cfccc9', '#d4c4b7', '#b2bbbe', '#bdaead', '#b89485', '#9a8878', '#856d72', 
                         '#617172', '#8b614d', '#624941', '#475164', '#363433', ]
                }

