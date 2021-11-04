import matplotlib.pyplot 
import matplotlib
import altair as alt
import bokeh.plotting 
import bokeh.io 
import bokeh.palettes
import bokeh.themes
from bokeh.models import * 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.path import Path
from matplotlib.patches import BoxStyle
from matplotlib.offsetbox import AnchoredText
import seaborn as sns


def load_markercolors():
    """
    Returns a dictionary mapping sources of the E. coli data with standard colors 
    and glyphs. This ensures constant marking of data across plots.
    """
    mapper = {
        'Bremer & Dennis, 2008': {'m':'X', 'c':'#556895', 'm_bokeh':'circle_dot'},
        'Brunshede et al., 1977': {'m':'s', 'c':'#343158', 'm_bokeh':'square'},
        'Dai et al., 2016': {'m':'o',  'c':'#4b5583', 'm_bokeh': 'circle'},
        'Forchhammer & Lindahl, 1971': {'m': 'v', 'c':'#9ac5d0', 'm_bokeh': 'inverted_triangle'},
        'Li et al., 2014': {'m':'d', 'c':'#27231f', 'm_bokeh':'diamond'},
        'Schmidt et al., 2016': {'m':'8', 'c':'#88b3c7', 'm_bokeh':'hex'},
        'Scott et al., 2010': {'m':'^', 'c':'#6c8fb2', 'm_bokeh': 'square_pin'},
        'Wu et al., 2021': {'m':'<', 'c':'#add4d9', 'm_bokeh': 'square_dot'},
        'Bremer & Dennis, 1996': {'m':'>', 'c':'#5f7ba4', 'm_bokeh': 'circle_cross'},
        'Dalbow & Young, 1975': {'m':'P', 'c':'#c3e2e3', 'm_bokeh': 'hex_dot'},
        'Young & Bremer, 1976': {'m':'h', 'c':'#3f426e', 'm_bokeh': 'triangle_pin'},
        'Skjold et al., 1973' : {'m': '*', 'c': '#6c8fb2', 'm_bokeh': 'star'},
        'Dong et al., 1996' : {'m': 'D', 'c':'#556895', 'm_bokeh': 'diamond_dot'}
        }
    return mapper

def get_colors(all_palettes=False):
    """
    Generates a dictionary of standard colors and returns a sequential color
    palette.

    Parameters
    ----------
    all_palettes : bool
        If True, lists of `dark`, `primary`, and `light` palettes will be returned. If
        False, only the `primary` palette will be returned. 
    """
    # Define the colors
    colors = {
        'dark_black': '#2b2b2a',
        'black': '#3d3d3d',
        'primary_black': '#4c4b4c',
        'light_black': '#8c8c8c',
        'pale_black': '#afafaf',
        'dark_blue': '#154577',
        'blue': '#005da2',
        'primary_blue': '#3373ba',
        'light_blue': '#5fa6db',
        'pale_blue': '#8ec1e8',
        'dark_green': '#356835',
        'green': '#488d48',
        'primary_green': '#5cb75b',
        'light_green': '#99d097',
        'pale_green': '#b8ddb6',
        'dark_red': '#79302e',
        'red': '#a3433f',
        'primary_red': '#d8534f',
        'light_red': '#e89290',
        'pale_red': '#eeb3b0',
        'dark_gold': '#84622c',
        'gold': '#b1843e',
        'primary_gold': '#f0ad4d',
        'light_gold': '#f7cd8e',
        'pale_gold': '#f8dab0',
        'dark_purple': '#43355d',
        'purple': '#5d4a7e',
        'primary_purple': '#8066ad',
        'light_purple': '#a897c5',
        'pale_purple': '#c2b6d6' 
        }

    # Generate the sequential color palettes.
    keys = ['black', 'blue', 'green', 'red', 'purple', 'gold']
    dark_palette = [colors[f'dark_{k}'] for k in keys]
    primary_palette = [colors[f'primary_{k}'] for k in keys]
    light_palette = [colors[f'light_{k}'] for k in keys]

    # Determine what to return. 
    if all_palettes:
        palette = [dark_palette, primary_palette, light_palette]
    else:
        palette = primary_palette

    return [colors, palette]



def matplotlib_style(return_colors=True, return_palette=True, **kwargs):
    """
    Assigns the plotting style for matplotlib generated figures. 
    
    Parameters
    ----------
    return_colors : bool
        If True, a dictionary of the colors is returned. Default is True.
    return_palette: bool
        If True, a sequential color palette is returned. Default is True.
    """
    # Define the matplotlib styles.
    rc = {
        # Axes formatting
        "axes.facecolor": "#f0f3f7",
        "axes.edgecolor": "#ffffff", #5b5b5b",
        "axes.labelcolor": "#5b5b5b",
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.axisbelow": True,
        "axes.linewidth": 0.15,
        "axes.grid": True,

        # Formatting of lines and points. 
        "lines.linewidth": 0.5,
        "lines.dash_capstyle": "butt",
        "patch.linewidth": 0.25,
        "lines.markeredgecolor": '#ffffff',
        "lines.markeredgewidth": 0.5,

        # Grid formatting
        "grid.linestyle": '-',
        "grid.linewidth": 0.5,
        "grid.color": "#FFFFFF",

        # Title formatting
        "axes.titlesize": 8,
        "axes.titleweight": 700,
        "axes.titlepad": 3,
        "axes.titlelocation": "left",

        # Axes label formatting. 
        "axes.labelpad": 0,
        "axes.labelweight": 700,
        "xaxis.labellocation": "center",
        "yaxis.labellocation": "center",
        "axes.labelsize": 8,
        "axes.xmargin": 0.03,
        "axes.ymargin": 0.03,

        # Legend formatting
        "legend.fontsize": 6,
        "legend.labelspacing": 0.25,
        "legend.title_fontsize": 6,
        "legend.frameon": True,
        "legend.edgecolor": "#5b5b5b",

        # Tick formatting
        "xtick.color": "#5b5b5b",
        "ytick.color": "#5b5b5b",
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.major.width": 0.25,
        "ytick.major.width": 0.25,
        "xtick.major.pad": 2,
        "ytick.major.pad": 2,
        "xtick.minor.size": 0,
        "ytick.minor.size": 0,

        # General Font styling
        "font.family": "sans-serif",
        "font.family": "Nunito",
        "font.weight": 400, # Weight of all fonts unless overriden.
        "font.style": "normal",
        "text.color": "#5b5b5b",

        # Higher-order things
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "figure.dpi": 300,
        "errorbar.capsize": 1,
        "savefig.bbox": "tight",
        "mathtext.default": "regular",
    }
    matplotlib.style.use(rc)

    # Load the colors and palettes. 
    colors, palette = get_colors(**kwargs)
    sns.set_palette(palette)

    # Determine what, if anything should be returned
    out = []
    if return_colors == True:
        out.append(colors)
    if return_palette == True:
        out.append(palette)
    
    if len(out) == 1:
        return out[0]
    else:
        return out


def titlebox(
    ax, text, color, bgcolor=None, size=8, boxsize=0.1, pad=0.05, loc=10, **kwargs
):
    """Sets a colored box about the title with the width of the plot"""
    boxsize=str(boxsize * 100)  + '%'
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size=boxsize, pad=pad)
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.spines["top"].set_visible(False)
    cax.spines["right"].set_visible(False)
    cax.spines["bottom"].set_visible(False)
    cax.spines["left"].set_visible(False)

    matplotlib.pyplot.setp(cax.spines.values(), color=color)
    if bgcolor != None:
        cax.set_facecolor(bgcolor)
    else:
        cax.set_facecolor("white")
    at = AnchoredText(text, loc=loc, frameon=False, prop=dict(size=size, color=color))
    cax.add_artist(at)


def altair_style(return_colors=True, return_palette=True, pub=False, **kwargs):
    """
    Assigns the plotting style for matplotlib generated figures. 
    
    Parameters
    ----------
    return_colors : bool
        If True, a dictionary of the colors is returned. Default is True.
    return_palette: bool
        If True, a sequential color palette is returned. Default is True.
    pub: bool    
        If True, sizes and scales will be adjusted for print formatting
    """
    colors, palette = get_colors(**kwargs)
    if len(palette) == 3:
        primary_palette = palette[2]
    else:
        primary_palette = palette
    if pub:
        fontsize_primary = 8
        fontsize_secondary = 6
        width_primary = 360
        height_primary = 240
        ps = 15
        lw = 0.5
        ew = 0.25
    else:
        fontsize_primary = 14
        fontsize_secondary = 10 
        width_primary = 600
        height_primary = 400
        ps=80
        lw=2
        ew=0.75

    def _theme():
        return {
            'config': {
                'background': 'white',
                    'group': { 
                    'fill': 'white', 
                    },
                'view': {
                    'strokeWidth': 0,
                    'height': height_primary,
                    'width': width_primary,
                    'fill': '#f0f3f7', #ebeef2', #f8f8fa'
                    },
                'point': {
                    'size': ps,
                    'filled': True,
                    'opacity': 1,
                    'strokeWidth': ew,
                    'stroke': '#FFFFFF'
                    },    
                'square': {
                    'size': ps,
                    'filled': True,
                    'opacity': 1,
                    'strokeWidth': ew,
                    'stroke': '#FFFFFF'
                    },      
                'circle': {
                    'size': ps,
                    'filled': True,
                    'opacity': 0.75,
                    'strokeWidth': ew,
                    'stroke': '#f0f3f7'
                    },  
                'line': {
                    'size': lw,
                },
                'axis': {
                    'domainColor': '#ffffff', #5b5b5b',
                    'domainWidth': 0.5,
                    'labelColor': '#5b5b5b',
                    'labelFontSize': fontsize_secondary,
                    'labelFont': 'Arial',
                    'titleFont': 'Arial',
                    'titleFontWeight': 700,
                    'titleFontSize':fontsize_primary,
                    'titleColor': '#4b4b4b',
                    # 'titleAnchorX': 'end',
                    'grid': True,
                    'gridColor': '#ffffff', #c1c1c1',
                    'gridWidth': 0.5,
                    'ticks': False,
                },
                'range': {
                    'category': primary_palette
                },
                'legend': {
                    'labelFontSize': fontsize_secondary,
                    'labelFont': 'Arial',
                    'titleFont': 'Arial',
                    'titleFontSize': fontsize_primary,
                    'titleFontWeight': 700,
                    'titleFontColor': '#44b4b4b',
                    'symbolSize': ps,
                },
                'title' : { 
                    'font': 'Arial',
                    'fontWeight': 700,
                    'fontSize': fontsize_primary,
                    'fontColor': '#4b4b4b',
                    # 'anchor': 'start',
                }
                  }
                }

    alt.themes.register('personal', _theme)# enable the newly registered theme
    alt.themes.enable('personal')
    # Determine what, if anything should be returned
    out = []
    if return_colors == True:
        out.append(colors)
    if return_palette == True:
        out.append(palette)
    
    if len(out) == 1:
        return out[0]
    else:
        return out


def bokeh_style(return_colors=True, return_palette=True):
    theme_json = {
        "attrs": {
            "Figure": {"background_fill_color": "#f0f3f7",},
            "Axis": {
                "axis_line_color": None,
                "major_tick_line_color": None,
                "minor_tick_line_color": None,
            },
            "Legend": {
                "border_line_color": "slategray",
                "background_fill_color": "#f0f3f7",
                "border_line_width": 0.75,
                "background_fill_alpha": 0.75,
            },
            "Grid": {"grid_line_color": "#FFFFFF", "grid_line_width": 0.75,},
            "Text": {
                "text_font_style": "regular",
                "text_font_size": "12pt",
                "text_font": "Nunito"
            },
            "Title": {
                "background_fill_color": "#FFFFFF",
                "text_color": "#3c3c3c",
                "align": "left",
                'text_font_style': 'normal',
                'text_font_size': "10pt",
                "offset": 5 
            },
        }
    }

    colors, palette = get_colors()
    theme = bokeh.themes.Theme(json=theme_json)
    bokeh.io.curdoc().theme = theme
    out = []
    if return_colors:
        out.append(colors)  
    if return_palette:  
       out.append(palette) 
    if return_colors | return_palette:
        return out


def load_js(fname, args):
    """
    Given external javascript file names and arguments, load a bokeh CustomJS
    object
    
    Parameters
    ----------
    fname: str or list of str
        The file name of the external javascript file. If the desired javascript
        exists in multiple external files, they can be provided as a list of
        strings.
    args: dict
        The arguments to supply to the custom JS callback. 
    
    Returns
    -------
    cb : bokeh CustomJS model object
        Returns a bokeh CustomJS model object with the supplied code and
        arguments. This can be directly assigned as callback functions.
    """
    if type(fname) == str:
        with open(fname) as f:
            js = f.read() 
    elif type(fname) == list:
        js = ''
        for _fname in fname:
            with open(_fname) as f:
                js += f.read()

    cb = CustomJS(code=js, args=args)
    return cb