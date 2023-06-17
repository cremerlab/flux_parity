import matplotlib.pyplot 
import matplotlib
import seaborn as sns
import bokeh.plotting 
import bokeh.io 
import bokeh.palettes
import bokeh.themes
from bokeh.models import * 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.offsetbox import AnchoredText
import seaborn as sns


def load_markercolors():
    """
    Returns a dictionary mapping sources of the E. coli data with standard colors 
    and glyphs. This ensures constant marking of data across plots.
    """
    colors, _ = get_colors()
    mapper = {
        'Bremer & Dennis, 2008': {'m':'X', 'm_bokeh':'circle_dot'},
        'Brunschede et al., 1977': {'m':'s', 'm_bokeh':'square'},
        'Büke et al., 2022': {'m':'X', 'm_bokeh':'hex'},
        'Lazzarini et al., 1971' : {'m': 'd'},
        'Sokawa et al., 1975' : {'m':'D'},
        'Ryals et al., 1981': {'m':'>'},
        'Baracchini et al., 1988' : {'m': 'h'},
        'Buckstein et al., 2008' : {'m': 's'},
        'Imholz et al., 2020' : {'m':'o'},
        'Dai et al., 2016': {'m':'o',  'm_bokeh': 'circle'},
        'Dong et al., 1996': {'m': 's', 'm_bokeh': 'circle_dot'},
        'Forchhammer & Lindahl, 1971': {'m': 'v', 'm_bokeh': 'inverted_triangle'},
        'Li et al., 2014': {'m':'d', 'm_bokeh':'diamond'},
        'Schmidt et al., 2016': {'m':'8', 'm_bokeh':'hex'},
        'Scott et al., 2010': {'m':'^', 'm_bokeh': 'square_pin'},
        'Wu et al., 2021': {'m':'<', 'm_bokeh': 'square_dot'},
        'Bremer & Dennis, 1996': {'m':'>', 'm_bokeh': 'circle_cross'},
        'Dalbow & Young, 1975': {'m':'P', 'm_bokeh': 'hex_dot'},
        'Young & Bremer, 1976': {'m':'h', 'm_bokeh': 'triangle_pin'},
        'Skjold et al., 1973' : {'m': '*', 'm_bokeh': 'star'},
        'Dong et al., 1996' : {'m': 'p', 'm_bokeh': 'diamond_dot'},
        'Dong et al., 1995' : {'m':'v', 'm_bokeh': 'triangle_pin'},
        'Bentley et al., 1990': {'m': 'X', 'm_bokeh': 'star'},
        'Erickson et al., 2017': {'m': 'o', 'm_bokeh': 'hex_dot'},
        'Oldewurtle et al., 2021': {'m': 's', 'm_bokeh': 'square_pin'},
        'Mori et al., 2017': {'m': '*', 'm_bokeh': 'hex_dot'},
        'Sloan and Urban, 1976': {'m': 'h', 'm_bokeh': 'star'},
        'Li et al., 2018': {'m':'>', 'm_bokeh': 'triangle_pin'},
        'Korem Kohanim et al., 2018': {'m':'d', 'm_bokeh': 'diamond'},
        'Panlilio et al., 2021': {'m': 'p', 'm_bokeh':'diamond_dot'},
        'Basan et al., 2015' : {'m': '8', 'm_bokeh': 'circle'},
        'You et al., 2013' : {'m': 'h', 'm_bokeh':'hex_dot'},
        'Hernandez & Bremer, 1993' : {'m':'X', 'm_bokeh': 'diamond_dot'},
        'Hernandez & Bremer, 1990' : {'m':'v', 'm_bokeh': 'diamond_dot'},
        'Farewell & Neidhart, 1998' : {'m': 'h', 'm_bokeh': 'circle_cross'},
        'Kepes & Beguin, 1966' : {'m':'o', 'm_bokeh': 'circle'},
        'Coffman et al., 1971' :  {'m':'s', 'm_bokeh': 'square'},
        'Morris & Hansen, 1973': {'m': '*', 'm_bokeh': 'star'},
        'Schleif et al., 1973' : {'m':'v', 'm_bokeh': 'triangle'},
        'Lacroute & Stent, 1968': {'m':'p', 'm_bokeh':'hex'},
        'Dennis & Bremer, 1974' : {'m': 's', 'm_bokeh': 'square'},
        'Albertson & Nyström, 1994': {'m':'^', 'm_bokeh': 'circle_cross'},
        'Gausing, 1972': {'m':'>', 'm_bokeh':'diamond'},
        'Schleif, 1967': {'m': '<', 'm_bokeh': 'diamond_dot'},
        'Hernandez & Bremer, 1993': {'m': 'v', 'm_bokeh':'star'},
        'Pedersen, 1984': {'m':'X', 'm_bokeh':'triangle_pin'},
        'Zhu & Dai, 2019': {'m': 'p', 'm_bokeh': 'circle_cross'},
        'Si et al., 2017': {'m': 'v', 'm_bokeh': 'hex_dot'},
        'Sarubbi et al., 1988': {'m': '^', 'm_bokeh':'triangle'}
        }
    # Set colors rooted in blue
    cmap = sns.color_palette(f"light:{colors['primary_black']}", 
                             n_colors=len(mapper)).as_hex()
    cmap.reverse()
    counter = 0
    for k, _ in mapper.items():
        mapper[k]['c'] = cmap[counter]
        counter += 1
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
        "lines.markeredgecolor": '#f0f3f7',
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
        "text.color": "#3d3d3d", #"#5b5b5b",

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