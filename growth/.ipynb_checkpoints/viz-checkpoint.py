import matplotlib.pyplot 
import matplotlib
import altair as alt
import seaborn as sns

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
        'dark_black': '#2b2ba',
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
        "grid.linestyle": ':',
        "grid.linewidth": 0.25,
        "grid.color": "#c1c1c1",

        # Title formatting
        "axes.titlesize": 6,
        "axes.titleweight": 700,
        "axes.titlepad": 3,
        "axes.titlelocation": "left",

        # Axes label formatting. 
        "axes.labelpad": 0,
        "axes.labelweight": 700,
        "xaxis.labellocation": "right",
        "yaxis.labellocation": "top",
        "axes.labelsize": 6,
        "axes.xmargin": 0.03,
        "axes.ymargin": 0.03,

        # Legend formatting
        "legend.fontsize": 5,
        "legend.labelspacing": 0.25,
        "legend.title_fontsize": 5,
        "legend.frameon": True,
        "legend.edgecolor": "#5b5b5b",

        # Tick formatting
        "xtick.color": "#5b5b5b",
        "ytick.color": "#5b5b5b",
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
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

def altair_style(return_colors=True, return_palette=True, **kwargs):
    """
    Assigns the plotting style for matplotlib generated figures. 
    
    Parameters
    ----------
    return_colors : bool
        If True, a dictionary of the colors is returned. Default is True.
    return_palette: bool
        If True, a sequential color palette is returned. Default is True.
    """
    colors, palette = get_colors(**kwargs)
    if len(palette) == 3:
        primary_palette = palette[2]
    else:
        primary_palette = palette
    def _theme():
        return {
            'config': {
                'background': 'white',
                    'group': { 
                    'fill': 'white', 
                    },
                'view': {
                    'strokeWidth': 0,
                    'height': 300,
                    'width': 400,
                    'fill': '#f0f3f7', #ebeef2', #f8f8fa'
                    },
                'point': {
                    'size': 40,
                    'filled': True,
                    'opacity': 1,
                    'strokeWidth': 0.75,
                    'stroke': '#FFFFFF'
                    },    
                'square': {
                    'size': 40,
                    'filled': True,
                    'opacity': 1,
                    'strokeWidth': 0.75,
                    'stroke': '#FFFFFF'
                    },      
                'circle': {
                    'size': 40,
                    'filled': True,
                    'opacity': 1,
                    'strokeWidth': 0.75,
                    'stroke': '#FFFFFF'
                    },  
                'line': {
                    'size': 2,
                },
                'axis': {
                    'domainColor': '#ffffff', #5b5b5b',
                    'domainWidth': 0.5,
                    'labelColor': '#5b5b5b',
                    'labelFontSize': 10,
                    'labelFont': 'Nunito',
                    'titleFont': 'Nunito',
                    'titleFontWeight': 700,
                    'titleFontSize':14,
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
                    'labelFontSize': 14,
                    'labelFont': 'Nunito',
                    'titleFont': 'Nunito',
                    'titleFontSize': 14,
                    'titleFontWeight': 700,
                    'titleFontColor': '#44b4b4b',
                    'symbolSize': 75,
                },
                'title' : { 
                    'font': 'Nunito',
                    'fontWeight': 700,
                    'fontSize': 14,
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