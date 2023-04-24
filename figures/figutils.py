"""Bundles together some common imports, functions and settings for figure generation / plotting"""

import os
import subprocess
from os.path import join
from pathlib import Path

import drjit as dr
import mitsuba as mi

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as path_effects

import seaborn as sns
import numpy as np
import json

mi.set_variant('cuda_ad_rgb', 'scalar_rgb')

_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
OUTPUT_DIR = join(Path(__file__).parents[3], 'output')
FIG_OUTPUT_DIR = join(_SCRIPT_DIR, 'output')
del _SCRIPT_DIR

TEXT_WIDTH = 5.90666
# TODO: get the proper value
PAGE_WIDTH = 2*TEXT_WIDTH

DEFAULT_FONTSIZE = 10  # Font size used by captions
DEFAULT_FONTSIZE_SMALL = 8

MPL_STYLE = {
    "text.usetex": True,
    "text.color": 'black',
    "font.size": DEFAULT_FONTSIZE,
    "axes.titlesize": DEFAULT_FONTSIZE,
    "axes.labelsize": DEFAULT_FONTSIZE_SMALL,
    "xtick.labelsize": DEFAULT_FONTSIZE_SMALL - 2,
    "ytick.labelsize": DEFAULT_FONTSIZE_SMALL - 2,
    "legend.fontsize": DEFAULT_FONTSIZE_SMALL,
    "figure.titlesize": DEFAULT_FONTSIZE,
    "text.latex.preamble": r"""\usepackage{libertine}
                               \usepackage[libertine]{newtxmath}
                               \usepackage{amsmath}
                               \usepackage{amsfonts}
                               \usepackage{bm}
                               \usepackage{bbm}""",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    'axes.edgecolor': 'black',
    'axes.linewidth': 0.4,
    'xtick.major.size': 0.5,
    'xtick.major.width': 0.5,
    'xtick.minor.size': 0.25,
    'xtick.minor.width': 0.5,

    'ytick.major.size': 0.5,
    'ytick.major.width': 0.5,
    'ytick.minor.size': 0.25,
    'ytick.minor.width': 0.5,

    'lines.linewidth': 0.75,
    'patch.linewidth': 0.5,

    'grid.linewidth': 0.5,

    'axes.titley': -0.18,

    'figure.dpi': 120, # Controls the display size of the figure in the notebook (we overwrite this when saving)
}


def __init():
    # Override any style changes by VSCode
    matplotlib.style.use('default')
    matplotlib.rcParams.update(MPL_STYLE)
    sns.set()
    matplotlib.rcParams.update(MPL_STYLE)

__init()


def save_fig(fig_name, fig_sub_dir='', dpi=300, pad_inches=0.005, bbox_inches='tight', compress=True):
    if fig_sub_dir == '':
        output_dir = os.path.join(FIG_OUTPUT_DIR, fig_name)
    else:
        output_dir = os.path.join(FIG_OUTPUT_DIR, fig_sub_dir, fig_name)
    os.makedirs(output_dir, exist_ok=True)
    fn = join(output_dir, fig_name + '.pdf')
    orig_fn = fn
    if compress:
        fn = fn.replace('.pdf', '_uc.pdf')
    plt.savefig(fn, format='pdf', dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
    if compress:
        gs = f"gs -o {orig_fn} -dQUIET -f -dNOPAUSE -dBATCH "
        gs += "-sDEVICE=pdfwrite -dPDFSETTINGS=/prepress -dCompatibilityLevel=1.6 "
        gs += f"-dDownsampleColorImages=false -DownsampleGrayImages=false {fn}"
        subprocess.call(gs, shell=True)
    return orig_fn

def gridspec_aspect(n_rows, n_cols, w, h, wspace=0, hspace=0):
    if isinstance(w, int) or isinstance(w, float):
        Ws = n_cols * w
    elif isinstance(w, list) or isinstance(w, tuple):
        Ws = sum(w)

    if isinstance(h, int):
        Hs = n_rows * h
    elif isinstance(h, list) or isinstance(h, tuple):
        Hs = sum(h)

    w_spacing = wspace * Ws / n_cols
    h_spacing = hspace * Hs / n_rows

    return (Ws + (n_cols - 1) * w_spacing) / (Hs + (n_rows - 1) * h_spacing)

def disable_ticks(ax):
    ax.axes.get_xaxis().set_ticklabels([])
    ax.axes.get_yaxis().set_ticklabels([])
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])


def disable_border(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


def set_spine_width(ax, w):
    for _, s in ax.spines.items():
        s.set_linewidth(w)


def set_aspect(ax, aspect):
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * aspect)


def apply_color_map(data, cmap='coolwarm', vmin=None, vmax=None):
    from matplotlib import cm

    data = np.array(data)
    if vmin is None:
        vmin = np.min(data)
    if vmax is None:
        vmax = np.max(data)
    return getattr(cm, cmap)(plt.Normalize(vmin, vmax)(data))[..., :3]


def merge_pdfs(fn1, fn2, out_fn):
    """Merges two PDF files"""
    from PyPDF2 import PdfReader, PdfWriter
    reader_base = PdfReader(fn1)
    page_base = reader_base.pages[0]
    reader = PdfReader(fn2)
    page_box = reader.pages[0]
    page_base.merge_page(page_box)
    writer = PdfWriter()
    writer.add_page(page_base)
    with open(out_fn, 'wb') as fp:
        writer.write(fp)


def repeat_img(img, times):
    return np.repeat(np.repeat(img, times, axis=1), times, axis=0)


def time_to_string(duration):
    duration = round(duration)
    m, s = divmod(duration, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    result = ''
    if d > 0:
        result += f'{d}d '
    if h > 0:
        result += f'{h}h '
    if m > 0:
        result += f'{m}m '
    result += f'{s}s'
    return result


def read_img(fn, exposure=0, tonemap=True, background_color=None,
             handle_inexistant_file=False):
    if handle_inexistant_file and not os.path.isfile(fn):
        return np.ones((256, 256, 3)) * 0.3
    bmp = mi.Bitmap(fn)
    if tonemap:
        if background_color is not None:
            img = np.array(bmp.convert(mi.Bitmap.PixelFormat.RGBA, mi.Struct.Type.Float32, False))
            background_color = np.array(background_color).ravel()[None, None, :]
            # img = img[:, :, :3] * img[..., -1][..., None] + (1.0 - img[..., -1][..., None]) * background_color
            img = img[:, :, :3] + (1.0 - img[..., -1][..., None]) * background_color
        else:
            img = np.array(bmp.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, False))
        img = img * 2 ** exposure

        return np.clip(np.array(mi.Bitmap(img).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, True)), 0, 1)
    else:
        return np.array(bmp)


def tonemap(img):
    return np.clip(np.array(mi.Bitmap(img).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, True)), 0, 1)


def identify_axes(ax_dict, fontsize=48):
    """
    Helper to identify the Axes in the examples below.

    Draws the label in a large font in the center of the Axes.

    Parameters
    ----------
    ax_dict : dict[str, Axes]
        Mapping between the title / label and the Axes.
    fontsize : int, optional
        How big the label should be.
    """
    kw = dict(ha="center", va="center", fontsize=fontsize, color="darkgrey")
    for k, ax in ax_dict.items():
        ax.text(0.5, 0.5, k, transform=ax.transAxes, **kw)
