"""Nature-journal figure style for this project's plots.
Conventions (from the nature-skills approach): restrained palette, no top/right
spines, Arial-metric sans (Liberation Sans), bold panel labels, thin spines.
Reusable: `import natstyle as ns; ns.set_style()`."""
import matplotlib as mpl

# --- restrained Nature-ish palette ---
BLUE   = "#0F4D92"   # primary / upper-bound
GREEN  = "#5AA469"   # positive region
RED    = "#B64342"   # ACCENT = our result (draws the eye)
ORANGE = "#E1A948"
PURPLE = "#6A51A3"
GRAY   = "#8C8C8C"    # baselines / competitors (muted)
GRAYL  = "#D0D0D0"
INK    = "#1A1A1A"

def set_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Nimbus Sans", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 9, "axes.titlesize": 9, "axes.titleweight": "bold",
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
        "axes.linewidth": 0.8, "axes.edgecolor": INK,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "legend.frameon": False, "axes.grid": False,
        "figure.dpi": 150, "savefig.dpi": 320,
    })

def panel_label(ax, s, dx=-0.15, dy=1.04):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=11, fontweight="bold",
            va="top", ha="left")
