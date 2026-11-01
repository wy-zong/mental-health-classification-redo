"""Run the project's plot_custom15 (src/plot_paper_confusion_matrices.py) for
MAIN_EVAL_AUGV2_K1. Data, colour map, shared 0-1 scale and cell text come from
the project function; only layout/labels are adjusted for the paper figure.

Writes confusion_rownorm_C1-C5_en.{png,pdf} into OUT_DIR.

Usage:  python src/plot_fig5_confusion_en.py ROOT OUT_DIR SCALE [CELL_SCALE]
Paper figure:  python src/plot_fig5_confusion_en.py . runs/NESTED_70_10_20_INCREMENTAL/MAIN_EVAL_AUGV2_K1/figures 1.2 1.2
"""
import os, sys, types
from pathlib import Path
ROOT, OUT, S = Path(sys.argv[1]).resolve(), Path(sys.argv[2]), float(sys.argv[3])
CELL = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0  # extra scale for cell values only
sys.path.insert(0, str(ROOT / "src"))
from matplotlib import font_manager
FONT_DIRS = [Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts", Path.home() / "mnt/Fonts"]
for f in ["msjh.ttc", "msjhbd.ttc"]:
    font_manager.fontManager.addfont(str(next(d / f for d in FONT_DIRS if (d / f).exists())))
import matplotlib, matplotlib.pyplot as plt
from matplotlib.axes import Axes
import plot_paper_confusion_matrices as M

ABBR = {"Normal": "Nor", "Depression": "Dep", "Anxiety": "Anx", "Bipolar": "Bip", "INVALID": "Inv"}
_text, _title, _xt, _yt = Axes.text, Axes.set_title, Axes.set_xticks, Axes.set_yticks
def text(self, *a, **k):
    if "fontsize" in k: k["fontsize"] *= S * CELL
    return _text(self, *a, **k)
def set_title(self, *a, **k):
    if "fontsize" in k: k["fontsize"] *= S
    k["pad"] = 4
    return _title(self, *a, **k)
def ticks(orig):
    def f(self, t, labels=None, **k):
        k.pop("rotation", None); k.pop("ha", None)
        return orig(self, t, [ABBR.get(l, l) for l in labels] if labels is not None else None, **k)
    return f
Axes.text, Axes.set_title = text, set_title
Axes.set_xticks, Axes.set_yticks = ticks(_xt), ticks(_yt)
_ylab = Axes.set_ylabel
Axes.set_xlabel = lambda self, *a, **k: None          # one shared axis title instead
Axes.set_ylabel = lambda self, *a, **k: None

class Rc:
    def update(self, d):
        d = dict(d)
        if "font.size" in d: d["font.size"] *= S
        matplotlib.rcParams.update(d)
    def __getattr__(self, n): return getattr(matplotlib.rcParams, n)
proxy = types.SimpleNamespace(**{n: getattr(plt, n) for n in dir(plt) if not n.startswith("__")})
proxy.rcParams = Rc()
_subplots = plt.subplots
def subplots(nrows, ncols, **k):
    # L-shaped 3 x 2 grid: C1 alone on top, C2|C3 (base prompt), C4|C5 (optimized prompt);
    # the empty top-right cell holds the shared colorbar.
    k["figsize"] = (6.9, 7.6)
    fig, g = _subplots(3, 2, **k)
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.04, wspace=0.03, hspace=0.05)
    axes = [g[0, 0], g[1, 0], g[1, 1], g[2, 0], g[2, 1]]
    corner = g[0, 1]; corner.axis("off")
    _cb = fig.colorbar
    def colorbar(mappable, ax=None, **kw):
        fig.supxlabel("Predicted label", fontsize=11 * S)
        fig.supylabel("True label", fontsize=11 * S)
        cax = corner.inset_axes([0.10, 0.10, 0.08, 0.80])
        cb = _cb(mappable, cax=cax)
        cb.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        for a in axes: a.tick_params(labelsize=11 * S)
        _ylab(cax, "Row proportion", fontsize=11 * S, labelpad=6)
        return cb
    fig.colorbar = colorbar
    _save = fig.savefig
    def savefig(path, *a, **kw):  # keep the Chinese-label figure in place: write *_en.*
        path = Path(path)
        return _save(path.with_name(f"{path.stem}_en{path.suffix}"), *a, **kw)
    fig.savefig = savefig
    return fig, axes
proxy.subplots = subplots
M.plt = proxy

E, profile = M.load_profile("v2_k1")
orig = M.custom_experiments
M.custom_experiments = lambda p: [dict(e, title=e["code"]) for e in orig(p)]
OUT.mkdir(parents=True, exist_ok=True)
M.plot_custom15(ROOT, OUT, False, profile)
print("done")
