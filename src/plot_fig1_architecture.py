"""Draw the paper's Fig. 1 (overall system architecture) for the v2, top-k = 1 version.

All settings shown in the figure follow the final experiment
(runs/NESTED_70_10_20_INCREMENTAL/MAIN_EVAL_AUGV2_K1) and the manuscript text:
70:10:20 split, aug_v2 corpus (6,994 originals + 12,501 paraphrases = 19,495 records),
all-MiniLM-L6-v2 + FAISS IndexFlatIP, top-1 retrieval, Llama 3.1 8B (temperature 0),
unique-match label parsing.  No result numbers are drawn (they live in Tables IX-XI).

Usage:  python src/plot_fig1_architecture.py [--out-dir reports/figures]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FONT_DIRS = [Path("C:/Windows/Fonts"), Path.home() / "mnt/Fonts"]
for d in FONT_DIRS:
    for f in ("arial.ttf", "arialbd.ttf"):
        if (d / f).is_file():
            font_manager.fontManager.addfont(str(d / f))
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 7})

# colours (Blues family, matching Fig. 5)
PANEL = "#F3F7FB"
EDGE = "#2F5D8A"
BOX = "#FFFFFF"
KEY = "#DCE9F5"      # data stores
TPE = "#FFF4DD"
TITLE = "#1F4E79"

W, H = 14.0, 7.1


def box(ax, x, y, w, h, title, body="", fc=BOX, title_size=6.8, body_size=5.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=EDGE, lw=0.9))
    if body:
        ax.text(x + w / 2, y + h * 0.74, title, ha="center", va="center", fontsize=title_size,
                fontweight="bold", color="#10263D", linespacing=1.05)
        ax.text(x + w / 2, y + h * 0.31, body, ha="center", va="center",
                fontsize=body_size, color="#26313B", linespacing=1.2)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=title_size,
                fontweight="bold", color="#10263D")
    return (x, y, w, h)


def arrow(ax, p, q, label="", dashed=False, label_dy=0.17, rad=0.0, color=EDGE):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=0.9,
                                 color=color, linestyle="--" if dashed else "-",
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))
    if label:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + label_dy, label, ha="center",
                va="bottom", fontsize=5.6, color="#3B4A57", linespacing=1.15)


def right(b):
    x, y, w, h = b
    return (x + w, y + h / 2)


def left(b):
    x, y, w, h = b
    return (x, y + h / 2)


def top(b):
    x, y, w, h = b
    return (x + w / 2, y + h)


def bottom(b):
    x, y, w, h = b
    return (x + w / 2, y)


def draw(out_dir: Path) -> None:
    fig = plt.figure(figsize=(7.0, 7.0 * H / W))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    bw, gap, x0 = 1.58, 0.80, 0.26
    xs = [x0 + i * (bw + gap) for i in range(6)]

    # ---------------- Stages 1 & 2 ----------------
    ax.add_patch(FancyBboxPatch((0.1, 4.0), W - 0.2, 3.0, boxstyle="round,pad=0,rounding_size=0.15",
                                fc=PANEL, ec="#B7C9DB", lw=0.8))
    ax.text(0.3, 6.75, "Stages 1 & 2: Data preparation, LLM-based augmentation, and vector database construction",
            fontsize=7.4, fontweight="bold", color=TITLE, va="center")
    y1, h1 = 4.25, 2.15
    b1 = box(ax, xs[0], y1, bw, h1, "Mental-health\ndataset", "Normal, Depression,\nAnxiety, Bipolar")
    b2 = box(ax, xs[1], y1, bw, h1, "Training\nset", "6,994 statements\n(70:10:20 split)")
    b3 = box(ax, xs[2], y1, bw, h1, "LLM\nparaphrasing", "Llama 3.1 8B\ntemperature 0.7\n2 per statement")
    b4 = box(ax, xs[3], y1, bw, h1, "Augmented\ncorpus", "19,495 records\n(6,994 originals +\n12,501 paraphrases)", fc=KEY)
    b5 = box(ax, xs[4], y1, bw, h1, "Sentence-\nBERT", "all-MiniLM-L6-v2\n384-d")
    b6 = box(ax, xs[5], y1, bw, h1, "FAISS vector\ndatabase", "IndexFlatIP\n(cosine)", fc=KEY)
    arrow(ax, right(b1), left(b2), "Clean,\nbalance,\nsplit")
    arrow(ax, right(b2), left(b3), "Rewrite")
    arrow(ax, right(b3), left(b4), "Rule-based\nfilters")
    arrow(ax, right(b4), left(b5), "Encode")
    arrow(ax, right(b5), left(b6), "Index")

    # ---------------- Stage 3 ----------------
    ax.add_patch(FancyBboxPatch((0.1, 0.1), W - 0.2, 3.45, boxstyle="round,pad=0,rounding_size=0.15",
                                fc=PANEL, ec="#B7C9DB", lw=0.8))
    ax.text(0.3, 3.3, "Stage 3: Retrieval-augmented classification",
            fontsize=7.4, fontweight="bold", color=TITLE, va="center")
    y3, h3 = 0.3, 2.15
    c1 = box(ax, xs[0], y3, bw, h3, "Test\nstatement", "Unseen input text")
    c2 = box(ax, xs[1], y3, bw, h3, "Query\nembedding", "Same\nSentence-BERT")
    c3 = box(ax, xs[2], y3, bw, h3, "Top-1\nretrieval", "\"{statement}\ntrue_label is\n{status}\"")
    c4 = box(ax, xs[3], y3, bw, h3, "Prompt\nassembly", "Task + context\n+ input text\n+ output rule")
    c5 = box(ax, xs[4], y3, bw, h3, "Llama 3.1 8B", "Ollama, Q4_K_M\ntemperature 0")
    c6 = box(ax, xs[5], y3, bw, h3, "Label\nparsing", "One label →\nprediction;\notherwise invalid")
    arrow(ax, right(c1), left(c2), "Encode")
    arrow(ax, right(c2), left(c3), "Search")
    arrow(ax, right(c3), left(c4), "Context")
    arrow(ax, right(c4), left(c5), "Prompt")
    arrow(ax, right(c5), left(c6), "Response")

    # TPE-optimized prompt feeding the prompt assembly (C4/C5)
    t = box(ax, xs[3] - 0.55, 2.62, bw + 1.1, 0.78, "TPE prompt optimization",
            "300 prompt variants, validation set", fc=TPE, title_size=6.4, body_size=5.6)
    arrow(ax, bottom(t), top(c4))
    # retrieval link from the vector database
    yl = 3.77
    p = (xs[5] + bw / 2, y1)
    q = (xs[2] + bw / 2, y3 + h3)
    kw = dict(lw=0.9, color=EDGE, linestyle="--")
    ax.add_patch(FancyArrowPatch(p, (p[0], yl), arrowstyle="-", **kw))
    ax.add_patch(FancyArrowPatch((p[0], yl), (q[0], yl), arrowstyle="-", **kw))
    ax.add_patch(FancyArrowPatch((q[0], yl), q, arrowstyle="-|>", mutation_scale=8, shrinkB=1, **kw))
    ax.text(xs[4] + bw / 2, yl + 0.04, "Similarity search (top-k = 1)", ha="center", va="bottom",
            fontsize=5.6, color="#3B4A57")

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig1_architecture.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "reports" / "figures")
    draw(ap.parse_args().out_dir)


if __name__ == "__main__":
    main()
