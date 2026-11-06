"""Plot the CNN architecture, dataset examples and recorded validation metrics."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from draw_architecture import draw_architecture, draw_dataset_examples

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
NAVY, MUTED = "#183448", "#536779"
BLUE, TEAL = "#326b9b", "#247e74"
LINE, PALE = "#d4dfe7", "#f4f7fa"
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 12,
                     "text.color": NAVY, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.edgecolor": LINE, "svg.fonttype": "path",
                     "svg.hashsalt": "facial-expression-recognition",
                     "savefig.facecolor": "white"})


def finish(fig, name):
    fig.savefig(ASSETS / f"{name}.svg", metadata={"Date": None})
    plt.close(fig)


def validation_curve():
    with (ROOT / "results/epoch_metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    x = [int(r["epoch"]) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.45))
    fig.subplots_adjust(left=.075, right=.96, bottom=.20, top=.84, wspace=.28)
    for ax, key, color, title in zip(axes,
            ["validation_accuracy", "validation_loss"], [TEAL, BLUE],
            ["(a) Validation accuracy", "(b) Validation cross-entropy"]):
        y = [float(r[key]) for r in rows]
        ax.plot(x, y, color=color, linewidth=2)
        ax.scatter([x[-1]], [y[-1]], color=color, s=35, zorder=3)
        ax.set(xlim=(1, 53), xlabel="Epoch", title=title)
        ax.title.set_fontsize(12)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e8edf1", linewidth=.8)
        ax.set_axisbelow(True)
        ax.set_xticks([1, 10, 20, 30, 40, 50])
        ax.tick_params(labelsize=10, length=3)
        if key == "validation_accuracy":
            ax.set_ylim(0, .75)
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.annotate("65.22%", (50, y[-1]), xytext=(-4, 12),
                        textcoords="offset points", ha="right", color=color, weight="bold", fontsize=11)
        else:
            ax.set_ylim(.8, 2.05)
            ax.annotate("0.9432", (50, y[-1]), xytext=(-4, 12),
                        textcoords="offset points", ha="right", color=color, weight="bold", fontsize=11)
    finish(fig, "validation-curves")


def class_distribution():
    with (ROOT / "results/class_counts.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    fig.subplots_adjust(left=.14, right=.92, bottom=.14, top=.73)
    fig.text(.025, .94, "FER2013 class balance", fontsize=17, weight="bold")
    fig.text(.025, .86, "Full training directory · 28,709 images before generator subsetting", fontsize=11, color=MUTED)
    counts = [int(r["train_folder_images"]) for r in rows]
    ax.barh(range(7), counts, height=.62,
            color=[TEAL if r["label"]=="disgust" else BLUE for r in rows])
    for i, count in enumerate(counts):
        ax.text(count+110, i, f"{count:,}", fontsize=10, va="center", color=NAVY)
    ax.set_yticks(range(7), [r["label"].capitalize() for r in rows])
    ax.invert_yaxis()
    ax.set(xlim=(0, 8300), xlabel="Images")
    ax.set_xticks([0, 2000, 4000, 6000, 8000], ["0", "2,000", "4,000", "6,000", "8,000"])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e8edf1")
    ax.set_axisbelow(True)
    finish(fig, "class-distribution")


if __name__ == "__main__":
    ASSETS.mkdir(exist_ok=True)
    draw_architecture()
    draw_dataset_examples()
    validation_curve()
    class_distribution()
    print("Built four SVG figures from the extracted notebook evidence.")
