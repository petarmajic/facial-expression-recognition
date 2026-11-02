"""Build README figures from extracted notebook evidence. No model execution."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
NAVY, MUTED = "#183448", "#536779"
BLUE, TEAL = "#326b9b", "#247e74"
LINE, PALE = "#d4dfe7", "#f4f7fa"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                     "text.color": NAVY, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.edgecolor": LINE, "svg.fonttype": "path",
                     "svg.hashsalt": "facial-expression-recognition",
                     "savefig.facecolor": "white"})


def finish(fig, name):
    fig.savefig(ASSETS / f"{name}.svg", metadata={"Date": None})
    plt.close(fig)


def overview():
    fig, ax = plt.subplots(figsize=(11, 3.7))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set(xlim=(0, 1100), ylim=(370, 0))
    ax.axis("off")
    ax.text(25, 34, "From pixels to seven expression classes", fontsize=17, weight="bold")
    ax.text(25, 67, "Input: 48 × 48 × 1  ·  pixel values rescaled to [0, 1]", color=MUTED, fontsize=12)
    cards = [
        (25, 225, "01  FEATURE BLOCK", "32 → 64 filters", "3 × 3 Conv + ReLU + BN", "Pool 2 × 2 · Dropout 0.25", "22 × 22 × 64"),
        (285, 225, "02  FEATURE BLOCK", "128 → 128 filters", "3 × 3 Conv + ReLU + BN", "Pool 2 × 2 · Dropout 0.25", "9 × 9 × 128"),
        (545, 225, "03  FEATURE BLOCK", "256 → 256 filters", "3 × 3 Conv + ReLU + BN", "Pool 2 × 2 · Dropout 0.25", "2 × 2 × 256"),
        (805, 270, "04  CLASSIFIER", "1,024 → 256 → 7", "Flatten · Dense 256 + ReLU", "BN · Dropout 0.50 · Dense 7", "Softmax probabilities")
    ]
    for i, (x, width, heading, title, line1, line2, shape) in enumerate(cards):
        edge, fill = (TEAL, "#edf6f2") if i == 3 else (LINE, PALE)
        box = FancyBboxPatch((x, 102), width, 185, boxstyle="round,pad=0,rounding_size=7",
                             facecolor=fill, edgecolor=edge, linewidth=1.1)
        ax.add_patch(box)
        ax.text(x+14, 128, heading, fontsize=10, color=TEAL, weight="bold")
        ax.text(x+14, 166, title, fontsize=15, weight="bold")
        ax.text(x+14, 199, line1, fontsize=10.5, color=MUTED)
        ax.text(x+14, 223, line2, fontsize=10.3, color=MUTED)
        ax.text(x+14, 263, shape, fontsize=12, weight="bold", color=BLUE if i<3 else TEAL)
        if i < 3:
            ax.add_patch(FancyArrowPatch((x+width+5, 190), (cards[i+1][0]-6, 190),
                                        arrowstyle="-|>", mutation_scale=13,
                                        color=MUTED, linewidth=1.3))
    ax.text(25, 326, "Six convolutions · three pooling stages · 1,394,183 parameters", fontsize=12, weight="bold")
    ax.text(25, 352, "BN = Batch Normalization after every convolution and after Dense 256.", fontsize=10.5, color=MUTED)
    finish(fig, "model-overview")


def validation_curve():
    with (ROOT / "results/epoch_metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    x = [int(r["epoch"]) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    fig.subplots_adjust(left=.075, right=.96, bottom=.20, top=.75, wspace=.28)
    fig.text(.025, .94, "Validation over 50 epochs", weight="bold", fontsize=17)
    fig.text(.025, .86, "1,432 images · monitored during training and checkpoint selection", color=MUTED, fontsize=11)
    for ax, key, color, title in zip(axes,
            ["validation_accuracy", "validation_loss"], [TEAL, BLUE],
            ["Accuracy", "Cross-entropy loss"]):
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
    fig.text(.025, .035, "Saved epoch logs, rounded to four decimal places. These are validation metrics, not an independent test result.",
             fontsize=9.5, color=MUTED)
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
    overview()
    validation_curve()
    class_distribution()
    print("Built three SVG figures from the extracted notebook evidence.")
