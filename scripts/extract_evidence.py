"""Extract documentation data from saved notebook cells without executing them.

This reader is specific to the archived thesis notebook. It does not load
TensorFlow, download FER2013, run inference, or train a model.
"""

import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/zavrsni.ipynb"
RESULTS = ROOT / "results"
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def source(cell):
    return "".join(cell.get("source", []))


def output_text(cell):
    return ANSI.sub("", "".join(
        "".join(o.get("text", [])) for o in cell.get("outputs", [])
    ))


def write_csv(name, rows):
    with (RESULTS / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def architecture(cell):
    """Derive shapes/parameter counts for the exact Sequential model in cell 3."""
    rows = []
    shape = (48, 48, 1)
    for statement in ast.parse(source(cell)).body:
        call = statement.value if isinstance(statement, ast.Expr) else None
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "model" and call.func.attr == "add"):
            continue
        layer = call.args[0]
        kind = layer.func.id
        args = [ast.literal_eval(a) for a in layer.args]
        kw = {k.arg: ast.literal_eval(k.value) for k in layer.keywords}
        params = trainable = 0
        detail = ""
        if kind == "Conv2D":
            filters = args[0]
            kh, k_width = kw["kernel_size"]
            height, width, channels = shape
            params = trainable = (kh * k_width * channels + 1) * filters
            shape = (height - kh + 1, width - k_width + 1, filters)
            detail = f"{filters} filters; {kh}x{k_width}; ReLU; valid padding"
        elif kind == "BatchNormalization":
            params = 4 * shape[-1]
            trainable = 2 * shape[-1]
            detail = "gamma, beta, moving mean, moving variance"
        elif kind == "MaxPooling2D":
            ph, pw = kw["pool_size"]
            shape = (shape[0] // ph, shape[1] // pw, shape[2])
            detail = f"{ph}x{pw}; stride {ph}x{pw}"
        elif kind == "Dropout":
            detail = f"rate {args[0]}"
        elif kind == "Flatten":
            shape = (math.prod(shape),)
        elif kind == "Dense":
            params = trainable = (shape[0] + 1) * args[0]
            shape = (args[0],)
            detail = f"{args[0]} units; {kw['activation']}"
        else:
            raise ValueError(f"Unrecognised layer: {kind}")
        rows.append(dict(layer=len(rows) + 1, type=kind, details=detail,
                         output_shape=" x ".join(map(str, shape)),
                         parameters=params, trainable_parameters=trainable))
    assert sum(r["parameters"] for r in rows) == 1394183
    assert sum(r["trainable_parameters"] for r in rows) == 1391943
    return rows


def main():
    notebook_bytes = NOTEBOOK.read_bytes()
    cells = json.loads(notebook_bytes)["cells"]
    assert len(cells) == 11, "Review cell references before extracting a new notebook."
    RESULTS.mkdir(exist_ok=True)

    # Counts come from os.walk's saved output: five names + the remaining files.
    folder = label = None
    counts = {}
    for line in output_text(cells[1]).splitlines():
        if line in ("  train/", "  test/"):
            folder = line.strip().rstrip("/")
        elif re.fullmatch(r"    [a-z]+/", line):
            label = line.strip().rstrip("/")
        elif re.fullmatch(r"      \.\.\. i još \d+ datoteka", line):
            counts[(folder, label)] = int(re.search(r"\d+", line).group()) + 5
    labels = sorted(label for folder, label in counts if folder == "train")
    assert labels == ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    class_rows = [dict(label=label, train_folder_images=counts[("train", label)],
                       test_folder_images=counts[("test", label)]) for label in labels]
    assert sum(r["train_folder_images"] for r in class_rows) == 28709
    assert sum(r["test_folder_images"] for r in class_rows) == 7178
    write_csv("class_counts.csv", class_rows)

    # Only full epoch-end lines include all four metrics. Console values are rounded.
    pattern = (r"accuracy: ([\d.]+) - loss: ([\d.]+) - "
               r"val_accuracy: ([\d.]+) - val_loss: ([\d.]+)")
    values = re.findall(pattern, output_text(cells[4]))
    assert len(values) == 50
    epoch_rows = [dict(epoch=i, console_training_accuracy=a,
                       console_training_loss=loss,
                       validation_accuracy=va, validation_loss=vl)
                  for i, (a, loss, va, vl) in enumerate(values, start=1)]
    write_csv("epoch_metrics.csv", epoch_rows)
    assert max(float(r["validation_accuracy"]) for r in epoch_rows) == 0.6522
    layer_rows = architecture(cells[3])
    write_csv("architecture.csv", layer_rows)

    summary = output_text(cells[7])
    def number(pattern):
        return float(re.search(pattern, summary).group(1))
    totals = [int(x) for x in re.findall(r"Found (\d+) images", output_text(cells[2]))]
    assert totals == [22968, 1432]
    evidence = {
        "source_notebook": "notebooks/zavrsni.ipynb",
        "source_sha256": hashlib.sha256(notebook_bytes).hexdigest(),
        "cell_numbering": "zero-based JSON cell indices",
        "dataset_source_in_code": "msambare/fer2013",
        "generator_images": {"training": totals[0], "validation": totals[1]},
        "validation_origin": "20% subset of the test directory, used throughout fit()",
        "epochs": len(epoch_rows),
        "saved_history_summary": {
            "training_accuracy": number(r"Finalna training accuracy: ([\d.]+)"),
            "validation_accuracy": number(r"Finalna validation accuracy: ([\d.]+)"),
            "training_loss": number(r"Finalni training loss: ([\d.]+)"),
            "validation_loss": number(r"Finalni validation loss: ([\d.]+)"),
            "best_validation_epoch": int(number(r"u epohi (\d+)"))
        },
        "model_parameters": {"total": 1394183, "trainable": 1391943,
                             "non_trainable": 2240},
        "epoch_csv_source": "cell 4 saved console output, rounded to four decimal places",
        "class_counts_source": "cell 1 saved directory listing, not a new dataset scan",
        "architecture_source": "cell 3 layer definitions; counts checked against saved summary",
        "notes": [
            "Training console metrics differ from the saved history summary; do not merge them.",
            "No independent full test evaluation is saved in this notebook.",
            "The saved confusion matrix uses a shuffled prediction generator against fixed labels."
        ]
    }
    (RESULTS / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"Extracted {len(epoch_rows)} epoch rows, {len(class_rows)} class counts "
          f"and {len(layer_rows)} layer records.")
    print("Source notebook SHA-256:", evidence["source_sha256"])


if __name__ == "__main__":
    main()
