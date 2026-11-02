# Running and inspecting the project

The [original notebook](../notebooks/zavrsni.ipynb) contains code, saved logs, a model summary and plots. It is retained byte-for-byte as supplied. Documentation extraction and figure generation have been run separately; model training and inference have not been repeated for this repository.

## Inspect the saved experiment

Start with the [notebook guide](../notebooks/README.md) to locate the data pipeline, model, training output and plots. The [experiment document](experiment.md) explains the metric sources and evaluation issue. The [results directory](../results/README.md) contains small extracts that can be inspected without TensorFlow or dataset access.

## Rebuild the documentation figures

Run from the repository root:

```bash
python -m pip install matplotlib==3.10.0
python scripts/extract_evidence.py
python scripts/make_figures.py
```

`extract_evidence.py` reads the notebook as JSON, inspects selected code using `ast`, and extracts saved text outputs. It uses only the Python standard library. It writes CSV/JSON files under `results/`; `make_figures.py` reads those extracts and writes four SVGs under `assets/`.

The plotting entry point calls `draw_architecture.py` for the two-panel architecture and dataset examples. Image excerpts come from the original montage in `assets/source/`; Pillow, a Matplotlib dependency, reads that PNG. No separate dataset download is needed. [Image provenance](../assets/source/README.md) explains the illustration's source and limits. None of these scripts executes notebook cells, trains a model or runs inference.

Figures use STIX typography and a consistent muted palette. SVG text is stored as glyph paths so that the figures keep the same typography in GitHub and other viewers. Images have a white background for predictable rendering in light and dark interfaces.

## Environment recorded in the notebook

The saved package output shows **Python 3.11 paths**, **TensorFlow 2.18.0**, and **Keras 3.8.0**. [requirements-observed.txt](../requirements-observed.txt) lists relevant versions visible in that output. It is a snapshot of observed package versions, not a complete lock file or a newly validated training environment.

The thesis describes Google Colab. The code metadata contains Colab information, but the saved run uses `/kaggle/input/fer2013` paths. The exact execution host and accelerator cannot be established from those artefacts alone, so no GPU model or training-time benchmark is claimed.

## Before executing the training notebook

The original currently needs a few explicit repairs. Make a working copy so the archived notebook and saved outputs remain intact.

| Item | What is in the original | Required adjustment in a working copy |
| --- | --- | --- |
| Training-cell import, cell 4 | Starts with `Žr` and `om tensorflow.keras.callbacks import ModelCheckpoint` | Replace those two lines with `from tensorflow.keras.callbacks import ModelCheckpoint`. The saved logs show an earlier execution, but the current source does not parse. |
| Dataset paths, cell 2 | Hard-coded `/kaggle/input/fer2013/train` and `/test` | Use `os.path.join(path, "train")` and `os.path.join(path, "test")` after the download cell, or the actual mounted dataset location. |
| Split roles, cell 2 | Training subset of `train/`; validation subset of `test/` | Keep this only when deliberately reproducing the archived protocol. A new train/validation/test design is a different experiment. |
| Confusion-matrix generator, cell 8 | Shuffled generator predictions compared with fixed `.classes` labels | Use a separate unshuffled iterator for prediction and its corresponding labels. |
| Checkpoint loading, cell 10 | Calls `load_model()` on a weights-only checkpoint | Rebuild the architecture from cell 3, then call `model.load_weights(...)`. |

These adjustments are documented rather than silently applied to the original. A clean “Run all” of the training notebook has not been verified.

## Weights and dataset

The checkpoint callback is configured to save weights to `model.weights.h5`. The supplied notebook's final existence check prints that the model file is missing, and no weights file is present in the reviewed attachments. Saved training logs document the run; they are not a substitute for the trained parameters.

For a recovered weights file, loading should follow the original architecture:

```python
# After constructing the same model as notebook cell 3:
model.load_weights("model.weights.h5")
```

This is the correct loading method for the weights-only format; it has not been tested against this run's unavailable checkpoint. See [Keras saving and loading guidance](https://keras.io/getting_started/faq/#what-are-my-options-for-saving-models).

Obtain images from the [FER2013 source dataset](https://www.kaggle.com/datasets/msambare/fer2013). The repository does not include the full dataset or trained weights. A new run would need data access and would consume training compute; it should have its own output directory and results record.

[Notebook cell map](../notebooks/README.md) · [Evidence ledger](evidence.md) · [Back to overview](../README.md)
