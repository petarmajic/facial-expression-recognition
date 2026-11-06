# Execution

[Notebook](../notebooks/zavrsni.ipynb) · [Notebook guide](../notebooks/README.md)

## Environment

The notebook records **Python 3.11**, **TensorFlow 2.18.0** and **Keras 3.8.0**. Package versions are listed in [requirements-observed.txt](../requirements-observed.txt).

Images are downloaded through KaggleHub. The generators use `/kaggle/input/fer2013/train` and `/kaggle/input/fer2013/test`; adapt these to the downloaded dataset location in your environment.

## Training notebook setup

The first two lines of the training cell are damaged. Replace both with the callback import shown below, then adjust the dataset paths and loading settings for your environment.

| Part | Required setup |
| --- | --- |
| Callback import | Replace the first two lines of the training cell with `from tensorflow.keras.callbacks import ModelCheckpoint`. |
| Input paths | Point the generators to the FER2013 `train/` and `test/` directories. |
| Data subsets | The recorded run uses 80% of `train/` for training and 20% of `test/` for validation. |
| Prediction iterator | Set `shuffle=False` when producing a confusion matrix and use that iterator's labels. |
| Weights loading | Build the model architecture, then call `model.load_weights("model.weights.h5")`. |

The checkpoint callback uses `save_weights_only=True`. To load a checkpoint, first build the same model architecture and then load its weights:

```python
model.load_weights("model.weights.h5")
```

The repository contains the training logs and results, but the dataset and trained checkpoint must be obtained separately.

## Rebuild the figures

From the repository root:

```bash
python -m pip install matplotlib==3.10.0
python scripts/extract_evidence.py
python scripts/make_figures.py
```

`extract_evidence.py` reads model definitions and saved outputs into the CSV/JSON files in [results/](../results/README.md). `make_figures.py` builds the architecture, expression examples, class-distribution chart and validation curves. Image examples come from [assets/source/](../assets/source/README.md).

The figures use Matplotlib, Pillow and the included result files.

[Experiment](experiment.md) · [References](references.md) · [Overview](../README.md)
