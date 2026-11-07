# Execution

The [training notebook](../notebooks/zavrsni.ipynb) includes the output of the original run. To inspect the learning curves or regenerate the figures, you can work from those outputs. To train another model, you will need the FER2013 images and the notebook adjustments below.

## Environment

I trained the CNN with **Python 3.11**, **TensorFlow 2.18.0** and **Keras 3.8.0**. The first notebook cell prints the installed package versions. [Packages recorded during training](../requirements-observed.txt).

## Training notebook setup

1. **Point both generators to the downloaded images.** The KaggleHub download cell already sets `train_path` and `test_path`. In the next cell, use `train_dir = train_path` and `test_dir = test_path` in place of the fixed paths `/kaggle/input/fer2013/train` and `/kaggle/input/fer2013/test`. To follow the recorded split, keep the training subset of `train/` (80%) and the validation subset of `test/` (20%).
2. **Repair the callback import.** The first two lines of the archived training cell read `Žr` and `om tensorflow.keras.callbacks import ModelCheckpoint`. Replace both lines with `from tensorflow.keras.callbacks import ModelCheckpoint` before running the cell.
3. **Load the saved weights into the CNN.** The callback writes weights only when validation accuracy improves. After training, replace the final cell’s `load_model` call with `model.load_weights("model.weights.h5")`. In a fresh session, run the model definition cell first. The repository includes the logs, but you will need to train the model or supply a compatible checkpoint to run predictions.
4. **Keep prediction order fixed for the confusion matrix.** Create a separate iterator from `validation_datagen` using the same directory, image and label settings as `validation_generator`. Keep `subset="validation"` and pass `shuffle=False`. Use that iterator for both `model.predict(...)` and `.classes` so each prediction is compared with the label of the same image. [How ordering affects the saved matrix](experiment.md#confusion-matrix-ordering).

## Rebuild the figures

Run these commands from the repository root to recreate the published figures from the notebook’s saved outputs and the included [FER2013 example image](../assets/source/README.md).

```bash
python -m pip install matplotlib==3.10.8 pillow==13.0.0
python scripts/extract_evidence.py
python scripts/make_figures.py
```

The first script reads the model definition and training logs and writes the [result tables](../results/README.md). The second draws the architecture, expression examples, class counts and validation curves. The plotting environment above reproduces the committed figures. The original training environment used Matplotlib 3.10.0 and Pillow 11.2.1.

[Notebook guide](../notebooks/README.md) · [Experiment](experiment.md) · [References](references.md) · [Overview](../README.md)
