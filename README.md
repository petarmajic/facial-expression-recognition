# Facial Expression Recognition

I implemented and trained a CNN to classify seven facial-expression categories in FER2013. The project covers image preparation, a six-layer convolutional feature extractor, regularisation, and analysis of a saved 50-epoch training run on low-resolution, imbalanced image data.

**Python · TensorFlow / Keras · CNNs · image augmentation · NumPy · Matplotlib / Seaborn**

[Original notebook](notebooks/zavrsni.ipynb) · [Architecture](docs/architecture.md) · [Experiment and results](docs/experiment.md) · [Running the code](docs/reproduction.md)

## What I implemented

| Area | Work in this project | Code and evidence |
| --- | --- | --- |
| Image preparation | Loaded 48 × 48 grayscale images, rescaled pixels to [0, 1], and applied random horizontal flips and shifts of up to 10%. | [Data pipeline](docs/data.md#image-pipeline) |
| CNN implementation | Built six 3 × 3 convolutions, three pooling stages and a seven-class classifier, with Batch Normalization and Dropout. | [Layer shapes and parameters](docs/architecture.md) |
| Model training | Trained with Adam and categorical cross-entropy for 50 epochs; selected the best weights by validation accuracy. | [Training configuration](docs/experiment.md#training-configuration) |
| Experiment analysis | Plotted training and validation curves, inspected class balance and produced a confusion matrix. | [Results and interpretation](docs/experiment.md#recorded-results) |

## Model at a glance

![CNN architecture: three feature blocks with 32/64, 128/128 and 256/256 filters, followed by a 256-unit dense layer and seven-class Softmax.](assets/model-overview.svg)

Each feature block contains two convolutions followed by pooling and Dropout. Spatial resolution decreases from 48 × 48 to 2 × 2 while the number of channels increases. The final 1,024 features feed a 256-unit dense layer and the seven-class output. The model has **1,394,183 parameters**, including **1,391,943 trainable parameters**.

[Full architecture and parameter counts →](docs/architecture.md)

## Experiment and results

| Data and run | Verified value |
| --- | --- |
| Dataset | FER2013, loaded from `msambare/fer2013` |
| Training / validation images used | **22,968 / 1,432** |
| Input / batch size | 48 × 48 × 1 / 64 |
| Optimiser / learning rate | Adam / **0.0001** |
| Training duration | 50 epochs |
| Best recorded validation accuracy | **65.22%**, epoch 50 |
| Validation cross-entropy at that epoch | **0.9432** |

![Validation accuracy rises to 65.22 percent and validation cross-entropy falls to 0.9432 over 50 epochs.](assets/validation-curves.svg)

The run shows improving validation accuracy and decreasing validation loss. These values come from the saved notebook outputs. The validation generator uses a subset of the dataset's `test` directory throughout training and checkpoint selection, so **65.22% is a validation result**, rather than an independent test score.

The original confusion matrix also has an ordering issue: predictions from a shuffled generator are compared with fixed-order class labels. It is preserved in the notebook, but it cannot support reliable per-class conclusions until evaluation is repeated with aligned predictions and labels.

[Data split](docs/data.md#the-split-actually-used) · [Result interpretation and evaluation details](docs/experiment.md#evaluation-details) · [Extracted epoch values](results/epoch_metrics.csv)

## Dataset and task

The model predicts the dataset labels **angry, disgust, fear, happy, neutral, sad, and surprise** from grayscale face images. Low resolution and differences in lighting, pose, and expression make the task difficult. The source training directory is also imbalanced: `happy` has 7,215 images, compared with 436 for `disgust`.

[Class distribution and preprocessing →](docs/data.md)

## Code and documentation

| Start here | Contents |
| --- | --- |
| [Original notebook](notebooks/zavrsni.ipynb) | Source code, model summary, training logs and saved plots. |
| [Notebook guide](notebooks/README.md) | Cell map and execution notes. |
| [Architecture](docs/architecture.md) | Layer-by-layer shapes, parameter counts and regularisation. |
| [Experiment](docs/experiment.md) | Configuration, recorded results and interpretation. |
| [Reproduction notes](docs/reproduction.md) | Observed environment, paths, checkpoint handling and execution issues. |
| [Evidence and sources](docs/evidence.md) | Where each reported value comes from, including differences from the thesis text. |
| [Results](results/README.md) | Small CSV/JSON extracts used by the documentation and figures. |

The documentation figures can be rebuilt without downloading data or training a model:

```bash
python scripts/extract_evidence.py
python scripts/make_figures.py
```

The extraction script uses the Python standard library; the figure script requires Matplotlib. See [reproduction notes](docs/reproduction.md#rebuild-the-documentation-figures) for setup.

*Developed for my bachelor's thesis at the University of Zagreb, Faculty of Electrical Engineering and Computing (FER), June 2025. The English documentation presents the implementation and saved experiment outputs; the original Croatian notebook is retained.*
