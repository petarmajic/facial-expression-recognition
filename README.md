# Facial Expression Recognition

I trained a convolutional neural network from scratch to classify **48 × 48 grayscale face images** into seven FER2013 expression categories. The project covers image loading and augmentation, CNN design in Keras, training and analysis of the validation results.

**Python · TensorFlow / Keras · NumPy · Matplotlib / Seaborn**

[Notebook](notebooks/zavrsni.ipynb) · [Architecture](docs/architecture.md) · [Data](docs/data.md) · [Experiment](docs/experiment.md)

## Expression classes

![FER2013 examples of the seven target classes: angry, disgust, fear, happy, neutral, sad and surprise.](assets/dataset-examples.svg)

**Figure 1.** FER2013 examples of the seven expression classes. [Image source](assets/source/README.md).

## Architecture

![Two-panel CNN architecture: a FER2013 face passes through six convolutional layers and three pooling stages; the resulting feature tensor feeds a 256-unit dense layer and a seven-class Softmax output.](assets/model-overview.svg)

**Figure 2.** Six convolutional layers and three pooling stages feed a dense classifier. Tensor dimensions are **height × width × channels**; volumes are schematic. The network has **1,394,183 parameters**, of which **1,391,943 are trainable**.

[Layer-by-layer architecture](docs/architecture.md)

Each feature block applies two convolutions followed by max pooling and Dropout. As the image becomes smaller, the network increases the number of feature channels. The final **2 × 2 × 256** tensor is flattened into 1,024 values and passed to a 256-unit dense layer and the seven-class Softmax output.

## What I implemented

| Area | Technical work | Details |
| --- | --- | --- |
| Image preprocessing | Loaded grayscale images, rescaled pixels to 0–1 and applied random shifts and horizontal flips. | [Preprocessing](docs/data.md#image-preprocessing) |
| CNN | Six 3 × 3 convolutions, max pooling, Batch Normalization, Dropout and a dense classifier. | [Model definition](docs/architecture.md) |
| Training | Trained for 50 epochs with Adam and categorical cross-entropy, saving weights when validation accuracy improved. | [Configuration](docs/experiment.md#training-configuration) |
| Analysis | Examined image counts by class, plotted learning curves and checked how validation results were calculated. | [Findings](docs/experiment.md#recorded-results) |

## Recorded experiment

After each epoch, I evaluated the model on the validation subset. A checkpoint callback kept the weights with the highest validation accuracy.

| Setting or result | Value |
| --- | --- |
| Images used | **22,968 training · 1,432 validation** |
| Training setup | 50 epochs · batch 64 · Adam, learning rate **0.0001** |
| Best validation accuracy | **65.22%**, epoch 50 |
| Validation cross-entropy at epoch 50 | **0.9432** |

![Validation accuracy and cross-entropy over the 50 recorded epochs.](assets/validation-curves.svg)

**Figure 3.** Validation accuracy and cross-entropy over the 50 training epochs.

Validation accuracy rose from 21.30% to 65.22% while the loss fell from 1.8888 to 0.9432. Both reached their best recorded value at epoch 50. This is a validation result from one training run; the notebook does not report a separate final test score.

[Training setup and evaluation](docs/experiment.md)

## Repository guide

| Resource | Contents |
| --- | --- |
| [Notebook](notebooks/zavrsni.ipynb) · [Notebook guide](notebooks/README.md) | Image loading, augmentation, model definition, training logs and plots. |
| [Data](docs/data.md) · [Architecture](docs/architecture.md) | Dataset examples, images per class, preprocessing and layer dimensions. |
| [Experiment](docs/experiment.md) · [Results](results/README.md) | Hyperparameters, learning curves and metric tables. |
| [Execution](docs/reproduction.md) · [References](docs/references.md) | Environment, notebook setup, figure commands and thesis reference. |

*Bachelor's thesis, University of Zagreb, Faculty of Electrical Engineering and Computing (FER), June 2025. Supervisor: Prof. Sonja Grgić, PhD.*
