# Facial Expression Recognition

Seven-class facial-expression classification on **FER2013** using a CNN trained from scratch. The experiment examines learning from **48 × 48 grayscale images**, with a focus on convolutional feature extraction, regularisation and class imbalance.

**Python · TensorFlow / Keras · NumPy · Matplotlib / Seaborn**

[Notebook](notebooks/zavrsni.ipynb) · [Architecture](docs/architecture.md) · [Data](docs/data.md) · [Experiment](docs/experiment.md)

## Expression classes

![FER2013 examples of the seven target classes: angry, disgust, fear, happy, neutral, sad and surprise.](assets/dataset-examples.svg)

**Figure 1.** One FER2013 training example per target class. These are dataset examples, not model predictions. [Image source](assets/source/README.md).

## Architecture

![Two-panel CNN architecture: a FER2013 face passes through six convolutional layers and three pooling stages; the resulting feature tensor feeds a 256-unit dense layer and a seven-class Softmax output.](assets/model-overview.svg)

**Figure 2.** CNN implemented in the notebook. Tensor dimensions are **height × width × channels**. The input photograph illustrates the dataset; tensor volumes are schematic. The network has **1,394,183 parameters**, of which **1,391,943 are trainable**.

[Layer-by-layer architecture](docs/architecture.md) · [Image source](assets/source/README.md)

## What I implemented

| Area | Technical work | Details |
| --- | --- | --- |
| Image pipeline | Grayscale loading, pixel rescaling, random shifts and horizontal flips. | [Preprocessing](docs/data.md#image-pipeline) |
| CNN | Six 3 × 3 convolutions, max pooling, Batch Normalization, Dropout and a dense classifier. | [Model definition](docs/architecture.md) |
| Training | Adam, categorical cross-entropy and checkpoint selection by validation accuracy. | [Configuration](docs/experiment.md#training-configuration) |
| Analysis | Learning curves, class-distribution analysis and review of the saved evaluation outputs. | [Findings](docs/experiment.md#recorded-results) |

## Recorded experiment

| Setting or result | Value |
| --- | --- |
| Images used | **22,968 training · 1,432 validation** |
| Training setup | 50 epochs · batch 64 · Adam, learning rate **0.0001** |
| Best validation accuracy | **65.22%**, epoch 50 |
| Validation cross-entropy at epoch 50 | **0.9432** |

![Validation accuracy and cross-entropy over the 50 recorded epochs.](assets/validation-curves.svg)

**Figure 3.** Validation metrics extracted from the saved epoch logs. The validation subset was monitored throughout training and used for checkpoint selection; **65.22% is a validation score**. A separate final test evaluation is not recorded.

The saved confusion matrix needs to be recomputed with aligned prediction and label order before drawing conclusions about individual classes. [Protocol, interpretation and evaluation details →](docs/experiment.md)

## Code and documentation

| Resource | Contents |
| --- | --- |
| [Original notebook](notebooks/zavrsni.ipynb) · [Cell guide](notebooks/README.md) | Implementation, model summary, training logs and original plots. |
| [Data](docs/data.md) · [Architecture](docs/architecture.md) | Dataset examples, class balance, preprocessing and exact layer dimensions. |
| [Experiment](docs/experiment.md) · [Results](results/README.md) | Hyperparameters, metric tables, interpretation and extracted CSV/JSON evidence. |
| [Execution notes](docs/reproduction.md) · [Sources](docs/evidence.md) | Environment, known execution issues, figure-generation commands and evidence references. |

*Bachelor's thesis, University of Zagreb, Faculty of Electrical Engineering and Computing (FER), June 2025. Supervisor: Prof. Sonja Grgić, PhD.*
