# Training notebook

[Open zavrsni.ipynb](zavrsni.ipynb)

The notebook follows the experiment from loading FER2013 images to building and training the CNN. It includes the model summary, 50 epochs of logs, learning curves and the printed results.

| Section | Contents |
| --- | --- |
| Environment | Package installation and imports |
| Dataset | KaggleHub download and class-directory counts |
| Preprocessing | Grayscale generators, rescaling, shifts, flips and subset selection |
| Model | Six convolutional layers, pooling, Batch Normalization, Dropout and dense classification |
| Training | Checkpoint callback and 50 epochs of logs |
| Analysis | Learning curves, history summary and confusion matrix |
| Checkpoint | Weights-file check and loading code |

[Architecture](../docs/architecture.md) · [Experiment](../docs/experiment.md) · [Execution setup](../docs/reproduction.md)
