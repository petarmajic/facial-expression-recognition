# Training notebook

I used this notebook to load FER2013, build the CNN in Keras and train it for 50 epochs. The saved output includes the model summary and the training log for each epoch. [Open the notebook](zavrsni.ipynb).

| To inspect | Look for |
| --- | --- |
| Environment and images | Package installation and imports, then the KaggleHub download and file counts for each class. |
| Preprocessing | The two grayscale image generators. Both rescale the pixels. Only the training generator shifts and flips images. |
| Architecture | The six convolutional layers, pooling, Batch Normalization, Dropout and dense classifier in the model definition. |
| Training | The checkpoint callback and 50 epochs of logs. The final epoch recorded the highest validation accuracy. |
| Analysis | The learning curves and printed history summary. The [experiment notes](../docs/experiment.md#confusion-matrix-ordering) explain the ordering problem in the saved confusion matrix. |
| Checkpoint | The final cell checks for a local weights file. Follow the loading instructions below before running it. |

To rerun the experiment, start with the setup notes for the dataset paths, damaged callback import and weights loader. [Execution setup](../docs/reproduction.md#training-notebook-setup).

[Architecture](../docs/architecture.md) · [Experiment](../docs/experiment.md)
