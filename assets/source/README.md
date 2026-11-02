# FER2013 illustration source

`fer2013-training-examples.png` is the original 2,058 × 593 image supplied with the thesis materials as `slike/slika4_train.png`. It appears as Figure 3.2, “Uzorci slika iz train skupa za svaku emociju”, in Petar Majić's bachelor's thesis (FER, June 2025). The thesis credits the [FER2013 dataset on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013).

The montage is retained unmodified. The plotting script extracts the top-row image from each class and replaces the surrounding Croatian headings with English labels in the new figure. The individual source dataset filenames are not recorded in the supplied montage.

| Derived figure | Use of the original montage |
| --- | --- |
| `assets/model-overview.svg` | The top-row `happy` example, cropped at `(904, 78, 1154, 328)`, illustrates the model input. It is not a prediction or a new inference result. |
| `assets/dataset-examples.svg` | Seven top-row examples, one per class, reordered to the generator's alphabetical class order. |

Crop coordinates use Pillow's `(left, upper, right, lower)` convention. For each source column, `left = 10 + 298 × column`, `upper = 78`, `right = left + 250`, and `lower = 328`. Source column order is angry, disgust, fear, happy, sad, surprise, neutral.

The image excerpts are illustrations from the thesis figure, not original-resolution dataset files. The **48 × 48 × 1** label in the architecture describes the input tensor used by the implementation. The montage and thumbnails do not demonstrate that a particular example was selected by the recorded training generator.

The tensor volumes and neuron glyphs in the architecture are schematic. They do not show measured feature activations or confidence values. Output labels show the seven classes; no prediction probabilities have been invented.

[Dataset documentation](../../docs/data.md) · [Evidence and sources](../../docs/evidence.md)
