# Evidence and sources

The public description is grounded in [notebooks/zavrsni.ipynb](../notebooks/zavrsni.ipynb), its saved outputs, and Petar Majić's bachelor's thesis, *Prepoznavanje emocija iz slika lica uporabom konvolucijskih neuronskih mreža*, University of Zagreb, Faculty of Electrical Engineering and Computing, June 2025. The Croatian thesis was consulted as background; it is not included in this repository.

The notebook is preserved byte-for-byte. Its SHA-256 is:

```text
d8622bd1896a88b2bf637dbba14627f9a3cdd29b2ac6a18b24f703214e401f66
```

Cell references use zero-based JSON indices, independent of execution counters. Several execution counters are missing or out of sequence, so they are not used as identifiers.

## Claim-to-source map

| Claim | Source | How it is used |
| --- | --- | --- |
| Dataset identifier | Cell 1, `kagglehub.dataset_download` | Identifies the directory-based FER2013 release. |
| Source directory counts | Cell 1, saved `os.walk` output | Extracted to `results/class_counts.csv`; the first five filenames plus each remaining-file count give class totals. |
| Actual generator sizes | Cell 2, saved `Found ... images` lines | 22,968 training; 1,432 validation. |
| Preprocessing and augmentation | Cell 2, generator definitions | Grayscale, 48 × 48, rescaling, shifts and horizontal flip. |
| Six convolutional layers | Cell 3, `model.add` calls | Grouped into three feature blocks for the diagram. |
| Shapes and parameter totals | Cell 3, definitions and saved model summary | Derived arithmetically without TensorFlow; totals agree with the saved summary. |
| Optimiser and learning rate | Cell 3, `model.compile` | Adam, learning rate 0.0001. |
| Fifty epochs and checkpoint rule | Cell 4, source and logs | Best weights selected by validation accuracy. |
| Validation curves | Cell 4, fifty saved epoch-end log lines | Rounded values in `results/epoch_metrics.csv`. |
| Final and best metrics | Cell 7, printed `history.history` summary | Best validation accuracy 0.6522 at epoch 50; validation loss 0.9432. |
| Confusion matrix | Cell 8, code and saved image | Retained as original output; excluded from performance conclusions because ordering is not aligned. |
| Observed packages | Cell 0, saved install output | Relevant versions listed in `requirements-observed.txt`. |
| Checkpoint availability | Cell 10, saved existence-check message; reviewed attachments | No trained weights supplied. |

## Differences resolved during documentation

| Topic | Thesis text or other saved record | Implementation / treatment here |
| --- | --- | --- |
| Block count | Text describes five convolutional blocks. | Code has six convolutions and three pooling stages; the diagram groups them into three two-convolution blocks. |
| Parameter count | Text reports 1,491,905 total and 1,485,061 trainable. | Code and saved summary give **1,394,183 total** and **1,391,943 trainable**. |
| Learning rate | Text reports 0.001. | The compile call sets **0.0001**. |
| Dataset size versus used subset | Dataset discussion describes 28,709 training / 7,178 test images. | These are source directory totals; generator outputs confirm **22,968 / 1,432** images actually used. |
| Test accuracy wording | Thesis describes approximately 65% test accuracy. | The saved run reports **validation accuracy**, used for checkpoint selection. |
| Best epoch | Text places the maximum around epoch 40. | Saved history summary and checkpoint log place the best recorded score at **epoch 50**. |
| Training metrics | History summary: accuracy 0.6368, loss 0.9747. Console's final line: 0.6327, 0.9799. | Both are retained as different saved records. The new figure plots only the validation series, which agree to four decimal places. |
| Per-class conclusions | Text interprets the saved confusion matrix as class-specific performance. | The prediction/label ordering issue prevents that interpretation. No corrected metrics have been invented. |
| Execution host | Thesis describes Colab; metadata mentions Colab; paths use `/kaggle/input`. | The exact host and hardware are left unspecified. |

The original notebook also contains a malformed import and a weights-loading mismatch, documented in [execution notes](reproduction.md#before-executing-the-training-notebook). Preserving the source does not imply that its current cells have been verified by a fresh training run.

## Derived documentation assets

| File | Source and transformation |
| --- | --- |
| `assets/model-overview.svg` | Model definitions and verified layer shapes in cell 3; schematic tensor volumes and classification head, with an illustrative image excerpt from thesis Figure 3.2. |
| `assets/dataset-examples.svg` | Seven image excerpts from the supplied training-example montage; English labels follow the source figure's class headings. |
| `assets/source/fer2013-training-examples.png` | Unmodified original `slike/slika4_train.png` from the supplied thesis attachments; see [image provenance](../assets/source/README.md). |
| `assets/validation-curves.svg` | Cell 4 epoch logs → `results/epoch_metrics.csv` → two validation plots. |
| `assets/class-distribution.svg` | Cell 1 listing → `results/class_counts.csv` → full training-directory counts. |
| `results/architecture.csv` | Shape and parameter arithmetic from `model.add` definitions. |
| `results/evidence.json` | Source hash, sizes, parameter totals and saved summary values. |

The extraction and plotting scripts were added for documentation. They are not part of the original thesis training implementation. `scripts/draw_architecture.py` draws the architecture from the extracted layer table and crops the supplied image montage for illustration. It does not fabricate feature maps or class scores. No model was retrained, no new predictions were generated, and no new model comparison was performed.

## Technical references

- [FER2013 directory dataset](https://www.kaggle.com/datasets/msambare/fer2013): dataset used by the code.
- [Keras 3.8.0 image preprocessing source](https://github.com/keras-team/keras/blob/v3.8.0/keras/src/legacy/preprocessing/image.py): directory subsets, label ordering and shuffle defaults.
- [TensorFlow ImageDataGenerator API](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator): generator arguments.
- [Keras FAQ: training and validation loss](https://keras.io/getting_started/faq/#why-is-my-training-loss-much-higher-than-my-testing-loss): effect of training-only regularisation and augmentation.
- [Keras FAQ: model and weights saving](https://keras.io/getting_started/faq/#what-are-my-options-for-saving-models): distinction between whole-model and weights-only loading.

[Back to overview](../README.md)
