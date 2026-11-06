# Experiment and results

The [CNN](architecture.md) was trained for 50 epochs to classify seven FER2013 expression labels from **48 × 48 grayscale images**. Training combines image augmentation, Batch Normalization and Dropout.

## Training configuration

| Setting | Value |
| --- | --- |
| Training / validation images | **22,968 / 1,432** |
| Input | 48 × 48 × 1, rescaled by `1./255` |
| Labels | Seven classes, categorical encoding |
| Augmentation | Width/height shifts up to 10%; horizontal flip |
| Batch size | 64 |
| Optimiser | Adam |
| Learning rate | **0.0001** |
| Objective | Categorical cross-entropy |
| Epochs | 50 |
| Training / validation batches | 359 / 23 |
| Checkpoint | `model.weights.h5`, best validation accuracy |
| Early stopping / LR schedule | Not configured |
| Random seed | Not set |

The validation generator selects 20% of the source `test/` directory. It is monitored during training and used for checkpoint selection. See [data preparation](data.md#the-split-actually-used).

## Recorded results

| Metric | Value |
| --- | ---: |
| Best validation accuracy | **65.22%** |
| Best epoch | **50** |
| Validation cross-entropy at epoch 50 | **0.9432** |
| Final training accuracy in the history summary | 63.68% |
| Final training loss in the history summary | 0.9747 |

The values above follow the printed `history.history` summary. The checkpoint log reports the same validation accuracy at higher precision: `0.65223`.

![Validation accuracy and cross-entropy over 50 epochs.](../assets/validation-curves.svg)

**Figure 1.** Validation accuracy increases from **21.30% to 65.22%**, while cross-entropy decreases from **1.8888 to 0.9432**. The best values occur at the final recorded epoch. The plotted values are in [epoch_metrics.csv](../results/epoch_metrics.csv).

The CSV follows the epoch logs. Its final training accuracy and loss are **0.6327** and **0.9799**, while the history summary reports **0.6368** and **0.9747**. The figure uses the validation series, which agree across both records.

## Interpretation

The validation curves continue to improve late in the run, although the changes become smaller. The final epoch gives both the highest validation accuracy and the lowest validation loss.

Training applies random image augmentation and active Dropout. Validation uses rescaled images without augmentation, and Dropout is disabled. The two scores are therefore calculated under different conditions, which helps explain why validation accuracy can exceed training accuracy. [Keras explanation](https://keras.io/getting_started/faq/#why-is-my-training-loss-much-higher-than-my-testing-loss).

The experiment contains one training run. The reported score is validation accuracy used for model selection; no separate final test score or controlled architecture comparison is available.

## Confusion-matrix ordering

The notebook creates its confusion matrix with:

```python
validation_labels = validation_generator.classes
validation_pred_probs = model.predict(validation_generator)
validation_pred_labels = np.argmax(validation_pred_probs, axis=1)
confusion_mtx = confusion_matrix(validation_labels, validation_pred_labels)
```

The validation generator uses the default `shuffle=True`. The predictions can therefore arrive in a different order from the fixed `.classes` vector. Comparing them directly pairs predictions with the wrong labels, so the saved matrix cannot support per-class precision, recall or F1.

The accuracy reported by `model.fit()` uses correctly paired batch labels. The ordering error arises later when the notebook builds a confusion matrix from separate predictions.

A correct matrix requires an unshuffled prediction iterator, its matching labels and the trained checkpoint. The checkpoint is not included in the repository.

[Results files](../results/README.md) · [Execution](reproduction.md) · [Overview](../README.md)
