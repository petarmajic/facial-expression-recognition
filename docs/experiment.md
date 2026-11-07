# Experiment and results

I trained the [CNN](architecture.md) for 50 epochs on **48 × 48 grayscale images** from FER2013. It predicted one of seven expression labels. I used shifts and flips to augment the training images and applied Batch Normalization and Dropout inside the network.

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

I used 20% of the source `test/` directory for validation. At the end of each epoch, Keras evaluated the same subset and saved the weights if validation accuracy had improved. [How the generators selected the images](data.md#the-split-actually-used).

## Recorded results

| Metric | Value |
| --- | ---: |
| Best validation accuracy | **65.22%** |
| Best epoch | **50** |
| Validation cross-entropy at epoch 50 | **0.9432** |
| Final training accuracy in the history summary | 63.68% |
| Final training loss in the history summary | 0.9747 |

I took the final training values from the printed `history.history` summary. The checkpoint log gives the best validation accuracy to five decimal places (`0.65223`).

![Validation accuracy and cross-entropy over 50 epochs.](../assets/validation-curves.svg)

**Figure 1.** Validation accuracy rose from **21.30% to 65.22%** and cross-entropy fell from **1.8888 to 0.9432** between the first and last epochs. [Values plotted from the epoch logs](../results/epoch_metrics.csv).

The two records agree on the validation metrics. For the final training epoch, the console printed an accuracy of **0.6327** and a loss of **0.9799**. The history summary printed **0.6368** and **0.9747**.

## Interpretation

Validation accuracy was still improving late in the run. Epoch 50 had both the highest validation accuracy and the lowest validation loss. The same validation subset determined which weights to save. This was a single run with no separate final test evaluation or controlled architecture comparison.

The validation accuracy also finished above the training accuracy. Keras applied random image shifts, flips and Dropout during training. During validation it used rescaled images and disabled Dropout, so the two scores reflect different conditions. [Keras explanation](https://keras.io/getting_started/faq/#why-is-my-training-loss-much-higher-than-my-testing-loss).

## Confusion-matrix ordering

I checked how the notebook pairs each predicted class with its label when it builds the confusion matrix.

```python
validation_labels = validation_generator.classes
validation_pred_probs = model.predict(validation_generator)
validation_pred_labels = np.argmax(validation_pred_probs, axis=1)
confusion_mtx = confusion_matrix(validation_labels, validation_pred_labels)
```

The generator defaults to `shuffle=True`, so prediction order can differ from the directory order in `.classes`. That mismatch makes the saved matrix unreliable for per-class precision, recall or F1. The validation accuracy from `model.fit()` uses the labels that accompany each batch and is unaffected by this later step.

To recompute the matrix, I would load the trained checkpoint and predict with `shuffle=False` so the predictions line up with their labels. The checkpoint is absent from the repository.

[Results files](../results/README.md) · [Execution](reproduction.md) · [Overview](../README.md)
