# Original notebook

[Open zavrsni.ipynb](zavrsni.ipynb)

This is the original thesis notebook, with its source, saved outputs, execution counters and metadata preserved. Code comments and some printed text are in Croatian. The surrounding repository documentation is in English.

## Cell map

The index is the position in the notebook's JSON `cells` array, starting at zero. It does not refer to the execution number displayed by Jupyter.

| Index | Contents | Saved evidence |
| --- | --- | --- |
| 0 | Package installation and imports | Observed package versions |
| 1 | FER2013 download and directory listing | Class counts in `train/` and `test/` |
| 2 | Training and validation image generators | 22,968 training and 1,432 validation images |
| 3 | CNN definition and compilation | Model summary, shapes and parameter counts |
| 4 | Checkpoint callback and `model.fit` | Fifty epochs of console logs; current source starts with a malformed import |
| 5 | Training and validation loss plot | Saved loss figure |
| 6 | Training and validation accuracy plot | Saved accuracy figure |
| 7 | Combined plots and printed history summary | Final/best metrics and epoch 50 |
| 8 | Predictions and confusion matrix | Saved matrix; ordering issue described in the experiment document |
| 9 | Empty cell | — |
| 10 | Check for saved model and attempted loading | Message that the model file is absent |

The current notebook is an experimental record, with execution counters that do not form a clean top-to-bottom sequence. See [reproduction notes](../docs/reproduction.md) before running it; the original has not been modified to conceal execution or evaluation issues.

[Architecture](../docs/architecture.md) · [Experiment and results](../docs/experiment.md) · [Back to overview](../README.md)
