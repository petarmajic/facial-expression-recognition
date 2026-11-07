# Experiment results

I used the model definition and saved outputs in the [training notebook](../notebooks/zavrsni.ipynb) to build the figures. The four files below let you trace a number in a figure back to its source.

| File | What it records |
| --- | --- |
| [epoch_metrics.csv](epoch_metrics.csv) | Training and validation accuracy and loss from the console log, one row for each of the 50 epochs. |
| [class_counts.csv](class_counts.csv) | Image counts for each expression in the full training and test directories, before subset selection. |
| [architecture.csv](architecture.csv) | Output shapes and parameter counts for all 23 layers in the model definition. |
| [evidence.json](evidence.json) | The subset sizes reported by the generators, the model’s parameter totals and the printed training-history summary. |

At epoch 50, the console printed a training accuracy of 0.6327 and the history summary printed 0.6368. Both report the same validation accuracy of 0.6522. The validation curve uses the epoch log. [Comparison of the two records](../docs/experiment.md#recorded-results).

[Regeneration commands](../docs/reproduction.md#rebuild-the-figures) · [Dataset](../docs/data.md) · [Overview](../README.md)
