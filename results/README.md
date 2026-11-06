# Experiment results

The files below contain the layer dimensions, image counts and metrics used in the project figures. They come from the definitions and saved outputs in the [training notebook](../notebooks/zavrsni.ipynb).

| File | Contents |
| --- | --- |
| [epoch_metrics.csv](epoch_metrics.csv) | Training and validation accuracy/loss for 50 epochs |
| [class_counts.csv](class_counts.csv) | Image counts in the complete FER2013 training and test directories |
| [architecture.csv](architecture.csv) | Output shapes and parameter counts for 23 layers |
| [evidence.json](evidence.json) | Dataset subsets, parameter totals and the training-history summary |

The epoch CSV follows the console logs; the JSON contains the printed history summary. Their final training values differ, as described in [recorded results](../docs/experiment.md#recorded-results).

[Regeneration commands](../docs/reproduction.md#rebuild-the-figures) · [Dataset](../docs/data.md) · [Overview](../README.md)
