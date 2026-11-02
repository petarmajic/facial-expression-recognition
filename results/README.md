# Extracted experiment evidence

These files are derived from the original [notebook](../notebooks/zavrsni.ipynb). They make the values behind the documentation easy to inspect. They are not new training, inference or dataset-scan results.

| File | Contents | Source |
| --- | --- | --- |
| [epoch_metrics.csv](epoch_metrics.csv) | 50 epoch rows: console training accuracy/loss and validation accuracy/loss | Cell 4 saved text output, four-decimal precision |
| [class_counts.csv](class_counts.csv) | Seven classes, with counts in the full source `train/` and `test/` directories | Cell 1 saved listing |
| [architecture.csv](architecture.csv) | Layer type, details, output shape, parameter count and trainable count | Cell 3 layer definitions; totals checked against saved summary |
| [evidence.json](evidence.json) | Notebook SHA-256, actual generator sizes, parameter totals, final history summary and source notes | Cells 2, 3 and 7 |

The `console_training_*` columns are labelled explicitly because the final printed console values differ from the later `history.history` summary. The new validation chart uses only the validation columns; see [recorded results](../docs/experiment.md#recorded-results).

Class counts refer to the complete source directories. They should not be confused with the smaller subsets actually used by the generators. The [data guide](../docs/data.md) explains that distinction.

## Regeneration

```bash
python scripts/extract_evidence.py
python scripts/make_figures.py
```

Run these commands from the repository root. The first script uses only the standard library; the second requires Matplotlib. Neither script executes the model. See [setup](../docs/reproduction.md#rebuild-the-documentation-figures) and the [source ledger](../docs/evidence.md).
