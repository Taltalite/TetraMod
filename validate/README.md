# TetraMod Train Validation Tools

This directory contains the `../bonito-mixed/validate` scripts that are relevant
to models produced by `tetramod train`.

## Scripts

- `evaluate_train_mod.py`: main checkpoint evaluator. It reports basecalling
  accuracy, modification-head metrics, target-axis projection coverage, TSV
  examples, JSON/text summaries, and optional plots.
- `check_base_mod_alignment.py`: diagnostic checker for shared base/mod time
  axes, decoded base emission counts, and target-axis modification projection
  coverage.
- `kmer_balanced_mod_report.py`: post-processes `mod_site_examples.tsv` from
  `evaluate_train_mod.py` into k-mer balanced binary modification metrics.

The original `bonito-mixed/validate` directory also contains basecaller output
debugging, POD5 inference, BAM comparison, and visualization scripts. Those are
not migrated here because they validate `basecaller_mod` outputs rather than
the training checkpoint evaluation path.

## Bonito Dependency Boundary

These tools use TetraMod modules for the migrated modified-base training path:

- `tetramod.train_mod_data.load_train_mod_data`
- `tetramod.util.load_model`

They still import upstream Bonito package APIs for unmodified helpers such as
`bonito.data` settings and base-only `load_data`.
