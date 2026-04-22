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
- `compare_basecaller_bams.py`: compares `tetramod basecaller` and `bonito
  basecaller` BAM/SAM outputs on the intersection of read ids.
- `evaluate_modbam_gold_sites.py`: aggregates MM/ML modified-base calls from an
  aligned modBAM into site-level m6A scores and compares them with gold BED or
  m6A-Atlas-style site tables.

## m6A Gold Site Evaluation

Example:

```bash
python validate/evaluate_modbam_gold_sites.py \
  --bam tetramod_basecaller_test_fix.bam \
  --gold-bed m6A_gold.hg38.bed \
  --reference hg38.fa \
  --output-dir val_res/m6A_gold_eval \
  --mod-code a \
  --canonical-base A \
  --min-coverage 5 \
  --prob-threshold 0.5
```

The evaluator writes `site_level_predictions.tsv`, positive/negative site TSVs,
`summary.json`, `summary.txt`, and PNG plots when matplotlib is available.

The original `bonito-mixed/validate` directory also contains basecaller output
debugging, POD5 inference, BAM comparison, and visualization scripts. Those are
not migrated here because they are broader one-off debugging tools rather than
the focused checkpoint and BAM comparison paths kept here.

## Bonito Dependency Boundary

These tools use TetraMod modules for the migrated modified-base training path:

- `tetramod.train_mod_data.load_train_mod_data`
- `tetramod.util.load_model`

They still import upstream Bonito package APIs for unmodified helpers such as
`bonito.data` settings and base-only `load_data`.
