# TetraMod Train Validation Tools

This directory contains the `../bonito-mixed/validate` scripts that are relevant
to models produced by `tetramod train`.

## Scripts

- `evaluate_train_mod.py`: main checkpoint evaluator. It reports basecalling
  accuracy, modification-head metrics, target-axis projection coverage, TSV
  examples, JSON/text summaries, and optional plots.
- `evaluate_promote_control.py`: small control-warmup evaluator for `train_promote`
  and baseline checkpoints. It checks whether the A-head separates full-mod and
  IVT, and whether mixed-ratio datasets show monotonic mean predicted m6A
  probability.
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
- `check_gold_coordinate_conventions.py`: sweeps gold-site coordinate shifts
  and strand interpretations to diagnose 0/1-based or strand convention
  mismatches before interpreting poor gold-site metrics as model failure.

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
`threshold_sweep.tsv`, `summary.json`, `summary.txt`, and PNG plots when
matplotlib is available. The threshold sweep reports TP/TN/FP/FN, FPR,
specificity, precision, recall, FDR, and F1 from score thresholds 0.00 to 1.00.
The plot set includes the full ROC curve and a low-FPR ROC view.

To check coordinate and strand conventions, run:

```bash
python validate/check_gold_coordinate_conventions.py \
  --bam tetramod_basecaller_test_fix.bam \
  --gold-bed m6A_gold.hg38.bed \
  --reference hg38.fa \
  --output-dir val_res/m6A_gold_convention_check \
  --mod-code a \
  --canonical-base A \
  --min-coverage 5
```

The diagnostic writes `coordinate_convention_summary.tsv` and `summary.json`.
If a shifted or flipped convention shows a large jump in covered gold sites,
reference-base compatibility, ROC AUC, PR AUC, or Top-N gold fraction, the gold
coordinate/strand convention should be corrected before drawing model
conclusions.

The original `bonito-mixed/validate` directory also contains basecaller output
debugging, POD5 inference, BAM comparison, and visualization scripts. Those are
not migrated here because they are broader one-off debugging tools rather than
the focused checkpoint and BAM comparison paths kept here.

## Promoted Control Evaluation

Example:

```bash
python validate/evaluate_promote_control.py \
  /path/to/model_dir \
  --ivt-dir /path/to/ivt_dataset \
  --mix-dataset 25:/path/to/mix25_dataset \
  --mix-dataset 50:/path/to/mix50_dataset \
  --mix-dataset 75:/path/to/mix75_dataset \
  --full-mod-dir /path/to/full_mod_dataset \
  --dataset valid \
  --valid-chunks 2000 \
  --batchsize 32 \
  --device cuda:0
```

The evaluator writes:

- `dataset_metrics.tsv`
- `summary.json`
- `summary.txt`

This path is intentionally small. It is designed to answer whether a checkpoint
learned the control warm-up distinction before LLP or MIL are introduced.

## Bonito Dependency Boundary

These tools use TetraMod modules for the migrated modified-base training path:

- `tetramod.train_mod_data.load_train_mod_data`
- `tetramod.util.load_model`

They still import upstream Bonito package APIs for unmodified helpers such as
`bonito.data` settings and base-only `load_data`.
