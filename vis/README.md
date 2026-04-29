# Evaluation Visualization

Plot Stage 1 vs Stage 2 promoted evaluation outputs produced by:

- `validate/evaluate_llp_bags.py`
- `validate/evaluate_promote_control.py`

Default usage with the current repository result layout:

```bash
python vis/plot_eval_results.py \
  --val-res val_res \
  --output-dir vis_out
```

The script writes PNG figures and compact TSV summaries to `--output-dir`.

Generated figures:

- `evaluation_summary_dashboard.png`
- `llp_ratio_calibration.png`
- `llp_error_by_ratio.png`
- `llp_bag_score_distributions.png`
- `llp_paired_bag_delta.png`
- `control_mean_probabilities.png`
- `control_threshold_metrics.png`

If result directories use different names, pass them explicitly:

```bash
python vis/plot_eval_results.py \
  --stage1-llp-dir /path/to/stage1_llp_eval \
  --stage2-llp-dir /path/to/stage2_llp_eval \
  --stage1-control-dir /path/to/stage1_control_eval \
  --stage2-control-dir /path/to/stage2_control_eval \
  --output-dir /path/to/figures
```
