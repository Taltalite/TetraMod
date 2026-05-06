# `create_dataset_bam_aligned_highmod.py` 命令模板指南

用途：为 high-mod control / ratio-IVT 样本生成 TetraMod chunk 数据集。该脚本信任整条 read 的 BAM CIGAR 投影，不做 chunk-local remapping，适合 chunk-local remapping 在高修饰样本上不稳定时作为 rescue path。

## 输入约定

- `BAM`：Dorado 产生的 mapped BAM，需包含 move table，例如 basecalling/alignment 时使用 `--reference` 和 `--emit-moves`。
- `POD5_DIR`：对应原始 POD5 目录。
- `REF_FASTA`：BAM alignment 使用的同一参考序列。
- `OUT_DIR`：输出 chunk 数据集目录。
- `RUN_ID`：样本/批次名；不传时脚本会尝试使用 BAM `RG` tag，否则写为 `unknown`。

## RNA 数据模板

适用于 RNA direct signal，尤其是 RNA002 m6A control / ratio-IVT 数据。

```bash
BAM=/path/to/sample.sorted.bam
POD5_DIR=/path/to/pod5
REF_FASTA=/path/to/reference.fasta
OUT_DIR=/path/to/chunks/sample_rna
RUN_ID=sample_rna

python gen_data/create_dataset_bam_aligned_highmod.py \
    --bam-file "$BAM" \
    --pod5-dir "$POD5_DIR" \
    --reference-fasta "$REF_FASTA" \
    --output-dir "$OUT_DIR" \
    --run-id "$RUN_ID" \
    --sample-type rna \
    --rna002 \
    --chunk-len 10000 \
    --overlap 500 \
    --filter-preset relaxed \
    --workers 8
```

### RNA ratio-IVT 批量模板

```bash
REF_FASTA=/path/to/reference.fasta
POD5_ROOT=/path/to/pod5
BAM_ROOT=/path/to/bam
OUT_ROOT=/path/to/chunks

for RATIO in 12p5 25 50 75; do
python gen_data/create_dataset_bam_aligned_highmod.py \
    --bam-file "$BAM_ROOT/cc${RATIO}.sorted.bam" \
    --pod5-dir "$POD5_ROOT/cc${RATIO}" \
    --reference-fasta "$REF_FASTA" \
    --output-dir "$OUT_ROOT/mod${RATIO}" \
    --run-id "rna002_cc${RATIO}" \
    --sample-type rna \
    --rna002 \
    --chunk-len 10000 \
    --overlap 500 \
    --filter-preset relaxed \
    --workers 8
done
```

## DNA 数据模板

适用于 DNA signal。通常不加 `--rna002`，让脚本默认使用 BAM 中的 Dorado scaling tag 做归一化；如果你明确要复用某个模型配置，请使用 `--model-config` 或显式设置 `--norm-strategy`。

```bash
BAM=/path/to/sample.sorted.bam
POD5_DIR=/path/to/pod5
REF_FASTA=/path/to/reference.fasta
OUT_DIR=/path/to/chunks/sample_dna
RUN_ID=sample_dna

python gen_data/create_dataset_bam_aligned_highmod.py \
    --bam-file "$BAM" \
    --pod5-dir "$POD5_DIR" \
    --reference-fasta "$REF_FASTA" \
    --output-dir "$OUT_DIR" \
    --run-id "$RUN_ID" \
    --sample-type dna \
    --chunk-len 10000 \
    --overlap 500 \
    --filter-preset relaxed \
    --workers 8
```

## 小规模检查模板

先限制 records/chunks，确认路径、POD5 匹配、BAM tags 和输出格式正常。

```bash
python gen_data/create_dataset_bam_aligned_highmod.py \
    --bam-file "$BAM" \
    --pod5-dir "$POD5_DIR" \
    --reference-fasta "$REF_FASTA" \
    --output-dir "$OUT_DIR.debug" \
    --run-id "$RUN_ID.debug" \
    --sample-type rna \
    --rna002 \
    --chunk-len 10000 \
    --overlap 500 \
    --filter-preset relaxed \
    --max-records 1000 \
    --max-chunks 200 \
    --workers 2
```

DNA 检查时把 `--sample-type rna --rna002` 换成 `--sample-type dna`。

## 参数说明

| 参数 | 简要说明 |
|---|---|
| `--bam-file` | 必填。mapped Dorado BAM，需有 `mv`/`ns` tags，并建议有 `NM`、`RG`、`qs`、`sm`、`sd` 等 tags。 |
| `--pod5-dir` | 必填。BAM reads 对应的 POD5 目录。 |
| `--reference-fasta` | 必填。BAM alignment 使用的参考 FASTA。 |
| `--output-dir` | 必填。输出 dataset 目录。 |
| `--run-id` | 可选。覆盖输出 metadata 中的 run id；不传则用 BAM `RG` tag 或 `unknown`。 |
| `--sample-type` | `rna` 或 `dna`。决定 read/query 与 signal 顺序的处理方式。 |
| `--chunk-len` | 每个 signal chunk 的长度，默认 `10000`。 |
| `--overlap` | 相邻 chunk 的 overlap，默认 `500`；必须小于 `--chunk-len`。 |
| `--max-label-len` | 限制 reference label 最大长度；不传则不额外限制。 |
| `--max-records` | 最多处理多少条 BAM record；默认 `-1` 表示不限制。 |
| `--max-chunks` | 最多写出多少个 chunk；默认 `-1` 表示不限制。 |
| `--workers` | 并行 worker 数；脚本会按本机 CPU 数再做上限裁剪。 |
| `--clip-value` | signal 归一化后裁剪范围，默认 `5.0`。 |
| `--filter-preset` | 过滤预设：`relaxed` 或 `strict`；high-mod rescue 通常先用 `relaxed`。 |
| `--min-read-identity` | read-level identity 下限；不传时由 `--filter-preset` 决定。 |
| `--min-read-aligned-fraction` | read 中已比对部分比例下限；不传时由 preset 决定。 |
| `--min-mapq` | BAM mapping quality 下限；不传时由 preset 决定。 |
| `--min-chunk-aligned-fraction` | chunk 内 query bases 可投影到 reference 的比例下限；不传时由 preset 决定。 |
| `--min-chunk-base-identity` | chunk 内 query/reference base identity 下限；不传时由 preset 决定。 |
| `--min-reference-len` | chunk 投影出的 reference 区间最小长度，默认 `25`。 |
| `--max-reference-span-factor` | reference span 相对 chunk query span 的最大倍数，默认 `2.5`。 |
| `--require-a` | 默认开启。只保留 reference label 中含 A 的 chunk，适合 m6A A-head 数据。 |
| `--allow-no-a` | 关闭 `--require-a`，允许无 A chunk。 |
| `--min-qscore` | 按 Dorado `qs` tag 过滤 read；不传则不按 qscore 过滤。 |
| `--rna002` | 使用 RNA002 兼容的模型配置默认归一化；RNA002 direct RNA 数据通常需要。 |
| `--model-config` | 指定模型 `config.toml`，用于读取 scaling/normalisation 参数。 |
| `--norm-strategy` | 信号归一化策略：`from-bam`、`pa`、`quantile` 或 `model-config`。不传时：有 `--rna002` 用 `model-config`，否则用 `from-bam`。 |
| `--pa-mean` | `--norm-strategy pa` 时使用的 pA mean，默认 `0.0`。 |
| `--pa-std` | `--norm-strategy pa` 时使用的 pA std，默认 `1.0`。 |
| `--quantile-a` | `quantile` 归一化的低分位点参数。 |
| `--quantile-b` | `quantile` 归一化的高分位点参数。 |
| `--shift-multiplier` | `quantile`/`model-config` 归一化的 shift multiplier。 |
| `--scale-multiplier` | `quantile`/`model-config` 归一化的 scale multiplier。 |
| `--metadata-kmer` | metadata 中 primary A 位点上下文 k-mer 长度，默认 `5`；必须为正奇数。 |
| `--seed` | 随机种子，默认 `1`。 |
| `--task-batch-size` | 每批提交给 worker 的 task 数；通常不用改。 |
| `--max-pending-batches` | 同时排队等待的 worker batch 上限；内存压力大时可调低。 |
| `--max-samples-per-worker-file` | 每个 worker 临时文件最多写入样本数；通常不用改。 |
| `--progress-log-interval` | 每写出多少 chunk 打印一次进度；通常不用改。 |
| `--mp-start-method` | multiprocessing 启动方式：`auto`、`fork`、`spawn`、`forkserver`；Linux/WSL 默认 `auto` 通常即可。 |

## 过滤预设

| preset | read identity | read aligned fraction | MAPQ | chunk aligned fraction | chunk base identity |
|---|---:|---:|---:|---:|---:|
| `relaxed` | `0.70` | `0.75` | `1` | `0.70` | `0.60` |
| `strict` | `0.80` | `0.85` | `20` | `0.85` | `0.70` |

## 期望输出

`OUT_DIR` 下应至少包含最终 chunk arrays、`dataset_summary.json` 和 `metadata_fields.json`。生成后先查看：

```bash
cat "$OUT_DIR/dataset_summary.json"
```

重点检查 `total_written`、`filter_preset`、各类 reject counters、`norm_strategy`、`sample_type` 和 `require_a`。
