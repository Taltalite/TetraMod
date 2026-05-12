# mAFiA RNA002 合成数据 Stage1 使用流程

本文档记录如何把 mAFiA RNA002 PRJEB74106 中的合成 oligo 数据处理成
`train_promote --promote-stage control` 可用的 Stage1 监督训练数据。

核心原则：

- Stage1 只使用可准确定位 m6A 位点的 synthetic oligo control 数据。
- 只标注合成 oligo 中设计好的中心 DRACH A 位点。
- unmodified control 的中心 A 标为 `0`，modified control 的中心 m6A 标为 `4`。
- 其他所有 base 位置都标为 `-100`，训练时忽略。
- HEK293 WT/IVT ratio 数据不要混入 Stage1 监督 warm-up。
- TEST1/TEST2 或 mixed run 默认作为后续验证/诊断数据，不作为 Stage1 训练主数据。

## 0. 推荐目录结构

远程服务器上建议使用环境变量管理路径，避免把机器路径写死进脚本或配置。

```bash
export MAFIA_ROOT=/data/biolab-backup-hdd2/public_data/mAFia_RNA002_PRJEB74106/HEK293
export WORK_ROOT=/data/biolab-nvme-pcie2/lijy/tetramod_mafia_rna002
export REPO=/home/lijy/TetraMod

mkdir -p \
  "$WORK_ROOT"/manifests \
  "$WORK_ROOT"/pod5 \
  "$WORK_ROOT"/bam \
  "$WORK_ROOT"/chunks/per_run \
  "$WORK_ROOT"/chunks/stage1_train \
  "$WORK_ROOT"/models
```

示例下载目录中已有的 run 名称类似：

```text
Random_Ligation_A_m6A_ERR12770807/
RL_run2_ERR12770808/
RL_mix_w_new_oligos_ERR12770809/
20230221_WUE_splint_lig_ERR12770810/
20230502_WUE_splint_batch2_ERR12770811/
20230510_WUE_splint_batch2_ERR12770812/
20230523_RL_M4_M5_ERR12770813/
20230626_WUE_batch_3_ERR12770814/
20230628_WUE_batch_3_ERR12770815/
100_WT_0_IVT_RTA_ERR12772618/
0_WT_100_IVT_RTA_ERR12953422/
25_WT_75_IVT_RTA_ERR12953423/
50_WT_50_IVT_RTA_rep1_ERR12953424/
75_WT_25_IVT_RTA_rep1_ERR12953425/
50_WT_50_IVT_RTA_rep2_ERR12953890/
75WT_rep2_ERR12953891/
```

Stage1 优先使用论文中 synthetic oligo TRAIN 集对应的 run，例如 random ligation
和 splint ligation 的全修饰/未修饰 synthetic oligo run。ratio-IVT run，如
`25_WT_75_IVT_RTA`、`50_WT_50_IVT_RTA`、`75_WT_25_IVT_RTA`，更适合 Stage2 LLP
校准，不应作为 Stage1 的中心位点强标签来源。

## 1. 准备 manifest

先生成模板：

```bash
cd "$REPO"

python gen_data/create_mafia_synthetic_stage1_dataset.py \
  --write-template-manifest "$WORK_ROOT/manifests"
```

这会生成两个模板文件：

```text
$WORK_ROOT/manifests/mafia_oligos.tsv
$WORK_ROOT/manifests/mafia_runs.tsv
```

### 1.1 `mafia_oligos.tsv`

这个文件描述每条 synthetic oligo。需要根据论文 Supplementary Table 1 填写真实序列。

必需字段：

```text
oligo_id
sequence
motif
ligation_strategy
role
```

推荐写法：

```text
oligo_id	sequence	motif	ligation_strategy	role
RL_M0_S0	ACUGGACU/m6AACUGGA	DRACH	random_ligation	train
RL_M0_S0_unmod	ACUGGACUAACUGGA	DRACH	random_ligation	train
SL_AB	UGGACU/m6AACUGGAC	splint_motif	splint_ligation	test
```

注意：

- `sequence` 可以用 `/m6A`、`[m6A]`、`(m6A)` 或 `m6A` 标出设计修饰位点。
- 如果序列列不方便写 modification token，也可以增加 `center_index` 字段，使用
  0-based canonical base 坐标指定中心 A。
- TetraMod 内部按 DNA alphabet 存储 target，脚本会把 `U` 归一化为 `T`。
- `role=train` 的 oligo 才建议进入 Stage1 训练；`role=test` 建议留作独立评估。

### 1.2 `mafia_runs.tsv`

这个文件描述每个测序 run 包含哪些 oligo，以及该 run 的标签属性。需要根据论文
Supplementary Table 3 和本地目录名填写。

必需字段：

```text
run_id
run_dir
modification_status
oligo_ids
modified_oligo_ids
role
```

示例：

```text
run_id	run_dir	modification_status	oligo_ids	modified_oligo_ids	role
RL_m6A_1	Random_Ligation_A_m6A_ERR12770807	modified	RL_M0_S0,RL_M1_S0,RL_M2_S0	RL_M0_S0,RL_M1_S0,RL_M2_S0	train
RL_unmod_1	RL_run2_ERR12770808	unmodified	RL_M0_S0_unmod,RL_M1_S0_unmod,RL_M2_S0_unmod		train
TEST1	RL_mix_w_new_oligos_ERR12770809	mixed	RL_M0_S0,RL_M1_S0,RL_M2_S0,RL_NEG_1	RL_M0_S0,RL_M1_S0	test
```

字段含义：

- `modification_status=modified`：该 run 中列出的 oligo 中心位点按 m6A positive 标注。
- `modification_status=unmodified`：该 run 中列出的 oligo 中心位点按 canonical A negative 标注。
- `modification_status=mixed`：必须填写 `modified_oligo_ids`，脚本只把这些 oligo
  标为 positive，其他列出的 oligo 标为 negative。
- `role=train`：可进入 Stage1 训练。
- `role=test`：保留作 held-out 诊断。

## 2. FAST5 转 POD5

对每个需要处理的 run 执行转换。下面用 `RUN_DIR` 表示下载目录名，用 `RUN_ID`
表示你在 `mafia_runs.tsv` 中填写的逻辑 ID。

```bash
cd "$REPO"

export RUN_ID=RL_m6A_1
export RUN_DIR=Random_Ligation_A_m6A_ERR12770807

python gen_data/convert_fast5_tar_to_pod5.py \
  "$MAFIA_ROOT/$RUN_DIR" \
  --output-dir "$WORK_ROOT/pod5/$RUN_ID" \
  --recursive \
  --jobs 1
```

如果 archive 里的 FAST5 member 没有 `.fast5` 后缀，增加：

```bash
  --fast5-member-pattern '*'
```

转换后建议检查 POD5 是否生成：

```bash
find "$WORK_ROOT/pod5/$RUN_ID" -name '*.pod5' | head
```

## 3. Dorado RNA002 basecalling

必须使用 `--emit-moves`，因为后续需要 Dorado move table 把 read sequence 映射回
raw signal chunk。

```bash
export DORADO_MODEL=/path/to/dorado/rna002_model

dorado basecaller "$DORADO_MODEL" "$WORK_ROOT/pod5/$RUN_ID" \
  --emit-moves \
  --device cuda:0 \
  > "$WORK_ROOT/bam/$RUN_ID.bam"
```

建议排序并建立索引，便于后续检查：

```bash
samtools sort -@ 8 -o "$WORK_ROOT/bam/$RUN_ID.sorted.bam" "$WORK_ROOT/bam/$RUN_ID.bam"
samtools index "$WORK_ROOT/bam/$RUN_ID.sorted.bam"
```

检查 BAM 中是否有 `mv` 和 `ns` tag：

```bash
samtools view "$WORK_ROOT/bam/$RUN_ID.sorted.bam" | head -1
```

如果没有 `mv:B:c` 或 `ns:i`，说明 basecalling 命令不满足数据集生成要求。

## 4. 生成 per-run TetraMod 数据集

对每个 train run 生成一个 per-run dataset：

```bash
cd "$REPO"

python gen_data/create_mafia_synthetic_stage1_dataset.py \
  --bam-file "$WORK_ROOT/bam/$RUN_ID.sorted.bam" \
  --pod5-dir "$WORK_ROOT/pod5/$RUN_ID" \
  --output-dir "$WORK_ROOT/chunks/per_run/$RUN_ID" \
  --oligo-manifest "$WORK_ROOT/manifests/mafia_oligos.tsv" \
  --run-manifest "$WORK_ROOT/manifests/mafia_runs.tsv" \
  --run-id "$RUN_ID" \
  --sample-type rna \
  --rna002 \
  --chunk-len 10000 \
  --overlap 500 \
  --workers 8
```

常用调试参数：

```bash
  --max-records 1000
  --max-chunks 200
  --min-qscore 7
  --min-oligo-identity 0.9
  --max-oligo-mismatches 2
```

输出文件：

```text
chunks.npy
references.npy
reference_lengths.npy
mod_targets.npy
metadata.npz
metadata_fields.json
dataset_summary.json
```

其中 `mod_targets.npy` 是 Stage1 的核心标签：

```text
0    canonical_A negative
4    m6A positive
-100 ignored position
```

生成后检查 summary：

```bash
cat "$WORK_ROOT/chunks/per_run/$RUN_ID/dataset_summary.json"
```

重点看：

- `samples_written` 是否大于 0。
- `positive_centers` / `negative_centers` 是否符合该 run 的预期。
- modified run 应主要产生 positive center。
- unmodified run 应主要产生 negative center。
- 如果 mixed run 没有填写 `modified_oligo_ids`，脚本会避免生成强标签。

## 5. 批量处理多个 run

可以写一个 shell 循环。示例只展示命令结构，`RUN_IDS` 应来自
`mafia_runs.tsv` 中你确认可用于 Stage1 的 train run。

```bash
cd "$REPO"

RUN_IDS=(
  RL_m6A_1
  RL_unmod_1
  SL_m6A_1
  SL_unmod_1
)

for RUN_ID in "${RUN_IDS[@]}"; do
  RUN_DIR=$(awk -F'\t' -v id="$RUN_ID" 'NR > 1 && $1 == id {print $2}' "$WORK_ROOT/manifests/mafia_runs.tsv")

  python gen_data/convert_fast5_tar_to_pod5.py \
    "$MAFIA_ROOT/$RUN_DIR" \
    --output-dir "$WORK_ROOT/pod5/$RUN_ID" \
    --recursive \
    --jobs 1

  dorado basecaller "$DORADO_MODEL" "$WORK_ROOT/pod5/$RUN_ID" \
    --emit-moves \
    --device cuda:0 \
    > "$WORK_ROOT/bam/$RUN_ID.bam"

  samtools sort -@ 8 -o "$WORK_ROOT/bam/$RUN_ID.sorted.bam" "$WORK_ROOT/bam/$RUN_ID.bam"
  samtools index "$WORK_ROOT/bam/$RUN_ID.sorted.bam"

  python gen_data/create_mafia_synthetic_stage1_dataset.py \
    --bam-file "$WORK_ROOT/bam/$RUN_ID.sorted.bam" \
    --pod5-dir "$WORK_ROOT/pod5/$RUN_ID" \
    --output-dir "$WORK_ROOT/chunks/per_run/$RUN_ID" \
    --oligo-manifest "$WORK_ROOT/manifests/mafia_oligos.tsv" \
    --run-manifest "$WORK_ROOT/manifests/mafia_runs.tsv" \
    --run-id "$RUN_ID" \
    --sample-type rna \
    --rna002 \
    --chunk-len 10000 \
    --overlap 500 \
    --workers 8
done
```

## 6. 合并 Stage1 train dataset

`merge_mafia_stage1_datasets.py` 会把多个 per-run dataset 合并成一个训练目录，并在
`validation/` 子目录中生成验证集。

```bash
cd "$REPO"

python gen_data/merge_mafia_stage1_datasets.py \
  --dataset RL_m6A_1:"$WORK_ROOT/chunks/per_run/RL_m6A_1" \
  --dataset RL_unmod_1:"$WORK_ROOT/chunks/per_run/RL_unmod_1" \
  --dataset SL_m6A_1:"$WORK_ROOT/chunks/per_run/SL_m6A_1" \
  --dataset SL_unmod_1:"$WORK_ROOT/chunks/per_run/SL_unmod_1" \
  --output-dir "$WORK_ROOT/chunks/stage1_train" \
  --valid-fraction 0.25 \
  --seed 114514
```

默认行为：

- 只保留存在明确中心标签的样本。
- 按 motif、ligation strategy、modification status、run_id 分层切分 validation。
- 训练集默认在每个 motif 内对 positive/negative 做下采样平衡。

如果只是想保留原始分布，不做训练集平衡：

```bash
python gen_data/merge_mafia_stage1_datasets.py \
  --dataset RL_m6A_1:"$WORK_ROOT/chunks/per_run/RL_m6A_1" \
  --dataset RL_unmod_1:"$WORK_ROOT/chunks/per_run/RL_unmod_1" \
  --output-dir "$WORK_ROOT/chunks/stage1_train_unbalanced" \
  --valid-fraction 0.25 \
  --no-balance-train
```

检查合并结果：

```bash
cat "$WORK_ROOT/chunks/stage1_train/mafia_stage1_merge_summary.json"
find "$WORK_ROOT/chunks/stage1_train" -maxdepth 2 -type f | sort
```

期望目录结构：

```text
$WORK_ROOT/chunks/stage1_train/
  chunks.npy
  references.npy
  reference_lengths.npy
  mod_targets.npy
  metadata.npz
  metadata_fields.json
  mafia_stage1_merge_summary.json
  validation/
    chunks.npy
    references.npy
    reference_lengths.npy
    mod_targets.npy
    metadata.npz
    metadata_fields.json
```

## 7. Stage1 训练

当前 `train_promote` 支持 `control` 和 `llp` 两个 stage。mAFiA synthetic oligo
强标签数据用于：

```bash
--promote-stage control
```

训练命令示例：

```bash
cd "$REPO"

export PRETRAINED=/path/to/rna002_tetramod_or_bonito_pretrained_model

python -m tetramod train_promote "$WORK_ROOT/models/stage1_mafia_control" \
  --force \
  --directory "$WORK_ROOT/chunks/stage1_train" \
  --config src/tetramod/models/configs/multihead_transformer.toml \
  --pretrained "$PRETRAINED" \
  --promote-stage control \
  --promote-base A \
  --epochs 20 \
  --batch 64 \
  --lr 1e-4 \
  --device cuda:0 \
  --num-workers 4 \
  --no-compile
```

如果已经安装了 `tetramod` console script，也可以用：

```bash
tetramod train_promote "$WORK_ROOT/models/stage1_mafia_control" \
  --force \
  --directory "$WORK_ROOT/chunks/stage1_train" \
  --config src/tetramod/models/configs/multihead_transformer.toml \
  --pretrained "$PRETRAINED" \
  --promote-stage control \
  --promote-base A \
  --epochs 20 \
  --batch 64 \
  --lr 1e-4 \
  --device cuda:0 \
  --num-workers 4 \
  --no-compile
```

Stage1 训练重点检查：

- encoder 是否保持冻结。
- 当前只训练 A-head / m6A 相关输出。
- validation 中 positive/negative 是否都存在。
- modified 和 unmodified synthetic control 是否能明显分开。
- 不要把 Stage1 结果直接解释成 wild-type 泛化能力。

## 8. Stage2 LLP 的衔接方式

ratio-IVT run 可以用于 Stage2 校准，但应从 Stage1 checkpoint 初始化：

```bash
python -m tetramod train_promote "$WORK_ROOT/models/stage2_mafia_ratio_50" \
  --force \
  --directory /path/to/ratio_ivt_llp_dataset \
  --config src/tetramod/models/configs/multihead_transformer.toml \
  --pretrained "$PRETRAINED" \
  --init-promote-checkpoint "$WORK_ROOT/models/stage1_mafia_control" \
  --promote-stage llp \
  --promote-base A \
  --llp-proportion 0.50 \
  --llp-loss huber \
  --llp-tolerance 0.05 \
  --llp-huber-delta 0.05 \
  --epochs 5 \
  --batch 64 \
  --lr 2e-5 \
  --device cuda:0 \
  --num-workers 4 \
  --no-compile
```

注意：

- Stage2 的 ratio-IVT 只证明 bag-level proportion 是否校准。
- ratio、run、k-mer/site distribution 可能混杂。
- Stage2 不能替代 Stage1 强标签，也不能单独证明 wild-type site-level calling 能力。

## 9. 常见问题

### 9.1 per-run dataset 没有样本

检查：

```bash
cat "$WORK_ROOT/chunks/per_run/$RUN_ID/dataset_summary.json"
samtools view "$WORK_ROOT/bam/$RUN_ID.sorted.bam" | head -1
```

可能原因：

- BAM 没有 `--emit-moves` 产生的 `mv` tag。
- `mafia_runs.tsv` 的 `run_id` 和命令传入的 `--run-id` 不一致。
- `mafia_runs.tsv` 的 `oligo_ids` 在 `mafia_oligos.tsv` 中不存在。
- oligo sequence 填错，无法在 read sequence 中匹配。
- `--min-oligo-identity` 太高或 `--max-oligo-mismatches` 太低。

### 9.2 modified run 没有 positive label

检查：

- `modification_status` 是否写成 `modified`。
- oligo sequence 中是否正确标出 `/m6A`，或是否提供了正确 `center_index`。
- mixed run 是否填写了 `modified_oligo_ids`。

### 9.3 是否能把多个样本来源混入 Stage1

可以混入多个 synthetic oligo 来源，但建议只混入满足同一类强标签假设的数据：

- 同样是 synthetic oligo。
- 同样有明确中心 m6A / canonical A 位点。
- 同时包含 modified 和 unmodified 对照。
- merge 时保留 run、ligation strategy、motif 的 metadata，validation 按这些字段分层。

不建议在 Stage1 混入 HEK293 WT、ratio-IVT、或只有 bag-level proportion 的数据。这些数据的标签语义不同，应放到 Stage2/Stage3。

## 10. 本地轻量验证命令

在本地 WSL2 只建议跑脚本级 smoke test，不处理完整 FAST5/BAM：

```bash
cd "$REPO"

python gen_data/create_mafia_synthetic_stage1_dataset.py --help
python gen_data/merge_mafia_stage1_datasets.py --help
python -m tetramod train_promote --help

UV_CACHE_DIR=.uv-cache XDG_CACHE_HOME=.cache \
  uv run python -m unittest discover -s tests -p 'test_*.py'
```
