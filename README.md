# TetraMod

TetraMod 是一个面向 Oxford Nanopore 直接 RNA 信号的修饰建模研究项目，目前重点是 m6A。项目目标不是替代 Bonito 或 Dorado，而是在官方/预训练 basecaller 的隐藏表示之上，训练一个独立的 modification branch，并探索有限监督条件下的 promoted training 流程。

本仓库当前更适合作为方法开发与实验验证代码使用，而不是生产级修饰检测工具。README 中的结论均以当前代码、根目录脚本和 `val_res/` 中已有结果为准。

## 与 Bonito 的关系

`../bonitov1.1.0` 是本项目参考的上游 Bonito 代码。Bonito 本身是 ONT 的研究型 basecaller，提供了以下基础能力：

- basecaller 模型配置、加载、训练和推理流程；
- CRF basecalling 组件；
- Bonito 风格的训练数据设置和 DataLoader 接口；
- POD5/FASTQ/SAM/BAM/CRAM IO；
- alignment、writer、beam search、模型下载等工具。

TetraMod 没有完整 fork Bonito，而是在 `pyproject.toml` 中依赖 `ont-bonito`，并在 `src/tetramod/` 中实现修饰建模相关的新增逻辑。

可以把两者关系理解为：

```text
Bonito / Dorado-compatible basecaller
  提供官方 basecalling encoder、CRF、数据与 IO 生态

TetraMod
  在冻结或复用 basecaller hidden representation 的基础上
  增加 m6A / modification branch、训练目标和验证工具
```

当前边界非常重要：

- `tetramod train` 是受保护的 baseline 路径，用于直接比较，不应被 promoted 实验逻辑污染。
- `tetramod train_promote` 是 promoted training 路径，目前支持 A-head 的 `control` 和 `llp` 两个阶段。
- `tetramod basecaller` 是 Bonito 风格 basecaller wrapper，可以加载 TetraMod 多头模型，在保持 basecalling 输出的同时生成 MM/ML 修饰标签。

对于常规生产 basecalling，请继续使用 Dorado。TetraMod 面向的是修饰模型开发、有限监督训练和实验验证。

## 项目架构

当前 TetraMod 的核心建模路径是：

```text
raw signal chunk
  -> 预训练 Bonito / ONT basecaller encoder
  -> basecaller hidden representation
  -> TetraMod mod input projection
  -> trainable shared mod trunk
  -> per-base modification heads
  -> read/site-level modification probabilities
  -> optional bag-level aggregation for LLP
```

主要代码结构：

- `src/tetramod/cli/__init__.py`：注册 CLI 子命令，包括 `train`、`train_promote` 和 `basecaller`。
- `src/tetramod/cli/train.py`：baseline 训练入口。加载预训练 basecaller 配置和权重，合并 runtime 配置，调用 `TrainerMod`。
- `src/tetramod/cli/train_promote.py`：promoted 训练入口。目前将 promoted path 限制为 A-head，并根据 `--promote-stage` 进入 control 或 LLP 训练。
- `src/tetramod/transformer/multihead_model.py`：多头修饰模型。它重建预训练 encoder/CRF，在 standalone mod-head 模式下冻结 basecaller 参数，并添加 `mod_input_proj`、`mod_trunk` 和 per-base `mod_heads`。
- `src/tetramod/training_mod.py`：baseline 训练循环，结构上继承 Bonito trainer 的风格，负责训练、验证、日志、checkpoint、梯度裁剪和 standalone mod-head checkpoint。
- `src/tetramod/training_promote.py`：promoted loss 和 trainer wrapper。当前实现 A-head control BCE 和 LLP bag proportion loss。
- `src/tetramod/train_mod_data.py`：TetraMod 数据加载器。支持 numpy 数据集，也支持动态 `dataset.py`。LLP 数据可额外提供 `bag_keys.npy` 和 `bag_targets.npy`。
- `src/tetramod/transformer/multihead_basecall.py`：TetraMod basecalling helper。复用 Bonito 风格 chunk/stitch 流程，并根据 modification logits 生成 MM/ML tag。
- `gen_data/`：数据构建脚本，包括 Dorado/BAM 到 chunk dataset、0/100 control、ratio-IVT LLP、mAFiA Stage 1 数据集等。
- `validate/`：验证工具，包括 checkpoint 评估、control 分离度、LLP bag 评估、modBAM gold-site 评估、negative control 和 basecaller BAM 比较。
- `vis/`：对 promoted control 与 LLP 结果的可视化脚本。

默认模型配置为：

```text
src/tetramod/models/configs/multihead_transformer.toml
```

默认多头设计为：

```text
A-head: canonical_A / m6A
C-head: canonical_C
G-head: canonical_G
T/U-head: canonical_T
```

但 `train_promote` 当前会将模型限制为：

```text
mod_bases = ["A"]
A = ["canonical_A", "m6A"]
```

这是有意设计。当前 promoted 实验只聚焦 m6A，不应在没有真实标签的情况下训练 C/G/T 修饰头。

## 数据格式

TetraMod 使用 Bonito-like CTC chunk 数据，并额外加入修饰标签：

```text
chunks.npy              raw signal chunks，shape [N, chunk_len]
references.npy          reference base id，padding 后 shape [N, max_ref_len]
reference_lengths.npy   每条 reference 的真实长度，shape [N]
mod_targets.npy         每个 reference position 的修饰标签或 ignore index
indices.npy             可选，用于选择子集
bag_keys.npy            可选，LLP 中每条 read 所属 bag id
bag_targets.npy         可选，LLP 中每条 read/bag 的目标比例
validation/             可选验证集目录，内部文件结构相同
```

不同训练阶段中 `mod_targets.npy` 的含义不同：

- Stage 1 control：`mod_targets.npy` 是强监督标签，例如 canonical A 或 m6A。
- Stage 2 LLP：`mod_targets.npy` 只表示哪些 A 位点可参与 LLP 聚合，是 candidate site 标记，不是 read-level 真值。

## CLI 用法

Baseline 训练：

```bash
tetramod train OUTPUT_DIR \
  --directory DATASET_DIR \
  --config src/tetramod/models/configs/multihead_transformer.toml \
  --pretrained PRETRAINED_BONITO_MODEL_DIR
```

Promoted Stage 1 control warm-up：

```bash
tetramod train_promote OUTPUT_DIR \
  --directory CONTROL_MIX_DATASET_DIR \
  --config src/tetramod/models/configs/multihead_transformer.toml \
  --pretrained PRETRAINED_BONITO_MODEL_DIR \
  --promote-stage control \
  --promote-base A
```

Promoted Stage 2 LLP fine-tuning：

```bash
tetramod train_promote OUTPUT_DIR \
  --directory LLP_RATIO_DATASET_DIR \
  --config src/tetramod/models/configs/multihead_transformer.toml \
  --pretrained PRETRAINED_BONITO_MODEL_DIR \
  --init-promote-checkpoint STAGE1_OUTPUT_DIR \
  --promote-stage llp \
  --promote-base A \
  --llp-loss huber \
  --llp-tolerance 0.025 \
  --llp-huber-delta 0.05
```

Basecalling 并输出修饰标签：

```bash
tetramod basecaller MODEL_DIR POD5_DIR \
  --device cuda:0 \
  --rna \
  --reference REF_FASTA \
  --mod-threshold 0.5 \
  > calls.bam
```

当前代码中 `train_promote` 只支持 `control` 和 `llp`。`mil` 是后续目标，还没有作为 CLI stage 实现。

## 训练阶段与目标

### Stage 1：0% vs 100% control 强监督 warm-up

目标：在冻结官方 basecaller encoder 的前提下，让 TetraMod mod branch 学到 m6A-sensitive signal。

当前实现：

- CLI：`--promote-stage control`
- 训练头：A-head，仅 `canonical_A / m6A`
- encoder：standalone mod-head 模式下冻结 basecaller 参数
- loss：promoted A-head BCE
- alignment：使用 basecalling 输出和 reference 的 Viterbi/edlib equal-base alignment，将 modification target 投影到预测时间轴
- 标签假设：0% IVT/control 是 canonical A，100% 或 fully modified control 是 m6A
- 输出层级：site/read 对齐后的 A-head 概率；验证时可汇总 dataset-level mean probability

这个阶段是整个 promoted pipeline 的监督锚点。LLP 不应作为第一个或唯一训练信号。

### Stage 2：ratio-IVT LLP fine-tuning / calibration

目标：利用 12.5%、25%、50%、75% 等 mixed-ratio IVT 数据，让 Stage 1 输出在 bag-level proportion 上更一致。

当前实现：

- CLI：`--promote-stage llp`
- 推荐初始化：使用 `--init-promote-checkpoint STAGE1_OUTPUT_DIR`
- candidate sites：由 `mod_targets.npy` 标记的 A-head 候选位点
- bag：优先使用 `bag_keys.npy`；否则根据 sample key 和 `--llp-bag-size` 分组
- bag label：来自 `bag_targets.npy` 或固定 `--llp-proportion`
- read score：一条 read 中候选 A 位点 m6A probability 的均值
- bag score：bag 内 read score 的均值
- loss：probability-space BCE、MSE 或 Huber，可配合 tolerance 做 relaxed LLP
- 输出层级：bag-level predicted fraction，同时记录 bag/read 数量和误差指标

需要注意：LLP 的 bag-level loss 变好并不能证明 read-level prediction 正确。ratio 和 run、site/k-mer 分布、read quality、mapping coverage 都可能混杂。ratio-IVT 结果也不能单独支持 wild-type generalization。

### Stage 3：wild-type / MIL adaptation

目标：用野生型位点级弱标签或生物学证据，将 control/ratio 校准后的模型推进到真实 biological site-level calling。

这个阶段当前还没有作为 `train_promote` CLI 实现。未来可能使用：

- gold-standard wild-type sites；
- miCLIP / m6A-CLIP labels；
- knockout 或 writer perturbation 数据；
- high-confidence positive / negative site sets。

默认 pooling 应该先用 mean pooling。只有在简单 baseline 可靠后，才考虑 attention pooling。

只有完成这一阶段并通过相应验证后，才应该把模型称为 wild-type m6A caller 候选。

## 已尝试训练方案与已有结果

根目录下的 `.sh` 文件主要是实验记录，包含服务器路径、数据路径和模型路径，不是开箱即用的通用脚本。下面总结的是这些脚本和 `val_res/` 中现有结果能支持的事实。

### 方案 1：0 vs 100 强监督，再用 ratio dataset 做 LLP

相关脚本：

- `train_2phase.sh`
- `train_promote.sh`
- `train_promote_llp.sh`

主要流程：

1. 将 FAST5/POD5 和 Dorado BAM 转成 TetraMod chunk dataset。
2. 生成 m6A targets：
   - 0% 或 canonical control：`make_mod_targets_m6a.py --mode canonical`
   - 100% 或 fully modified control：`make_mod_targets_m6a.py --mode full-mod`
   - ratio-IVT：`make_mod_targets_m6a.py --mode llp-candidate`
3. 合并 0% 与 100% control，训练 Stage 1：

```bash
tetramod train_promote ... --promote-stage control --promote-base A
```

4. 用 ratio-IVT 构建 LLP bag：
   - `build_llp_mixture_dataset.py`
   - `build_real_llp_from_ratio_ivt.py`
   - `build_synthetic_llp_from_controls.py`
5. 从 Stage 1 checkpoint 初始化，训练 Stage 2：

```bash
tetramod train_promote ... \
  --init-promote-checkpoint STAGE1_OUTPUT_DIR \
  --promote-stage llp \
  --llp-loss huber
```

6. 用以下工具验证：
   - `validate/evaluate_promote_control.py`
   - `validate/evaluate_llp_bags.py`
   - `validate/diagnose_llp_dataset.py`
   - `vis/plot_eval_results.py`

RNA002 Stage 1 control 结果保存在：

```text
val_res/rna002_m6a_stage1_control_run1
```

结果摘要：

- IVT mean predicted m6A probability：`0.013147`
- full-mod mean predicted m6A probability：`0.974931`
- full-mod minus IVT gap：`0.961784`
- 0% 和 100% control 在该验证集上明显可分。

RNA004 synthetic/control-derived LLP 结果保存在：

```text
val_res/llp_run1_all
```

结果摘要：

- ratio：`0, 25, 50, 75, 100`
- bags：`1025`
- reads：`20500`
- mean bag score：
  - 0%：`0.0131`
  - 25%：`0.2540`
  - 50%：`0.4950`
  - 75%：`0.7384`
  - 100%：`0.9786`
- mean bag score 随 ratio 单调增加。

RNA002 real ratio-IVT 数据诊断保存在：

```text
val_res/rna002_llp_aligned_highmod_bag20
```

结果摘要：

- train split：`283120` reads，`14156` bags
- 12.5%、25%、50%、75% 每个 ratio 平衡到 `3539` bags
- bag size 固定为 20
- 但不同 ratio 的 quality、mapping coverage、candidate A site 分布仍有差异，ratio/run confounding 仍然是主要风险。

注意：`train_2phase.sh` 中列出了 `stage2_llp_run1` 的训练和评估命令，但当前本地 `val_res/` 树中没有对应的 `stage2_llp_run1_evaluate_llp_bags` 等结果目录。因此 README 不把这部分写成已有结果，只记录为脚本中尝试过的流程。

### 方案 2：使用 mAFiA 数据做 Stage 1 监督训练

相关脚本：

- `train_mix_stage1.sh`
- `train_mix_stage1_6motif_dataset.sh`

主要流程：

1. 使用 mAFiA RNA002 HEK293 FAST5/TAR 数据。
2. 通过 `convert_fast5_tar_to_pod5.py` 转成 POD5。
3. 使用 Dorado RNA002 模型并开启 `--emit-moves` basecall。
4. 使用 `create_mafia_synthetic_stage1_dataset.py` 构建 per-run Stage 1 数据集。
5. 使用 `merge_mafia_stage1_datasets.py` 合并训练集。
6. 训练：

```bash
tetramod train_promote ... --promote-stage control
```

7. 使用 `validate/evaluate_mafia_stage1.py` 和 dataset balance 工具验证。

已有 mAFiA Stage 1 internal validation：

```text
val_res/mafia_stage1_epoch5
```

- sites：`19054`
- positive sites：`1938`
- negative sites：`17116`
- ROC AUC：`0.996878`
- PR AUC：`0.980901`
- balanced accuracy：`0.974189`
- mean positive probability：`0.955911`
- mean negative probability：`0.019863`

```text
val_res/mafia_stage1_epoch20
```

- ROC AUC：`0.996558`
- PR AUC：`0.975491`
- balanced accuracy：`0.978380`
- mean positive probability：`0.972120`
- mean negative probability：`0.016359`

但是 heldout batch2 per-run transfer 明显更弱：

- `WUE_splint_batch2_A_RTA`
  - mean negative probability：`0.131370`
  - specificity/accuracy：`0.901962`
- `WUE_splint_batch2_m6A_RTA`
  - mean positive probability：`0.605883`
  - recall：`0.600985`
- `WUE_splint_batch2_m6A_RTA_1`
  - mean positive probability：`0.627572`
  - recall：`0.644444`
- `WUE_splint_batch2_m6A_RTA_2`
  - mean positive probability：`0.536645`
  - recall：`0.534161`

因此，mAFiA Stage 1 在 internal validation 上很强，但跨 heldout run 泛化仍不稳定。这个结果更适合作为 Stage 1 监督训练可行性的证据，而不是 wild-type caller 的证据。

`train_mix_stage1_6motif_dataset.sh` 进一步构建了包含 6 个 DRACH motif 的更大训练集，并准备 final heldout runs。当前 `val_res/` 中还没有看到对应大训练集的完整评估结果。

### 其他历史实验

`scripts.sh` 记录了早期 RNA004 baseline-style `tetramod train` 实验，结果保存在：

```text
val_res/rna004_m6a_mix_tetra/summary.txt
```

synthetic heldout 上结果很强：

- modification accuracy：`0.9957`
- binary m6A-vs-canonical F1：`0.9944`
- ROC AUC：`0.999858`
- PR AUC：`0.999779`

但同一模型在 wild-type human gold-site 评估上很弱：

```text
val_res/rna004_m6a_mix_tetra/m6A_gold_eval
```

- ROC AUC：`0.5093`
- PR AUC：`0.0260`
- threshold 0.5 F1：`0.0036`

这说明 synthetic/control 分离不能直接外推到 wild-type。

`train_HeLa.sh` 记录了 RNA002 Stage 1/Stage 2 模型在 HeLa 和 IVT negative control 上的 basecalling 与 modBAM 评估。

Basecalling 与 Bonito 的一致性较高：

```text
val_res/hela_rna002_tetramod_15/basecall_compare
```

- shared reads：`60000`
- mean sequence identity vs Bonito：`0.999733`
- exact match fraction：`0.959467`

但 HeLa gold-site m6A 表现尚不足以支持 wild-type generalization：

- Stage 1 HeLa gold-site ROC AUC：`0.8230`
- Stage 1 HeLa gold-site PR AUC：`0.0175`
- Stage 2 HeLa gold-site ROC AUC：`0.8124`
- Stage 2 HeLa gold-site PR AUC：`0.0150`

0% IVT negative control 仍有一定 false positive：

```text
val_res/curlcakes_cc0_test4000/negative_control_mincov5
```

- `score >= 0.5` false-positive fraction：`0.102754`
- `score >= 0.8` false-positive fraction：`0.018359`
- `score >= 0.95` false-positive fraction：`0.002961`

## 验证工具

Control 分离度：

```bash
python validate/evaluate_promote_control.py MODEL_DIR \
  --ivt-dir IVT_DATASET \
  --full-mod-dir FULL_MOD_DATASET \
  --output-dir val_res/control_eval \
  --dataset valid
```

LLP bag-level calibration：

```bash
python validate/evaluate_llp_bags.py MODEL_DIR \
  --directory LLP_DATASET \
  --dataset valid \
  --output-dir val_res/llp_eval
```

modBAM gold-site 评估：

```bash
python validate/evaluate_modbam_gold_sites.py \
  --bam calls.sorted.bam \
  --gold-bed gold_sites.tsv \
  --reference ref.fa \
  --output-dir val_res/gold_eval \
  --mod-code a \
  --canonical-base A
```

negative control false-positive 评估：

```bash
python validate/evaluate_modbam_negative_control.py \
  --bam ivt_control.sorted.bam \
  --reference ref.fa \
  --output-dir val_res/negative_control \
  --mod-code a \
  --canonical-base A
```

更多说明见：

- `validate/README.md`
- `vis/README.md`

## 当前可以支持的结论

当前代码和结果支持：

- TetraMod 可以加载预训练 Bonito-style basecaller encoder，并在其上训练独立修饰分支。
- `train` 与 `train_promote` 是分离的 CLI 路径。
- `train_promote control` 可以训练 A-head m6A 强监督 warm-up。
- `train_promote llp` 可以做 bag-level proportion fine-tuning。
- Stage 1 control validation 可以明显区分 0% 和 100% controls。
- LLP 在部分 synthetic/control ratio 数据上可以得到单调且接近目标比例的 bag-level score。
- `tetramod basecaller` 的 basecalling 输出可以与 Bonito 保持很高一致性，并额外输出 modification tags。

当前不应声称：

- 已经实现可靠 wild-type m6A generalization；
- LLP loss 足以证明 read-level prediction 正确；
- C/G/T 修饰头已经可用；
- MIL training 已经实现；
- TetraMod 已经是 production-ready modified-base caller。

## 本地轻量测试

当前 smoke/unit tests 覆盖 CLI 注册、`train_promote` 参数、checkpoint 解析、A-head 限制、LLP 设置、RNA002 legacy encoder 重建和 modBAM gold-site helper。

```bash
UV_CACHE_DIR=.uv-cache XDG_CACHE_HOME=.cache \
uv run python -m unittest discover -s tests -p 'test_*.py'
```

