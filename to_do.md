• 我认为这个两阶段方案是更正确的主线，也比直接拿当前 RNA002 ratio-IVT 数据做 LLP 更接近你的 TetraMod 长期目标。但要明确一点：它能显著提高通向 wild type 泛化的概率，不能单独保证 wild type 泛化。

  核心判断如下。

  推荐训练路线

  第一阶段：0 vs 100 强监督 warm-up
  用未修饰数据和全修饰数据训练 mod trunk + mod head。官方 ONT basecaller encoder 作为冻结特征提取器，先不动它。

  目标是让模型学到：

  同一 basecaller hidden feature 下，修饰信号和未修饰信号如何分开

  这一步是必要的，因为 LLP 本身监督太弱。如果没有强监督 anchor，模型很容易只学到 batch/run/coverage/motif 分布差异，而不是修饰信号。

  第二阶段：ratio-IVT LLP fine-tune
  用 12.5/25/50/75 这类投料比例数据微调第一阶段模型。

  这一步的合理定位不是“从零训练分类器”，而是：

  校准第一阶段模型在部分修饰、混合比例、不同实验批次下的输出尺度

  也就是说，LLP 应该是 fine-tune / calibration，而不是主训练信号。

  为什么这比直接 LLP 好

  你现在的 LLP 数据已经暴露出问题：

  75% 组 bag 太小
  ratio 和 run 完全共线
  50% vs 75% 的 primary_site/kmer 分布差异明显

  直接 LLP 时，模型无法区分：

  这是 75% 修饰信号
  还是 rna002_75 这个 run 的 batch effect
  还是 75% 组 site/kmer 构成不同

  但如果先用 0/100 训练出一个已经懂“修饰信号长什么样”的 head，再用 LLP 做轻微校准，风险会小很多。

  但 wild type 泛化仍然差一个环节

  0/100 强监督有一个问题：它不是 wild type 分布。

  全修饰 IVT 会让模型看到一种极端状态：

  所有候选 A 都倾向 modified

  而 wild type 里真实情况通常是：

  只有部分 site 修饰
  同一 site 可能部分 reads 修饰
  修饰率和 motif、结构、细胞状态有关

  所以两阶段后，模型更可能具备“检测 m6A-like signal”的能力，但 site-level wild type 表现还需要第三类验证或微调：

  gold-standard WT site labels
  miCLIP / m6A-CLIP / knockout / writer perturbation
  或至少高可信 positive/negative site set

  没有这一步，最终只能说模型完成了 control-to-ratio 的弱监督训练，不能严格证明 WT 泛化。

  我建议的 TetraMod 总体框架

  Stage 1: control supervised warm-up
  0% vs 100%
  冻结 ONT encoder
  训练 mod trunk + A/m6A head

  Stage 2: LLP fine-tune
  12.5/25/50/75 ratio-IVT
  小学习率
  relaxed MSE/Huber
  主要校准 bag-level proportion

  Stage 3: WT/MIL adaptation
  用 site-level 弱标签或高可信 gold sites
  mean/attention pooling 到 site level
  训练 site-level probability

  对于你的“多修饰类型泛用”目标，架构上应该保持：

  official basecaller encoder
  -> shared mod trunk
  -> per-base / per-modification heads

  不同修饰类型可以共享 trunk，但 head 最好按 base/mod 类型分开，例如：

  A-head: canonical_A / m6A / ...
  C-head: canonical_C / m5C / hm5C / ...
  G-head: ...
  T/U-head: ...

  当前最小正确下一步

  不是继续调 LLP，而是先把 0 vs 100 强监督阶段做扎实：

  1. 准备 matched 的 unmodified 和 full-modified 数据。
  2. 用 train_promote --promote-stage control 训练 A-head。
  3. 在 heldout run / heldout site 上确认 0/100 可分。
  4. 再把这个 checkpoint 作为 LLP fine-tune 起点。

  结论：你的两阶段想法是对的，但我会把它表述为：

  0/100 强监督学习修饰信号
  ratio-IVT LLP 校准混合比例
  WT/MIL 负责最终生物样本泛化

  只做前两步，适合得到一个更可靠的候选模型；要声称 wild type 泛化，还需要第三步证据



---


---


• 你现在手上的两个 POD5 路径还不是 Stage 1 训练集。下一步要把它们变成 TetraMod 的 control warm-up 数据集：

  0% POD5 / 100% POD5
  → Dorado basecall BAM, 必须带 --emit-moves 和 reference alignment
  → TetraMod chunk dataset
  → 0% 标成 canonical_A，100% 标成 m6A
  → 合并成 balanced control dataset
  → train_promote --promote-stage control

  1. 先为两个 POD5 路径分别 basecall
  关键要求：同一个 reference FASTA、同一个 RNA002 basecaller model、BAM 里要有 move table。

  # 0% unmodified
  dorado basecaller /path/to/rna002_model \
    --reference /path/to/reference.fa \
    --emit-moves \
    /path/to/unmodified_0_pod5 \
    > /path/to/stage1/unmod/dorado.bam

  # 100% full-modified
  dorado basecaller /path/to/rna002_model \
    --reference /path/to/reference.fa \
    --emit-moves \
    /path/to/fullmod_100_pod5 \
    > /path/to/stage1/fullmod/dorado.bam

  这里的 /path/to/rna002_model 后面训练时也要作为 tetramod train_promote --pretrained 使用，避免 basecaller encoder 和数据生成不匹配。

  2. 把 BAM + POD5 转成 TetraMod chunk dataset
  RNA002 数据建议加 --rna002，让 gen_data/create_dataset_dorado_ctc_like.py 使用 RNA002 兼容的 normalisation 默认值。

  python gen_data/create_dataset_dorado_ctc_like.py \
    --bam-file /path/to/stage1/unmod/dorado.bam \
    --pod5-dir /path/to/unmodified_0_pod5 \
    --reference-fasta /path/to/reference.fa \
    --output-dir /path/to/stage1/unmod/chunks \
    --run-id unmod_0_run1 \
    --sample-type rna \
    --rna002 \
    --chunk-len 12000 \
    --overlap 600 \
    --filter-preset relaxed \
    --metadata-kmer 5 \
    --workers 8

  python gen_data/create_dataset_dorado_ctc_like.py \
    --bam-file /path/to/stage1/fullmod/dorado.bam \
    --pod5-dir /path/to/fullmod_100_pod5 \
    --reference-fasta /path/to/reference.fa \
    --output-dir /path/to/stage1/fullmod/chunks \
    --run-id fullmod_100_run1 \
    --sample-type rna \
    --rna002 \
    --chunk-len 12000 \
    --overlap 600 \
    --filter-preset relaxed \
    --metadata-kmer 5 \
    --workers 8

  生成后每个目录应至少有：

  chunks.npy
  references.npy
  reference_lengths.npy
  metadata.npz
  summary.json

  3. 生成 m6A Stage 1 标签
  0% 数据：所有有效 A 标成 canonical_A。
  100% 数据：所有有效 A 标成 m6A。
  非 A 位点先忽略，符合当前 A-head only 目标。

  python gen_data/make_mod_targets_m6a.py \
    --dataset-dir /path/to/stage1/unmod/chunks \
    --mode canonical \
    --non-a-policy ignore

  python gen_data/make_mod_targets_m6a.py \
    --dataset-dir /path/to/stage1/fullmod/chunks \
    --mode full-mod \
    --non-a-policy ignore

  生成后每个目录会多出：

  mod_targets.npy

  4. 合并 0% 和 100% controls
  用 gen_data/merge_mod_datasets.py：

  python gen_data/merge_mod_datasets.py \
    --full-mod-dir /path/to/stage1/fullmod/chunks \
    --canonical-dir /path/to/stage1/unmod/chunks \
    --output-dir /path/to/stage1/control_mix/chunks \
    --seed 1


  5. 训练 Stage 1 control warm-up
  用 promoted 入口，不要用 baseline train：

  tetramod train_promote -f /path/to/stage1/model/control_run1 \
    --directory /path/to/stage1/control_mix/chunks \
    --config /home/lijy/TetraMod/src/tetramod/models/configs/multihead_transformer.toml \
    --pretrained /path/to/rna002_model \
    --promote-stage control \
    --promote-base A \
    --epochs 10 \
    --batch 48 \
    --chunks 30000 \
    --valid-chunks 2000 \
    --device cuda:0

  当前代码里 train_promote 会走 standalone mod-head 模式，basecaller encoder 会被冻结，只训练 mod trunk 和 A/m6A head。

  6. Stage 1 验证
  训练后用 0% 和 100% 原始 chunk 目录检查分离度：

  python validate/evaluate_promote_control.py \
    /path/to/stage1/model/control_run1 \
    --output-dir /path/to/stage1/eval/control_run1 \
    --ivt-dir /path/to/stage1/unmod/chunks \
    --full-mod-dir /path/to/stage1/fullmod/chunks \
    --dataset valid \
    --chunks 30000 \
    --valid-chunks 2000 \
    --batchsize 32 \
    --device cuda:0

  你应该重点看：0% 的 A-head m6A 概率是否低，100% 是否高，ROC/PR AUC 是否能明显分开。

  当前 Stage 1 的标签假设很强：0% = 所有 A 未修饰，100% = 所有 A 全 m6A。这只适合作为 supervised warm-up anchor，不要直接拿它声称 wild-type
  generalization。下一步最小安全动作是：确认这套 control 数据能稳定分开后，再用 ratio-IVT 或 synthetic ratio bags 进入 Stage 2 LLP
  calibration。