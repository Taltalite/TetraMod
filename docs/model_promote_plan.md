可以结合，而且你的现有架构其实很适合弱监督：**Bonito 官方 encoder 负责“稳定提取纳米孔信号表征”，mod trunk/head 负责“在弱标签约束下学习修饰 posterior”**。重点不是先把 encoder 大幅改掉，而是把训练目标从“read-level 强监督 BCE”改成“control 强监督 + bag-level 弱监督 + consistency/self-supervised representation”。

下面我按可落地优先级说。

---

## 1. 先明确你的架构应该怎样抽象

你现在可以把模型写成：

```text
raw signal
  ↓
Bonito pretrained encoder E   # 冻结
  ↓
hidden representation H
  ↓
mod trunk Gθ                  # 可训练
  ↓
A/C/G/T mod heads             # 可训练
  ↓
per-read, per-site modification probability
```

这个设计和 Remora 的思想接近：Remora 的 mod calling 是把修饰碱基预测锚定到 canonical basecall 或 reference sequence 上，训练/预测单元是固定长度 chunk，包括 normalized signal、canonical bases，以及 signal-base mapping/move table。([GitHub][1]) Bonito 本身也是 ONT 的 PyTorch research basecaller，用于 basecaller training 和方法开发。([GitHub][2])

所以你的核心训练样本最好不是整条 read，而是：

```text
一个 read 在某个候选位点附近的 signal chunk + base/move alignment + focus base
```

然后根据 focus base 选择对应 head：

```python
if focus_base == "A":
    p = A_head(G(E(signal_chunk)))
elif focus_base == "C":
    p = C_head(...)
...
```

如果你现在只做 m6A，**建议先只训练 A-head**。C/G/T head 可以保留架构，但不要强行用“没有 C/G/T 修饰标签”去训练它们，否则会把“未观测到修饰”误当作“确定未修饰”。

---

## 2. 你不一定要先“自监督训练 Bonito encoder”

你提到“如何先做自监督 signal encoder”。这里要分清三种层级。

### 层级 A：最推荐，冻结 Bonito encoder，只对 mod trunk 做自监督预训练

这是最稳的路线。因为 Bonito encoder 已经在 ONT 大规模数据上学过 raw signal → sequence 的表征，你现在真正缺的是：**mod trunk 如何从这些 hidden features 里提取修饰相关的细微信号差异**。

做法是：

```text
raw signal → frozen Bonito encoder → hidden tokens H
                                      ↓
                              mod trunk Gθ
                                      ↓
                              SSL projection head
```

自监督任务可以设计为：

### 任务 1：同一 chunk 的增强一致性

对同一个 read-site chunk 做两种轻微增强：

* signal jitter；
* random crop；
* scaling；
* local masking；
* time shift；
* normalization perturbation。

要求两个 view 的 mod trunk 表征接近：

```text
z1 = G(E(aug1(signal)))
z2 = G(E(aug2(signal)))

contrastive / cosine loss: z1 ≈ z2
```

ICLR 2024 的 SoftCLT 指出，时间序列 contrastive learning 中不应该把相似时间序列或相邻时间点粗暴当成负样本，因此它使用 soft assignment 的 instance-wise 和 temporal contrastive loss。这个思想很适合 nanopore signal，因为相邻位置、相似 k-mer、同一 read 的不同 crop 往往不是绝对负样本。([OpenReview][3])

### 任务 2：masked hidden feature reconstruction

不是直接重建 raw signal，而是重建 frozen Bonito encoder 的 hidden feature：

```text
H = E(signal)
H_masked = mask_span(H)

Gθ(H_masked) → reconstruct H_original
```

loss 可以用：

```text
MSE / cosine loss between reconstructed hidden features and stop-gradient H
```

这样做的好处是：

* 不需要改 Bonito encoder；
* 不会破坏 basecalling 能力；
* mod trunk 被迫学习局部上下文和时序结构；
* 训练数据可以使用大量 unlabeled POD5。

### 任务 3：k-mer/context-aware contrastive learning

对同一 focus base、同一 k-mer context 的 reads，表征应该有一定相似性；但不能完全拉得太近，因为它们可能一个修饰、一个未修饰。

因此可以设置成 soft positive，而不是 hard positive：

```text
same read, two augmentations        strong positive
same site, different reads          medium positive
same k-mer, different genomic site  weak positive
different k-mer                     negative or weak negative
```

这比普通 SimCLR 更适合修饰检测，因为修饰信号强烈依赖 k-mer context。

---

## 3. 层级 B：加一个小型 raw-signal branch，而不是改 Bonito encoder

如果你担心 frozen Bonito encoder 在 basecalling 训练中已经丢掉了一些修饰相关细节，可以加一个并行的小 CNN/Conformer branch：

```text
raw signal ───────────────→ small SSL signal encoder Sφ ─┐
                                                         ├→ mod trunk → heads
raw signal → frozen Bonito encoder E → hidden H ─────────┘
```

这个小分支可以自监督训练：

* masked raw signal reconstruction；
* local current statistics prediction；
* dwell time / event length prediction；
* signal residual prediction；
* contrastive learning。

这样做比直接解冻 Bonito encoder安全，因为：

1. Bonito encoder 继续保持官方 basecalling 表征；
2. 小分支专门捕捉修饰相关的 raw signal residual；
3. 后续可以通过 gating/fusion 判断模型到底依赖哪一部分。

推荐 fusion 方式：

```python
h_bonito = bonito_encoder(signal).detach()
h_ssl = ssl_signal_encoder(signal)

h = concat([h_bonito, h_ssl])
p_mod = mod_head(mod_trunk(h))
```

如果你的 GPU/数据量有限，这个方案比全量 self-supervised pretraining Bonito encoder 更现实。

---

## 4. 层级 C：只加 adapter/LoRA，不全量解冻 Bonito encoder

如果你确实想让 encoder 适应修饰检测，不建议一开始全量 fine-tune。可以做：

```text
Bonito encoder block
  + small adapter / LoRA
```

训练时：

```text
原始 Bonito 参数 frozen
adapter/LoRA 可训练
mod trunk/head 可训练
```

阶段顺序：

1. 冻结 Bonito encoder，只训练 mod trunk/head；
2. 加 SSL loss 训练 mod trunk；
3. 加 weak annotation loss 训练 mod trunk/head；
4. 如果仍然不够，再开放最后 1–2 层 adapter；
5. 最后才考虑小学习率解冻 encoder 最后几层。

全量解冻的风险是：模型为了适配全修饰/未修饰数据，会破坏原本稳定的 basecalling representation，并且更容易学习 batch effect。

---

## 5. 弱标注应该怎样接到你现有 4-head 架构上

你现在不能把所有全修饰 reads 直接当作野生型 read-level 正样本来训练到底。更好的训练目标是组合式 loss：

```text
L = L_control + λ1 L_LLP + λ2 L_MIL + λ3 L_consistency + λ4 L_domain
```

### 5.1 control loss：只作为锚点，不作为全部训练目标

对于全修饰样本和 IVT/未修饰样本：

```text
全修饰 A-site read chunk → y = 1
未修饰 A-site read chunk → y = 0
```

可以用普通 BCE：

```python
loss_control = BCE(p_read, y_control)
```

但这里有一个重要限制：**control loss 只能教模型“两个极端条件如何区分”，不能保证模型在野生型 read 上校准正确**。

所以 control loss 的权重不能太大，且必须做：

* batch-balanced sampling；
* same k-mer context balancing；
* same run / different run validation；
* leave-one-run-out；
* leave-one-transcript/site-out。

否则模型可能学到“这个实验批次像全修饰”，而不是“这条 read 的这个 A 位点像 m6A”。

---

### 5.2 LLP loss：最适合你的 mixture 数据

如果你能构造 0%、25%、50%、75%、100% 的混合样本，或者用外部方法得到某个位点的修饰比例，就可以用 Learning from Label Proportions。

ICLR 2024 的 LLP 工作把问题定义为：训练时只有 bag-level aggregate labels，但目标是学到 instance-level classifier；他们还用 belief propagation 结合 covariate similarity 和 bag-level label 生成 pseudo labels。([OpenReview][4])

在你的任务中：

```text
bag = 同一个 site / 同一个 transcript position / 同一个 sample 下的一组 reads
bag label = 该 bag 的已知 m6A 比例 r
instance = 单条 read 的该位点 chunk
model output = p_i
```

训练目标：

```python
p_bag = mean(p_i for i in reads_at_this_site)
loss_llp = BCE(p_bag, ratio)
# 或 MSE(p_bag, ratio)
```

例如：

```text
某 site 在 50% mixture 中有 100 条 reads
模型不需要知道哪 50 条是 modified
只需要满足 mean(p_read) ≈ 0.5
```

这比给 50% mixture 的 read 随机打标签要合理得多。

---

### 5.3 MIL loss：适合 site-level gold standard

m6Anet 的核心启发就是这里。m6Anet 明确处理的是：实验方法通常只能提供 site-level training data，而缺失每条 RNA molecule/read 的修饰状态；它用 multiple instance learning 同时学习 read-level encoder 和 site-level classification。([Nature][5])

你的模型可以这样接：

```text
read chunk → Bonito encoder → mod trunk → p_read
多个 reads at same site → pooling → p_site
site-level label → loss
```

pooling 有几种选择：

#### 方案 1：mean pooling

适合位点修饰比例：

```python
p_site = mean(p_read)
```

#### 方案 2：attention pooling

适合 site-level binary label：

```python
weight_i = attention(z_i)
p_site = sum(weight_i * p_read_i)
```

#### 方案 3：noisy-OR

适合“只要有一部分 read 修饰，该 site 就算阳性”的定义：

```python
p_site = 1 - product(1 - p_read_i)
```

但对 m6A stoichiometry，我更建议优先使用 **mean pooling**，因为 read-level posterior 的平均值天然对应位点修饰比例。

---

### 5.4 consistency loss：让模型不要过度依赖噪声

对同一 read-site chunk 做两种增强：

```python
p1 = model(aug1(chunk))
p2 = model(aug2(chunk))

loss_consistency = MSE(p1, p2)
```

它的作用是稳定 read-level posterior，尤其适合中间置信度 reads。

---

### 5.5 domain adversarial loss：防止学到批次差异

你的场景里最大风险之一是：

```text
全修饰样本 vs IVT样本
```

不只差在 m6A，还差在建库、run、basecalling quality、read length、sequence distribution 等。

可以在 mod trunk 后面加一个 domain classifier：

```text
z = mod_trunk(...)
domain_head(z) → sample/run/batch id
gradient reversal
```

目标是：

* mod head 能预测修饰；
* domain head 不能从 z 预测样本来源。

这样可以减少 batch effect。

---

## 6. 推荐的训练流程

我建议你不要一上来做复杂 EM/RL，而是按下面阶段推进。

### 阶段 0：保留当前架构，做强基线

```text
E frozen
Gθ + A-head trainable
loss = BCE on full-mod vs IVT
```

目的不是得到最终模型，而是确认：

1. A-head 能否区分全修饰 vs 未修饰；
2. leave-one-run-out 是否崩；
3. 不同 k-mer context 是否泛化；
4. 模型输出是否在 0/25/50/75/100% mixture 上单调。

如果 0/100 都分不开，说明 mod trunk/head 或数据对齐有问题。
如果 0/100 很好，但 50% mixture 和野生型很差，说明是弱标注和 domain shift 问题。

---

### 阶段 1：冻结 Bonito encoder，对 mod trunk 做 SSL 预训练

输入所有 unlabeled reads：

```text
POD5/BAM/move table → chunks → frozen Bonito encoder → hidden H
```

训练：

```text
Gθ + projection head
```

loss：

```text
masked hidden reconstruction
+ time-series contrastive / SoftCLT-style contrastive
+ same-read augmentation consistency
```

此阶段不训练 mod head，或者只训练 projection head。

训练结束后丢弃 projection head，保留：

```text
frozen Bonito encoder + SSL-pretrained mod trunk
```

---

### 阶段 2：control supervised warm-up

加载阶段 1 的 mod trunk：

```text
loss = BCE(full-mod, IVT)
```

但要严格做 balanced sampling：

```text
same focus base
same k-mer
same coverage range
same read quality range
same run split
```

这里建议先只训练 A-head。

---

### 阶段 3：加入 LLP / mixture proportion loss

构造 bag：

```text
bag_key = (site_id, sample_id, mixture_ratio)
```

每个 bag 采样 K 条 reads，例如 K=20/50/100。

训练目标：

```python
p_reads = model(read_chunks)       # [K]
p_bag = p_reads.mean()
loss_prop = BCE(p_bag, ratio)
```

这一步是你从“极端标签”走向“read-level posterior”的关键。

---

### 阶段 4：加入野生型 site-level MIL

如果你有野生型 gold standard site-level BED，例如来自 GLORI、miCLIP、m6A-seq、SELECT 或可信数据库，那么：

```text
bag = wildtype 中某个位点的所有 reads
label = site-level positive/negative 或 modification ratio
```

训练：

```python
loss_mil = BCE(mean(p_reads), y_site)
```

如果是连续比例：

```python
loss_mil = MSE(mean(p_reads), site_ratio)
```

这一步可以让模型学会在真实野生型分布上校准。

---

### 阶段 5：高置信 pseudo-label refinement

最后才考虑 pseudo-label：

```text
p_read > 0.95 → pseudo positive
p_read < 0.05 → pseudo negative
中间区域不打 hard label
```

然后用 teacher-student 或 EMA model 迭代。

不要过早做 pseudo-label，否则会把初始模型的偏差放大。

---

## 7. 自监督 encoder 的最小实现建议

你可以先实现一个非常实用的版本：

```text
bonito_encoder.eval()
for p in bonito_encoder.parameters():
    p.requires_grad = False

mod_trunk.train()
ssl_projector.train()
```

数据流：

```python
signal_chunk = batch["signal"]          # [B, T]
H = bonito_encoder(signal_chunk).detach()

H1 = augment_hidden(H)
H2 = augment_hidden(H)

z1 = projector(mod_trunk(H1))
z2 = projector(mod_trunk(H2))

loss_ssl = contrastive_or_cosine_loss(z1, z2)
```

增强方式：

```python
augment_hidden(H):
    - random temporal crop
    - random span mask
    - gaussian noise on hidden features
    - dropout channels
    - slight time shift
```

更接近 raw signal 的版本：

```python
signal1 = augment_signal(signal)
signal2 = augment_signal(signal)

H1 = bonito_encoder(signal1).detach()
H2 = bonito_encoder(signal2).detach()

z1 = projector(mod_trunk(H1))
z2 = projector(mod_trunk(H2))
loss = contrastive(z1, z2)
```

但注意：如果 signal augmentation 太强，frozen Bonito encoder 可能产生异常表征，所以增强要轻。

---

## 8. 一个推荐的最终训练目标

可以写成：

```python
loss = (
    1.0 * loss_control
    + 0.5 * loss_ssl
    + 1.0 * loss_llp
    + 1.0 * loss_mil
    + 0.2 * loss_consistency
    + 0.1 * loss_domain_adv
)
```

实际训练时不要一开始全部打开，建议逐步加：

```text
Stage 1: SSL only
Stage 2: control BCE
Stage 3: control BCE + LLP
Stage 4: control BCE + LLP + MIL
Stage 5: + consistency + pseudo-label
Stage 6: + adapter/LoRA if necessary
```

---

## 9. 我对你当前架构的具体判断

你的设计方向是对的，尤其是：

```text
冻结 ONT pretrained basecaller encoder
只训练 mod trunk + mod head
```

这在数据量有限、标签弱、容易过拟合的情况下是合理的。

但我会调整两点：

### 第一，不要把“全修饰/未修饰”当成最终 read-level 强标签体系

它只能作为 anchor。真正让模型输出 read-level posterior 的训练信号应该来自：

```text
mixture proportion loss
+ site-level MIL loss
+ high-confidence pseudo-label
```

### 第二，自监督不一定要重新训练官方 encoder

更推荐：

```text
frozen Bonito encoder
+ SSL-pretrained mod trunk
+ optional small raw-signal branch
+ optional adapter/LoRA
```

这样既保留 ONT 官方模型的大规模预训练优势，又能让你的 mod branch 学到修饰相关的 signal residual。

一句话总结：

> 你现在的架构可以保留，但训练逻辑应从“read-level BCE 分类器”升级为“Bonito frozen representation + mod trunk SSL + LLP/MIL weak supervision + mixture calibration”的框架。这样比直接追求 read-level 强标注更符合纳米孔修饰检测的真实约束。

[1]: https://github.com/nanoporetech/remora "GitHub - nanoporetech/remora: Methylation/modified base calling separated from basecalling. · GitHub"
[2]: https://github.com/nanoporetech/bonito "GitHub - nanoporetech/bonito: A PyTorch Basecaller for Oxford Nanopore Reads · GitHub"
[3]: https://openreview.net/forum?id=pAsQSWlDUf "Soft Contrastive Learning for Time Series | OpenReview"
[4]: https://openreview.net/forum?id=KQe9tHd0k8 "Learning from Label Proportions: Bootstrapping Supervised Learners via Belief Propagation | OpenReview"
[5]: https://www.nature.com/articles/s41592-022-01666-1 "Detection of m6A from direct RNA sequencing using a multiple instance learning framework | Nature Methods"
