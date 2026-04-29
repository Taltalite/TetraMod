我已经把两个核心路径重新对齐过了。差异的核心可以一句话概括：ctc_like 对每个 chunk 重新拿 basecalled chunk sequence 去参考基因组局部比对；aligned_highmod 则信任整条 read 的 BAM CIGAR，把 chunk 内 query base 直
  接投影到 reference。下面我会给出算法图和决策建议。

• 可以，把两个脚本的核心差异理解成：

  - create_dataset_dorado_ctc_like.py: chunk-local re-alignment builder
  - create_dataset_bam_aligned_highmod.py: whole-read BAM CIGAR projection builder

  共同前半段
  两者前半段基本相同，都是从 Dorado BAM + POD5 恢复训练样本：

  Dorado BAM record
    ├─ query_sequence
    ├─ mv move table
    ├─ ts/ns basecalled signal interval
    ├─ pi/sp split-read parent info
    └─ qs/sm/sd optional quality/scaling tags

  POD5 raw signal
    ↓
  calibrated signal
    ↓
  signal[sp + ts : sp + ns]
    ↓
  normalise
    ↓
  fixed-length signal chunks

  关键共同点：

  - 都依赖 Dorado mv/ns tag。
  - 都用 sp + ts : sp + ns 找到 basecalled signal 区间。
  - 都用 move table 把 base emission 位置映射回 signal chunk。
  - 输出格式一致：chunks.npy, references.npy, reference_lengths.npy, metadata.npz。
  - 后续都可以接 make_mod_targets_m6a.py。

  ctc_like 算法
  代码位置：gen_data/create_dataset_dorado_ctc_like.py:515

  流程：

  one read
    ↓
  recover signal-order base sequence from Dorado query_sequence + strand
    ↓
  for each signal chunk:
      use move table to find emitted bases inside chunk
        ↓
      chunk query sequence
        ↓
      minimap2/mappy realign chunk sequence to reference
        ↓
      if mapping accuracy / coverage pass:
          fetch mapped reference interval
          orient reference to target direction
          write reference labels

  更具体：

  1. 从 BAM record 建 task，要求 primary/non-supplementary、有 mv 和 ns：gen_data/create_dataset_dorado_ctc_like.py:409
  2. 解码 move table，生成每个 emitted base 对应的 signal position：gen_data/create_dataset_dorado_ctc_like.py:556
  3. 每个 signal chunk 用 searchsorted 找出 chunk 内 basecalled bases：gen_data/create_dataset_dorado_ctc_like.py:589
  4. 把 chunk query sequence 重新 map 到 reference：gen_data/create_dataset_dorado_ctc_like.py:602
  5. 用 mapping.mlen / mapping.blen 作为 mapping_accuracy，用 query aligned span 作为 mapping_coverage：gen_data/create_dataset_dorado_ctc_like.py:608
  6. 通过阈值后，直接从 reference 取 mapping.r_st:r_en 作为 label sequence：gen_data/create_dataset_dorado_ctc_like.py:617

  示意图：

  signal chunk
     │
     ├─ move table → bases emitted in this chunk
     │
     └─ chunk basecalled sequence
            ↓
        minimap2 chunk-local map
            ↓
        reference interval from new chunk mapping
            ↓
        references.npy label

  优点：

  - 更接近 Bonito --save-ctc 的思想。
  - 每个 chunk 的 reference label 是由该 chunk 自己重新比对决定的。
  - 不强依赖整条 read BAM CIGAR 的连续性。
  - 对正常 basecall identity 的数据更干净。

  缺点：

  - 对高修饰/低 identity chunk 更脆弱。
  - 如果修饰比例升高导致 basecalled chunk sequence 和 reference 差异变大，会产生更多 chunks_no_mapping / chunks_low_accuracy。
  - 选择出来的数据可能偏向“容易被 basecaller 正确读出、容易 remap 的 chunk”。

  aligned_highmod 算法
  代码位置：gen_data/create_dataset_bam_aligned_highmod.py:1

  它的设计目标在文件注释里写得很明确：当 high-mod control 的 chunk-local remapping 太脆弱时，用 whole-read BAM alignment rescue。

  流程：

  one mapped BAM read
    ↓
  compute whole-read identity from NM/CIGAR
    ↓
  build query_pos → reference_pos map from BAM aligned pairs
    ↓
  for each signal chunk:
      use move table to find query bases inside chunk
        ↓
      project those query positions through BAM CIGAR to reference positions
        ↓
      derive min/max reference span
        ↓
      fetch that reference interval
        ↓
      filter by chunk aligned fraction + base identity
        ↓
      write reference labels

  更具体：

  1. 先要求 read 是 mapped primary，且有 mv/ns：gen_data/create_dataset_bam_aligned_highmod.py:168
  2. 用 NM 和 CIGAR 估算 whole-read identity：gen_data/create_dataset_bam_aligned_highmod.py:105
  3. 用 read.get_aligned_pairs() 建 query_pos -> ref_pos 表：gen_data/create_dataset_bam_aligned_highmod.py:118
  4. 每个 chunk 仍通过 move table 找 chunk 内 query positions：gen_data/create_dataset_bam_aligned_highmod.py:353
  5. 这些 query positions 通过 BAM CIGAR 投影到 reference：gen_data/create_dataset_bam_aligned_highmod.py:361
  6. 用 chunk aligned fraction、reference span、chunk base identity 过滤：gen_data/create_dataset_bam_aligned_highmod.py:366
  7. 通过后，从 whole-read BAM 投影出的 ref span 取 reference label：gen_data/create_dataset_bam_aligned_highmod.py:382

  示意图：

  whole-read BAM alignment
     ↓
  query_pos → ref_pos map
     ↓
  signal chunk
     ↓
  move table → query positions inside chunk
     ↓
  query positions → reference positions via BAM CIGAR
     ↓
  reference span
     ↓
  references.npy label

  优点：

  - 对高修饰导致的 chunk-local remap failure 更鲁棒。
  - 只要整条 read 的 BAM alignment 还能用，就能从 CIGAR 投影出 chunk labels。
  - 对 100% modified control 这种 extreme case 很有价值。

  缺点：

  - 更信任 Dorado 的 whole-read alignment。
  - chunk label 不是由 chunk 自己重新比对得到，而是从 whole-read alignment 截取。
  - 如果 whole-read CIGAR 局部错了，chunk label 会继承这个错误。
  - reference span 是 min/max aligned ref positions，可能跨过 chunk 内 indel/soft alignment 不连续区域。
  - 产生的数据选择偏向“whole-read alignment 可用”的 reads/chunks。

  核心差异表

  | 项目 | ctc_like | aligned_highmod |
  |---|---|---|
  | label 来源 | chunk sequence 重新比对 reference | whole-read BAM CIGAR 投影 |
  | 是否依赖 BAM 已 mapped | 不强依赖 existing CIGAR，自己 remap chunk | 强依赖 mapped BAM 和 CIGAR |
  | chunk 独立性 | 高，每个 chunk 自己 map | 较低，继承 whole-read alignment |
  | 高修饰鲁棒性 | 较弱 | 较强 |
  | label 保守性 | 高 identity chunk 更可靠 | 可 rescue 低 identity chunk |
  | 主要过滤指标 | chunk-local mlen/blen, query coverage | read identity/MAPQ + chunk aligned fraction/base identity |
  | metadata mapping_accuracy 含义 | minimap2 chunk mlen/blen | chunk base identity vs BAM-derived ref |
  | metadata mapping_coverage 含义 | chunk-local query coverage | chunk query bases with aligned BAM ref positions |
  | 风险 | 丢掉高修饰/难比对数据 | 引入 whole-read alignment bias |

  为什么 Stage 2 不应按 ratio 混 builder
  如果这样做：

  12.5/25 → ctc_like
  50/75   → aligned_highmod

  LLP 看到的是：

  low ratio  ≈ builder A distribution
  high ratio ≈ builder B distribution

  这会让比例标签和数据制作方法绑定。Stage 2 只有 bag-level weak labels，没有 read-level truth，模型可能学到 builder artifact，而不是 m6A proportion calibration。

  更稳的实验设计是：

  方案 A:
  12.5/25/50/75 全部 ctc_like，统一阈值

  方案 B:
  12.5/25/50/75 全部 aligned_highmod，统一阈值

  然后分别 diagnose + LLP train + evaluate

  我的建议
  你可以把 aligned_highmod 当成一条 parallel rescue pipeline，但不要只用于 50/75。更合理的是：

  1. 先确认 ctc_like 的掉样原因：
     看每个 ratio 的 dataset_summary.json 中 chunks_low_accuracy, chunks_no_mapping, chunks_written。
  2. 跑一套全比例 ctc_like_relaxed_uniform：
     所有 ratio 统一 --min-accuracy 0.90 --min-coverage 0.80 或类似参数。
  3. 跑一套全比例 aligned_highmod_all_ratios：
     12.5/25/50/75 全部使用 aligned_highmod，同样统一参数。
  4. 两套都跑 diagnose_llp_dataset.py，比较：
     bag_size, mapping_coverage, mapping_accuracy, reference_length, candidate_a_sites, kmer_context, motif_context。

  最后选一套进入 Stage 2。当前 75 的主要问题不是 mapping coverage 坏，而是可用 read/bag 太少，所以“统一 builder + 调整 bagging/阈值”比“只给 50/75 换 builder”更干净。