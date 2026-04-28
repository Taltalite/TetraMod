import pysam, numpy as np

bam = "/data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/dorado_rna002_bam/cc100/cc100.mapped_only.bam"

covs = []
softs = []
mapqs = []
qlens = []
rev = 0
n = 0
examples = []

with pysam.AlignmentFile(bam, "rb") as fh:
    for r in fh:
        if r.is_unmapped or r.is_secondary or r.is_supplementary:
            continue
        qlen = r.query_length or len(r.query_sequence or "")
        if qlen == 0:
            continue
        aln = r.query_alignment_length or 0
        soft = 0
        if r.cigartuples:
            soft = sum(length for op, length in r.cigartuples if op == 4)
        covs.append(aln / qlen)
        softs.append(soft / qlen)
        mapqs.append(r.mapping_quality)
        qlens.append(qlen)
        rev += int(r.is_reverse)
        n += 1
        if len(examples) < 20:
            examples.append((r.query_name, r.flag, r.mapping_quality, qlen, aln, soft, r.cigarstring))

def pct(x):
    x = np.asarray(x, dtype=float)
    return np.percentile(x, [0, 5, 25, 50, 75, 95, 100]).round(4).tolist()

print("mapped primary reads:", n)
print("reverse fraction:", rev / max(n, 1))
print("query length percentiles:", pct(qlens))
print("alignment coverage percentiles:", pct(covs))
print("soft clip fraction percentiles:", pct(softs))
print("MAPQ percentiles:", pct(mapqs))
print("\nexamples:")
for e in examples:
    print(e)