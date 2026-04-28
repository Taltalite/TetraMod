import pysam, numpy as np

bam = "/data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/dorado_rna002_bam/cc100/cc100.mapped_only.bam"

idents = []
mapqs = []
with pysam.AlignmentFile(bam, "rb") as fh:
    for r in fh:
        if r.is_unmapped or r.is_secondary or r.is_supplementary:
            continue
        if not r.has_tag("NM") or not r.cigartuples:
            continue
        nm = int(r.get_tag("NM"))
        denom = 0
        for op, ln in r.cigartuples:
            if op in (0, 1, 2, 7, 8):  # M/I/D/=/X
                denom += ln
        if denom:
            idents.append(1.0 - nm / denom)
            mapqs.append(r.mapping_quality)

def pct(x):
    x = np.asarray(x, dtype=float)
    return np.percentile(x, [0,5,25,50,75,95,100]).round(4).tolist()

print("n", len(idents))
print("identity percentiles", pct(idents))
print("mapq percentiles", pct(mapqs))