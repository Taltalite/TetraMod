import pysam, numpy as np

bam = "/data/biolab-nvme-pcie2/lijy/curlcakes/rna002_m6A/dorado_rna002_bam/cc100/cc100.mapped_only.bam"

rows = []
with pysam.AlignmentFile(bam, "rb") as fh:
    for r in fh:
        if r.is_unmapped or not r.has_tag("mv") or not r.has_tag("ns"):
            continue
        mv = list(r.get_tag("mv"))
        stride = int(mv[0])
        steps = len(mv) - 1
        move_span = steps * stride
        ts = int(r.get_tag("ts")) if r.has_tag("ts") else 0
        ns = int(r.get_tag("ns"))
        qlen = r.query_length or len(r.query_sequence or "")
        rows.append((qlen, ts, ns, move_span, ns - ts, move_span / max(ns - ts, 1), move_span / max(ns, 1)))
        if len(rows) >= 1000:
            break

arr = np.asarray(rows, dtype=float)
for i, name in enumerate(["qlen","ts","ns","move_span","ns_minus_ts","move/(ns-ts)","move/ns"]):
    print(name, np.percentile(arr[:, i], [0,25,50,75,100]).round(3).tolist())