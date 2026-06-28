import sys
import pandas as pd

gtf_path = sys.argv[1]
output = sys.argv[2]

genes = {}

with open(gtf_path) as f:
    for line in f:
        if line.startswith("#"): continue
        fields = line.strip().split('\t')
        if fields[2] != "exon": continue
        attrs = {x.split()[0]: x.split()[1].strip('"') for x in fields[8].split(';') if x.strip()}
        gene_id = attrs.get("gene_id")
        gene_name = attrs.get("gene_name", "")
        start, end = int(fields[3]), int(fields[4])
        if gene_id not in genes:
            genes[gene_id] = {"GeneSymbol": gene_name, "intervals": []}
        genes[gene_id]["intervals"].append((start, end))


def union_length(intervals):
    # Exons from different transcripts of the same gene overlap; merge them
    # before summing so shared bases aren't counted more than once.
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return sum(end - start + 1 for start, end in merged)


df = pd.DataFrame([
    {"Geneid": gid, "GeneSymbol": info["GeneSymbol"], "Length": union_length(info["intervals"])}
    for gid, info in genes.items()
])
df.to_csv(output, sep="\t", index=False)

