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
        length = end - start + 1
        if gene_id not in genes:
            genes[gene_id] = {"GeneSymbol": gene_name, "Length": 0}
        genes[gene_id]["Length"] += length

df = pd.DataFrame([
    {"Geneid": gid, "GeneSymbol": info["GeneSymbol"], "Length": info["Length"]}
    for gid, info in genes.items()
])
df.to_csv(output, sep="\t", index=False)

