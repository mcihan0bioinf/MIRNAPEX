import sys
import pandas as pd

counts_file = sys.argv[1]
metadata_file = sys.argv[2]
output_file = sys.argv[3]

df = pd.read_csv(counts_file, sep="\t", header=None)
df = df[~df[0].str.startswith("N_")]
df.columns = ["Geneid", "Unstranded", "Strand1", "Strand2"]
df["Counts"] = df["Unstranded"]

meta = pd.read_csv(metadata_file, sep="\t")
df = df.merge(meta, on="Geneid")

df["Length_kb"] = df["Length"] / 1000
df["RPK"] = df["Counts"] / df["Length_kb"]
df["TPM"] = df["RPK"] / df["RPK"].sum() * 1e6

df_out = df[["Geneid", "GeneSymbol", "TPM"]]
df_out.to_csv(output_file, sep="\t", index=False)

