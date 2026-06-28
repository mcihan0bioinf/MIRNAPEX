import sys
import pandas as pd

sample_names = sys.argv[1].split(",")
input_files = sys.argv[2:-1]
output = sys.argv[-1]

merged = None
for sample, path in zip(sample_names, input_files):
    df = pd.read_csv(path, sep="\t")
    df = df[["Geneid", "GeneSymbol", "TPM"]].rename(columns={"TPM": sample})
    if merged is None:
        merged = df
    else:
        merged = merged.merge(df, on=["Geneid", "GeneSymbol"])

merged.to_csv(output, index=False)

