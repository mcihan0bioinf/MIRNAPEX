import sys
import pandas as pd

input_files = sys.argv[1:-1]
output = sys.argv[-1]

merged = None
for path in input_files:
    sample = path.split("/")[-1].split("_")[0]
    df = pd.read_csv(path, sep="\t")
    df = df[["Geneid", "GeneSymbol", "TPM"]].rename(columns={"TPM": sample})
    if merged is None:
        merged = df
    else:
        merged = merged.merge(df, on=["Geneid", "GeneSymbol"])

merged.to_csv(output, index=False)

