# prepare_apa_features.py

import sys
import pandas as pd
import numpy as np
import yaml

# Inputs
dapars_file = sys.argv[1]
yaml_file = sys.argv[2]
output_file = sys.argv[3]

# Load config
with open(yaml_file, "r") as stream:
    config = yaml.safe_load(stream)

group1_samples = config["groups"]["group1"]
group2_samples = config["groups"]["group2"]

# Read DaPars output
df = pd.read_csv(dapars_file, sep="\t")

# Extract gene symbols from the 'Gene' column
df["GeneSymbol"] = df["Gene"].apply(lambda x: x.split("|")[1])

# Normalize column names (strip paths)
df.columns = [col.split("/")[-1].replace("_PDUI", "") if "PDUI" in col else col for col in df.columns]

# Select only PDUI columns + gene symbol
pdui_cols = [col for col in df.columns if col in group1_samples + group2_samples]
df_reduced = df[["GeneSymbol"] + pdui_cols]

# Aggregate by gene symbol (mean across isoforms, ignoring NA)
df_grouped = df_reduced.groupby("GeneSymbol").agg(lambda x: np.nanmean(x) if not x.isna().all() else 0)

# Compute APA difference
group1_mean = df_grouped[group1_samples].mean(axis=1)
group2_mean = df_grouped[group2_samples].mean(axis=1)
apa_diff = group1_mean - group2_mean

# Append _apa to feature names
apa_diff.index = apa_diff.index + "_apa"
apa_diff.index.name = "Feature"
# Save to file
apa_diff.to_csv(output_file, header=["value"])

