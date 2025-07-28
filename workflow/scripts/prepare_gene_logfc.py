import pandas as pd
import numpy as np
import argparse
import yaml

parser = argparse.ArgumentParser(description="Compute gene logFC from TPM matrix using group definitions.")
parser.add_argument("tpm_matrix", help="Path to TPM matrix CSV")
parser.add_argument("config_file", help="Path to YAML config")
parser.add_argument("output_file", help="Path to save logFC features CSV")
args = parser.parse_args()

# Load config.yaml to get group definitions
with open(args.config_file) as f:
    config = yaml.safe_load(f)

group1 = config["groups"]["group1"]
group2 = config["groups"]["group2"]

# Load TPM matrix
df = pd.read_csv(args.tpm_matrix)

# Extract gene symbols
genes = df["GeneSymbol"]

# Compute means for each group
group1_mean = df[group1].mean(axis=1)
group2_mean = df[group2].mean(axis=1)

# Compute log2 fold change
logfc = np.log2((group1_mean + 1) / (group2_mean + 1))
features = genes.values + "_gene"
out_df = pd.DataFrame(logfc.values, index=features, columns=["value"])
out_df.index.name = "Feature"
out_df.to_csv(args.output_file)

