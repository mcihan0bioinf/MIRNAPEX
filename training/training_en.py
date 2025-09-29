# Reproducible EN training with high/low stratified pairing in both directions.
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
  mean_absolute_error,
  median_absolute_error,
  r2_score,
  mean_squared_error,
)
import joblib

np.random.seed(42)


def generate_pairs(sample_list, num_pairs, mirna_name, isomir_df):
  """
    Stratify samples by miRNA expression (median split) and create random pairs
    in BOTH directions (high→low and low→high), all within the provided sample_list.
    """
isomir_subset = isomir_df[sample_list]

if mirna_name not in isomir_subset.index:
  raise ValueError(f"miRNA {mirna_name} not found in the dataset.")

mirna_values = isomir_subset.loc[mirna_name]
median_value = mirna_values.median() + 0.001  # small jitter to avoid ties

highs = mirna_values[mirna_values >= median_value].index.tolist()
lows = mirna_values[mirna_values < median_value].index.tolist()

pairs = {}

# High -> Low
for s in highs:
  k = min(num_pairs, len(lows))
if k == 0:
  pairs[s] = []
continue
pairs[s] = np.random.choice(lows, k, replace=False).tolist()

# Low -> High
for s in lows:
  k = min(num_pairs, len(highs))
if k == 0:
  pairs[s] = []
continue
pairs[s] = np.random.choice(highs, k, replace=False).tolist()

return pairs


def compute_pairwise_differences(pairs, apa_df, gene_df, isomir_df):
  """
    For each (s1, s2) in pairs, compute:
      - APA difference: APA[s1] - APA[s2]
      - Gene log2 fold-change: log2((gene[s1]+1)/(gene[s2]+1))
      - miRNA log2 fold-change: log2((isomir[s1]+1)/(isomir[s2]+1))
    Returns three DataFrames with pair columns.
    """
apa_diff, gene_logfc, mirna_logfc = {}, {}, {}

for s1, s2_list in pairs.items():
  for s2 in s2_list:
  pair_name = f"{s1}_{s2}"
apa_diff[pair_name] = apa_df[s1] - apa_df[s2]
gene_logfc[pair_name] = np.log2((gene_df[s1] + 1) / (gene_df[s2] + 1))
mirna_logfc[pair_name] = np.log2((isomir_df[s1] + 1) / (isomir_df[s2] + 1))

return pd.DataFrame(apa_diff), pd.DataFrame(gene_logfc), pd.DataFrame(mirna_logfc)


def main():
  # -------------------------
# Args (stay close to original)
# -------------------------
if len(sys.argv) < 3:
  print("Usage: python train_elasticnet_mirna.py <batch:int> <top_x_genes:int>")
sys.exit(1)

n = 1  # number of partner pairs per sample (kept from original)
batch = int(sys.argv[1])
top_x_genes = int(sys.argv[2])

# -------------------------
# Relative paths (adjust if needed)
# -------------------------
data_dir = Path("data/input")
out_dir = Path("outputs")
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "final").mkdir(parents=True, exist_ok=True)
(out_dir / "models").mkdir(parents=True, exist_ok=True)

apa_path     = data_dir / "apa.csv" #Stores samples in columns, Genes in rows and PDUI values 
gene_path    = data_dir / "gene.csv" #Stores samples in columns, Genes in rows and TPM-values 
isomir_path  = data_dir / "isomir.csv"#Stores samples in columns, MicroRNAs in rows and RPM-values
feature_path = data_dir / "microt_filtered_1000_features.csv" #Stores MicroRNAs in columns and top 1000 genes with highest microT values in values 
mirna_file   = data_dir / f"mirnas_batch_{batch}.csv" #Stores MicroRNAs in columns- can be feature_path file split in multiple batches for parallel processing


# -------------------------
# Load data
# -------------------------
apa_df = pd.read_csv(apa_path, index_col=0)
gene_df = pd.read_csv(gene_path, index_col=0)
isomir_df = pd.read_csv(isomir_path, index_col=0)
feature_df = pd.read_csv(feature_path, index_col=0)

samples = isomir_df.columns.tolist()
mirna_selection = pd.read_csv(mirna_file, index_col=0)
mirnas = mirna_selection.columns.tolist()

# -------------------------
# Split train/test on samples (no leakage)
# -------------------------
train_samples, test_samples = train_test_split(samples, test_size=0.2, random_state=42)

# CV grid
alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
l1_ratios = [0, 0.25, 0.5, 0.75, 1.0]

all_results = []

print("Start training...")
for miRNA in mirnas:
  print(f"[miRNA] {miRNA}")

# -------------------------
# Pairing (test and train folds) — both directions
# -------------------------
test_pairs = generate_pairs(test_samples, n, miRNA, isomir_df)

# Make 5 folds (disjoint) from training samples
folds = [list(arr) for arr in np.array_split(np.random.permutation(train_samples), 5)]
train_pairs = {i: generate_pairs(folds[i], n, miRNA, isomir_df) for i in range(5)}

# -------------------------
# Compute pairwise features
# -------------------------
test_apa_diff_g, test_gene_logfc_g, test_mirna_logfc_g = compute_pairwise_differences(
  test_pairs, apa_df, gene_df, isomir_df
)

train_apa_diff_g, train_gene_logfc_g, train_mirna_logfc_g = {}, {}, {}
for i in range(5):
  train_apa_diff_g[i], train_gene_logfc_g[i], train_mirna_logfc_g[i] = compute_pairwise_differences(
    train_pairs[i], apa_df, gene_df, isomir_df
  )

# -------------------------
# Feature selection per miRNA
# -------------------------
# Keep behavior close to original: use feature_df[miRNA] values (top N)
if miRNA not in feature_df.columns:
  print(f"  - Skipping {miRNA}: not found in feature file columns.")
continue

selected_features = feature_df[miRNA].dropna().values.tolist()[:top_x_genes]
if len(selected_features) == 0:
  print(f"  - Skipping {miRNA}: no features after selection.")
continue

selected_features_apa = [f"{g}_apa" for g in selected_features]
selected_features_gene = [f"{g}_gene" for g in selected_features]

# Build test matrices
test_apa_diff = test_apa_diff_g.loc[test_apa_diff_g.index.isin(selected_features_apa)]
test_gene_logfc = test_gene_logfc_g.loc[test_gene_logfc_g.index.isin(selected_features_gene)]
X_test = pd.concat([test_apa_diff, test_gene_logfc], axis=0).T
y_test = test_mirna_logfc_g.loc[miRNA].T

if X_test.shape[1] == 0:
  print(f"  - Skipping {miRNA}: no matching test features.")
continue

# Build train folds
X_train_folds, y_train_folds = [], []
for i in range(5):
  tr_apa = train_apa_diff_g[i].loc[train_apa_diff_g[i].index.isin(selected_features_apa)]
tr_gene = train_gene_logfc_g[i].loc[train_gene_logfc_g[i].index.isin(selected_features_gene)]
X_train_folds.append(pd.concat([tr_apa, tr_gene], axis=0).T)
y_train_folds.append(train_mirna_logfc_g[i].loc[miRNA])

# -------------------------
# 5-fold CV to pick alpha & l1_ratio
# -------------------------
best_alpha, best_l1_ratio = None, None
best_r2, best_mse = -np.inf, np.inf

for alpha in alphas:
  for l1_ratio in l1_ratios:
  r2_scores, mse_scores = [], []

for i in range(5):
  X_val = X_train_folds[i]
y_val = y_train_folds[i].T

X_tr = pd.concat([X_train_folds[j] for j in range(5) if j != i], axis=0)
y_tr = pd.concat([y_train_folds[j] for j in range(5) if j != i], axis=0)

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)
X_val = scaler.transform(X_val)

model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=100, warm_start=False)
model.fit(X_tr, y_tr)

y_pred_val = model.predict(X_val)
r2_scores.append(r2_score(y_val, y_pred_val))
mse_scores.append(mean_squared_error(y_val, y_pred_val))

avg_r2 = float(np.mean(r2_scores))
avg_mse = float(np.mean(mse_scores))

if avg_r2 > best_r2:
  best_r2 = avg_r2
best_mse = avg_mse
best_alpha = alpha
best_l1_ratio = l1_ratio

# -------------------------
# Train final model on all training folds and evaluate on test
# -------------------------
X_train_all = pd.concat(X_train_folds, axis=0)
y_train_all = pd.concat(y_train_folds, axis=0)
feature_names_used = X_train_all.columns.tolist()

scaler = StandardScaler()
X_train_all = scaler.fit_transform(X_train_all)
X_test_scaled = scaler.transform(X_test)

final_model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio, warm_start=False)
final_model.fit(X_train_all, y_train_all)

y_pred_test = final_model.predict(X_test_scaled)

# Metrics
res = {
  "miRNA Name": miRNA,
  "Best Alpha": best_alpha,
  "Best L1 Ratio": best_l1_ratio,
  "Test R² Score": r2_score(y_test, y_pred_test),
  "Test MSE": mean_squared_error(y_test, y_pred_test),
  "Test Mean Absolute Error": mean_absolute_error(y_test, y_pred_test),
  "Test Median Absolute Error": median_absolute_error(y_test, y_pred_test),
  "Mean Y (Test)": float(np.mean(y_test)),
  "Mean Y Pred (Test)": float(np.mean(y_pred_test)),
  "Std Y Test": float(np.std(y_test)),
  "Std Y Pred Test": float(np.std(y_pred_test)),
  "CV best R² Score": best_r2,
  "CV MSE": best_mse,
}
all_results.append(res)

# Save per-miRNA model bundle (model + scaler + feature names)
model_bundle = {
  "model": final_model,
  "scaler": scaler,
  "feature_names": feature_names_used,
}
joblib.dump(model_bundle, out_dir / "models" / f"elasticnet_{miRNA}.pkl")

# -------------------------
# Save summary CSV
# -------------------------
final_results_df = pd.DataFrame(all_results)
out_csv = out_dir / "final" / f"elasticnet_top{top_x_genes}_n_{n}_batch_{batch}.csv"
final_results_df.to_csv(out_csv, index=False)
print(f"Done. Wrote results to: {out_csv}")


if __name__ == "__main__":
  main()
