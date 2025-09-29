# OLS training with high/low stratified pairing in both directions.
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    mean_squared_error,
)

np.random.seed(42)


def generate_pairs(sample_list, num_pairs, mirna_name, isomir_df):
    isomir_subset = isomir_df[sample_list]
    if mirna_name not in isomir_subset.index:
        raise ValueError(f"miRNA {mirna_name} not found in the dataset.")
    mirna_values = isomir_subset.loc[mirna_name]
    median_value = mirna_values.median() + 0.001
    highs = mirna_values[mirna_values >= median_value].index.tolist()
    lows = mirna_values[mirna_values < median_value].index.tolist()
    pairs = {}
    for s in highs:
        k = min(num_pairs, len(lows))
        pairs[s] = np.random.choice(lows, k, replace=False).tolist() if k > 0 else []
    for s in lows:
        k = min(num_pairs, len(highs))
        pairs[s] = np.random.choice(highs, k, replace=False).tolist() if k > 0 else []
    return pairs


def compute_pairwise_differences(pairs, apa_df, gene_df, isomir_df):
    apa_diff, gene_logfc, mirna_logfc = {}, {}, {}
    for s1, s2_list in pairs.items():
        for s2 in s2_list:
            col = f"{s1}_{s2}"
            apa_diff[col] = apa_df[s1] - apa_df[s2]
            gene_logfc[col] = np.log2((gene_df[s1] + 1) / (gene_df[s2] + 1))
            mirna_logfc[col] = np.log2((isomir_df[s1] + 1) / (isomir_df[s2] + 1))
    return pd.DataFrame(apa_diff), pd.DataFrame(gene_logfc), pd.DataFrame(mirna_logfc)


def build_X(apa_block, gene_block, apa_index, gene_index):
    apa_block = apa_block.reindex(apa_index, fill_value=0)
    gene_block = gene_block.reindex(gene_index, fill_value=0)
    return pd.concat([apa_block, gene_block], axis=0).T


def main():
    if len(sys.argv) < 3:
        print("Usage: python train_ols_mirna.py <batch:int> <top_x_genes:int>")
        sys.exit(1)

    n = 1
    batch = int(sys.argv[1])
    top_x_genes = int(sys.argv[2])

    data_dir = Path("data/input")
    out_dir = Path("outputs")
    (out_dir / "final").mkdir(parents=True, exist_ok=True)

    apa_path     = data_dir / "apa.csv" #Stores samples in columns, Genes in rows and PDUI values 
    gene_path    = data_dir / "gene.csv" #Stores samples in columns, Genes in rows and TPM-values 
    isomir_path  = data_dir / "isomir.csv"#Stores samples in columns, MicroRNAs in rows and RPM-values
    feature_path = data_dir / "microt_filtered_1000_features.csv" #Stores MicroRNAs in columns and top 1000 genes with highest microT values in values 
    mirna_file   = data_dir / f"mirnas_batch_{batch}.csv" #Stores MicroRNAs in columns- can be feature_path file split in multiple batches for parallel processing


    apa_df = pd.read_csv(apa_path, index_col=0)
    gene_df = pd.read_csv(gene_path, index_col=0)
    isomir_df = pd.read_csv(isomir_path, index_col=0)
    feature_df = pd.read_csv(feature_path, index_col=0)

    samples = isomir_df.columns.tolist()
    mirnas = pd.read_csv(mirna_file, index_col=0).columns.tolist()

    train_samples, test_samples = train_test_split(samples, test_size=0.2, random_state=42)

    all_results = []

    print("Start training...")
    for idx, miRNA in enumerate(mirnas, start=1):
        print(f"{idx}: Processing {miRNA}")

        test_pairs = generate_pairs(test_samples, n, miRNA, isomir_df)
        folds = [list(arr) for arr in np.array_split(np.random.permutation(train_samples), 5)]
        train_pairs = {i: generate_pairs(folds[i], n, miRNA, isomir_df) for i in range(5)}

        test_apa_g, test_gene_g, test_mirna_g = compute_pairwise_differences(test_pairs, apa_df, gene_df, isomir_df)
        train_apa_g, train_gene_g, train_mirna_g = {}, {}, {}
        for i in range(5):
            train_apa_g[i], train_gene_g[i], train_mirna_g[i] = compute_pairwise_differences(
                train_pairs[i], apa_df, gene_df, isomir_df
            )

        if miRNA not in feature_df.columns:
            print(f"  - Skipping {miRNA}: not in feature file.")
            continue
        genes = feature_df[miRNA].dropna().tolist()[:top_x_genes]
        if not genes:
            print(f"  - Skipping {miRNA}: no features.")
            continue

        apa_index = [f"{g}_apa" for g in genes]
        gene_index = [f"{g}_gene" for g in genes]

        t_apa = test_apa_g.loc[test_apa_g.index.intersection(apa_index)]
        t_gene = test_gene_g.loc[test_gene_g.index.intersection(gene_index)]
        X_test = build_X(t_apa, t_gene, apa_index, gene_index)
        y_test = test_mirna_g.loc[miRNA].T

        if X_test.shape[1] == 0:
            print(f"  - Skipping {miRNA}: empty X_test.")
            continue

        X_train_folds, y_train_folds = [], []
        for i in range(5):
            tr_apa = train_apa_g[i].loc[train_apa_g[i].index.intersection(apa_index)]
            tr_gene = train_gene_g[i].loc[train_gene_g[i].index.intersection(gene_index)]
            X_train_folds.append(build_X(tr_apa, tr_gene, apa_index, gene_index))
            y_train_folds.append(train_mirna_g[i].loc[miRNA])

        cols0 = X_train_folds[0].columns
        if any(not X.columns.equals(cols0) for X in X_train_folds) or not X_test.columns.equals(cols0):
            raise RuntimeError(f"Feature misalignment for {miRNA}")

        # 5-fold CV metrics (no hyperparams to tune for OLS)
        cv_r2_scores, cv_mse_scores = [], []
        for i in range(5):
            X_val = X_train_folds[i]
            y_val = y_train_folds[i].T
            X_tr = pd.concat([X_train_folds[j] for j in range(5) if j != i], axis=0)
            y_tr = pd.concat([y_train_folds[j] for j in range(5) if j != i], axis=0)

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)

            lr = LinearRegression()
            lr.fit(X_tr, y_tr)
            y_hat = lr.predict(X_val_s)
            cv_r2_scores.append(r2_score(y_val, y_hat))
            cv_mse_scores.append(mean_squared_error(y_val, y_hat))

        cv_r2 = float(np.mean(cv_r2_scores))
        cv_mse = float(np.mean(cv_mse_scores))

        # Final train + test
        X_train_all = pd.concat(X_train_folds, axis=0)
        y_train_all = pd.concat(y_train_folds, axis=0)
        scaler = StandardScaler()
        X_train_all = scaler.fit_transform(X_train_all)
        X_test_s = scaler.transform(X_test)

        final = LinearRegression()
        final.fit(X_train_all, y_train_all)

        y_pred = final.predict(X_test_s)
        all_results.append({
            "miRNA Name": miRNA,
            "Test R² Score": r2_score(y_test, y_pred),
            "Test MSE": mean_squared_error(y_test, y_pred),
            "Test Mean Absolute Error": mean_absolute_error(y_test, y_pred),
            "Test Median Absolute Error": median_absolute_error(y_test, y_pred),
            "Mean Y (Test)": float(np.mean(y_test)),
            "Mean Y Pred (Test)": float(np.mean(y_pred)),
            "Std Y Test": float(np.std(y_test)),
            "Std Y Pred Test": float(np.std(y_pred)),
            "CV R² (5-fold)": cv_r2,
            "CV MSE (5-fold)": cv_mse,
        })

    out_csv = out_dir / "final" / f"ols_n_{n}_batch_{batch}_top{top_x_genes}.csv"
    pd.DataFrame(all_results).to_csv(out_csv, index=False)
    print(f"Done. Wrote results to: {out_csv}")


if __name__ == "__main__":
    main()
