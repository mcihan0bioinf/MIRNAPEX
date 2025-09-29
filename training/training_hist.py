# Reproducible HGBR training with high/low stratified pairing in both directions.

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    mean_squared_error,
)

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
    lows  = mirna_values[mirna_values <  median_value].index.tolist()

    pairs = {}
    for s in highs:
        k = min(num_pairs, len(lows))
        pairs[s] = np.random.choice(lows, k, replace=False).tolist() if k > 0 else []
    for s in lows:
        k = min(num_pairs, len(highs))
        pairs[s] = np.random.choice(highs, k, replace=False).tolist() if k > 0 else []
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
            col = f"{s1}_{s2}"
            apa_diff[col]   = apa_df[s1] - apa_df[s2]
            gene_logfc[col] = np.log2((gene_df[s1] + 1) / (gene_df[s2] + 1))
            mirna_logfc[col]= np.log2((isomir_df[s1] + 1) / (isomir_df[s2] + 1))
    return pd.DataFrame(apa_diff), pd.DataFrame(gene_logfc), pd.DataFrame(mirna_logfc)


def build_X(apa_block, gene_block, apa_index, gene_index):
    """
    Ensure consistent feature columns across folds/test by reindexing to the
    full expected lists and filling missing entries with 0.
    """
    apa_block  = apa_block.reindex(apa_index,  fill_value=0)
    gene_block = gene_block.reindex(gene_index, fill_value=0)
    return pd.concat([apa_block, gene_block], axis=0).T  # pairs x features


def main():
    # -------------------------
    # Args
    # -------------------------
    if len(sys.argv) < 3:
        print("Usage: python train_histgb_mirna.py <batch:int> <top_x_genes:int>")
        sys.exit(1)

    n = 1  # partner pairs per sample
    batch = int(sys.argv[1])
    top_x_genes = int(sys.argv[2])

    # -------------------------
    # Relative paths
    # -------------------------
    data_dir = Path("data/input")
    out_dir  = Path("outputs")
    (out_dir / "final").mkdir(parents=True, exist_ok=True)

    apa_path     = data_dir / "apa.csv" #Stores samples in columns, Genes in rows and PDUI values 
    gene_path    = data_dir / "gene.csv" #Stores samples in columns, Genes in rows and TPM-values 
    isomir_path  = data_dir / "isomir.csv"#Stores samples in columns, MicroRNAs in rows and RPM-values
    feature_path = data_dir / "microt_filtered_1000_features.csv" #Stores MicroRNAs in columns and top 1000 genes with highest microT values in values 
    mirna_file   = data_dir / f"mirnas_batch_{batch}.csv" #Stores MicroRNAs in columns- can be feature_path file split in multiple batches for parallel processing

    # -------------------------
    # Load datasets
    # -------------------------
    apa_df     = pd.read_csv(apa_path, index_col=0)
    gene_df    = pd.read_csv(gene_path, index_col=0)
    isomir_df  = pd.read_csv(isomir_path, index_col=0)
    feature_df = pd.read_csv(feature_path, index_col=0)

    samples = isomir_df.columns.tolist()
    mirnas  = pd.read_csv(mirna_file, index_col=0).columns.tolist()

    # -------------------------
    # Train-Test Split
    # -------------------------
    train_samples, test_samples = train_test_split(samples, test_size=0.2, random_state=42)

    # --- Small, high-impact hyperparameter grids ---
    learning_rates      = [0.01 0.03, 0.1]
    max_leaf_nodes_grid = [15, 31, 63]
    min_samples_leaf_grid = [10, 20, 30, 127]

    all_results = []
    print("Start training...")
    for i_mi, miRNA in enumerate(mirnas, start=1):
        print(f"{i_mi}: Processing {miRNA}")

        # -------------------------
        # Pairing (test and 5 train folds) — both directions
        # -------------------------
        test_pairs = generate_pairs(test_samples, n, miRNA, isomir_df)
        folds = [list(arr) for arr in np.array_split(np.random.permutation(train_samples), 5)]
        train_pairs = {i: generate_pairs(folds[i], n, miRNA, isomir_df) for i in range(5)}

        # -------------------------
        # Pairwise differences
        # -------------------------
        test_apa_g, test_gene_g, test_mirna_g = compute_pairwise_differences(test_pairs, apa_df, gene_df, isomir_df)
        train_apa_g, train_gene_g, train_mirna_g = {}, {}, {}
        for i in range(5):
            train_apa_g[i], train_gene_g[i], train_mirna_g[i] = compute_pairwise_differences(
                train_pairs[i], apa_df, gene_df, isomir_df
            )

        # -------------------------
        # Feature selection (top X per miRNA)
        # -------------------------
        if miRNA not in feature_df.columns:
            print(f"  - Skipping {miRNA}: not in feature file.")
            continue

        genes_for_miRNA = feature_df[miRNA].dropna().tolist()
        top_genes = genes_for_miRNA[:top_x_genes]
        if not top_genes:
            print(f"  - Skipping {miRNA}: no features.")
            continue

        apa_index  = [f"{g}_apa"  for g in top_genes]
        gene_index = [f"{g}_gene" for g in top_genes]

        # -------------------------
        # Build aligned matrices
        # -------------------------
        t_apa  = test_apa_g.loc[test_apa_g.index.intersection(apa_index)]
        t_gene = test_gene_g.loc[test_gene_g.index.intersection(gene_index)]
        X_test = build_X(t_apa, t_gene, apa_index, gene_index)
        y_test = test_mirna_g.loc[miRNA].T

        if X_test.shape[1] == 0:
            print(f"  - Skipping {miRNA}: empty X_test after selection.")
            continue

        X_train_folds, y_train_folds = [], []
        for i in range(5):
            tr_apa  = train_apa_g[i].loc[train_apa_g[i].index.intersection(apa_index)]
            tr_gene = train_gene_g[i].loc[train_gene_g[i].index.intersection(gene_index)]
            X_train_folds.append(build_X(tr_apa, tr_gene, apa_index, gene_index))
            y_train_folds.append(train_mirna_g[i].loc[miRNA])

        # Sanity: identical columns everywhere
        cols0 = X_train_folds[0].columns
        if any(not X.columns.equals(cols0) for X in X_train_folds) or not X_test.columns.equals(cols0):
            raise RuntimeError(f"Feature misalignment for {miRNA}")

        # -------------------------
        # 5-fold CV over (learning_rate, max_leaf_nodes, min_samples_leaf)
        # with internal early stopping on each fold's training set
        # -------------------------
        best_params = {"learning_rate": None, "max_leaf_nodes": None, "min_samples_leaf": None}
        best_r2, best_mse = -np.inf, np.inf

        for lr in learning_rates:
            for mln in max_leaf_nodes_grid:
                for msl in min_samples_leaf_grid:
                    r2s, mses = [], []
                    for i in range(5):
                        X_val = X_train_folds[i]
                        y_val = y_train_folds[i].T
                        X_tr  = pd.concat([X_train_folds[j] for j in range(5) if j != i], axis=0)
                        y_tr  = pd.concat([y_train_folds[j] for j in range(5) if j != i], axis=0)

                        model = HistGradientBoostingRegressor(
                            learning_rate=lr,
                            max_leaf_nodes=mln,
                            min_samples_leaf=msl,
                            max_iter=500,
                            early_stopping=True,
                            validation_fraction=0.1,   # split from X_tr only (no leakage)
                            n_iter_no_change=30,
                            random_state=42,
                        )
                        model.fit(X_tr, y_tr)

                        y_hat = model.predict(X_val)
                        r2s.append(r2_score(y_val, y_hat))
                        mses.append(mean_squared_error(y_val, y_hat))

                    avg_r2, avg_mse = float(np.mean(r2s)), float(np.mean(mses))
                    if avg_r2 > best_r2:
                        best_r2, best_mse = avg_r2, avg_mse
                        best_params = {"learning_rate": lr, "max_leaf_nodes": mln, "min_samples_leaf": msl}

        # -------------------------
        # Final train on all folds & evaluate on test
        # -------------------------
        X_train_all = pd.concat(X_train_folds, axis=0)
        y_train_all = pd.concat(y_train_folds, axis=0)

        final_model = HistGradientBoostingRegressor(
            learning_rate=best_params["learning_rate"],
            max_leaf_nodes=best_params["max_leaf_nodes"],
            min_samples_leaf=best_params["min_samples_leaf"],
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
            random_state=42,
        )
        final_model.fit(X_train_all, y_train_all)

        y_pred_test = final_model.predict(X_test)

        # Metrics
        all_results.append({
            "miRNA Name": miRNA,
            "Best learning_rate": best_params["learning_rate"],
            "Best max_leaf_nodes": best_params["max_leaf_nodes"],
            "Best min_samples_leaf": best_params["min_samples_leaf"],
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
        })

    # -------------------------
    # Save summary CSV
    # -------------------------
    out_csv = out_dir / "final" / f"histboost_n_{n}_batch_{batch}_top{top_x_genes}.csv"
    pd.DataFrame(all_results).to_csv(out_csv, index=False)
    print(f"Done. Wrote results to: {out_csv}")


if __name__ == "__main__":
    main()
