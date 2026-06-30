import pandas as pd
import joblib
import numpy as np
import sys
import sklearn
# Inputs
model_file = sys.argv[1]
apa_file = sys.argv[2]
gene_file = sys.argv[3]
r2_file = sys.argv[4]
output_file = sys.argv[5]

# Load models
models = joblib.load(model_file)

# Load feature data
apa_df = pd.read_csv(apa_file, index_col=0)
gene_df = pd.read_csv(gene_file)
gene_df.set_index("Feature", inplace=True)

# Merge APA and gene features into one feature vector
merged_df = pd.concat([apa_df, gene_df], axis=0)
feature_series = merged_df["value"] if "value" in merged_df.columns else merged_df["logFC"]
feature_vector = feature_series.to_dict()

# Load R² scores and filter at threshold ≥ 0.3
r2_df = pd.read_csv(r2_file)
r2_df = r2_df[r2_df["R_squared"] >= 0.3].copy()
r2_dict = r2_df.set_index("microRNA")["R_squared"].to_dict()

# Confidence tag
def tag_confidence(r2):
    return "High" if r2 >= 0.5 else "Moderate"

results = []

# Predict
for mirna, model_data in models.items():
    if mirna not in r2_dict:
        continue
    r2 = r2_dict[mirna]
    confidence = tag_confidence(r2)

    features = model_data["feature_names"]
    model = model_data["model"]
    scaler = model_data["scaler"]

    values = [feature_vector.get(f, 0.0) for f in features]
    X_df = pd.DataFrame([values], columns=features)
    X_scaled_array = scaler.transform(X_df)
    X_scaled_df = pd.DataFrame(X_scaled_array, columns=features, index=X_df.index)
    logfc = model.predict(X_scaled_df)[0]

    results.append({
        "microRNA": mirna,
        "logFC": logfc,
        "Confidence": confidence
    })

# Output: sort by absolute logFC
df = pd.DataFrame(results)
df["abs_logFC"] = df["logFC"].abs()
df = df.sort_values(by="abs_logFC", ascending=False).drop(columns=["abs_logFC"])
df.to_csv(output_file, index=False)

