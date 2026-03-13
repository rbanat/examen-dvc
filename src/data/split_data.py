import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_URL = "https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv"
DATA_RAW_PATH = os.path.join("data", "raw_data", "raw.csv")
PROCESSED_DIR = os.path.join("data", "processed_data")

print(f"Chargement des données depuis {DATA_RAW_PATH}")
df = pd.read_csv(DATA_RAW_PATH)

# Split
X = df.drop(columns=["silica_concentrate"])
y = df[["silica_concentrate"]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Sauvegarde
X_train.to_csv(os.path.join(PROCESSED_DIR, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(PROCESSED_DIR, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(PROCESSED_DIR, "y_test.csv"), index=False)

print(
    f"Split terminé — Train : {len(X_train)} lignes | Test : {len(X_test)} lignes\n"
    f"Fichiers sauvegardés dans '{PROCESSED_DIR}'"
)