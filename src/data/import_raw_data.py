import os
import pandas as pd

DATA_URL = "https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv"
RAW_PATH = os.path.join("data", "raw_data", "raw.csv")
PROCESSED_DIR = os.path.join("data", "processed_data")

print(f"Téléchargement des données depuis {DATA_URL} ...")
df = pd.read_csv(DATA_URL)
df.to_csv(RAW_PATH, index=False)
print(f"Données sauvegardées dans {RAW_PATH}")

print(f"Shape du dataset : {df.shape}")