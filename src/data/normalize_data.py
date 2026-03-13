import os
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROCESSED_DIR = os.path.join("data", "processed_data")
SCALER_DIR = os.path.join("models", "scaler")

X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))

# Suppression des colonnes non numériques
X_train = X_train.select_dtypes(include="number")
X_test = X_test.select_dtypes(include="number")

# Normalisation
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train), columns=X_train.columns
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test), columns=X_test.columns
)

# Sauvegarde
X_train_scaled.to_csv(os.path.join(PROCESSED_DIR, "X_train_scaled.csv"), index=False)
X_test_scaled.to_csv(os.path.join(PROCESSED_DIR, "X_test_scaled.csv"), index=False)

os.makedirs(SCALER_DIR, exist_ok=True)
with open(os.path.join(SCALER_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

print(
    "Normalisation terminée.\n"
    f"Fichiers sauvegardés dans '{PROCESSED_DIR}' et scaler dans '{SCALER_DIR}'"
)