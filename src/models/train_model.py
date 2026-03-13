import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

PROCESSED_DIR = os.path.join("data", "processed_data")
MODELS_DIR = os.path.join("models", "models")

# Chargement des jeux de données
X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train_scaled.csv"))
y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).squeeze()

# Chargement des meilleurs hyperparamètres
params_path = os.path.join(MODELS_DIR, "best_params.pkl")
with open(params_path, "rb") as f:
    best_params = pickle.load(f)

print(f"Paramètres utilisés pour l'entraînement : {best_params}")

# Entrainement
model = RandomForestRegressor(**best_params, random_state=42)
model.fit(X_train, y_train)

print("Entraînement terminé.")

# Sauvegarde
model_path = os.path.join(MODELS_DIR, "model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"Modèle sauvegardé dans '{model_path}'")