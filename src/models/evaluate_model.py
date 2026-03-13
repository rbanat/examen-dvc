import json
import os
import pickle
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROCESSED_DIR = os.path.join("data", "processed_data")
MODELS_DIR = "models"
METRICS_DIR = "metrics"

# Chargement des jeux de données
X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test_scaled.csv"))
y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).squeeze()

# Chargement du modèle
with open(os.path.join(MODELS_DIR, "model.pkl"), "rb") as f:
    model = pickle.load(f)

# Prédiction
y_pred = model.predict(X_test)

# Métriques
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

scores = {
    "mse": round(mse, 6),
    "rmse": round(rmse, 6),
    "mae": round(mae, 6),
    "r2": round(r2, 6),
}
print("Métriques d'évaluation :")
for k, v in scores.items():
    print(f"  {k.upper():<5} : {v}")

# Sauvegarde des métriques
os.makedirs(METRICS_DIR, exist_ok=True)
scores_path = os.path.join(METRICS_DIR, "scores.json")
with open(scores_path, "w") as f:
    json.dump(scores, f, indent=4)
print(f"\nscores.json sauvegardé dans '{METRICS_DIR}'")

# Sauvegarde des prédicitons
predictions_df = pd.DataFrame({
    "y_true": y_test.values,
    "y_pred": y_pred,
})
predictions_path = os.path.join(PROCESSED_DIR, "predictions.csv")
predictions_df.to_csv(predictions_path, index=False)
print(f"predictions.csv sauvegardé dans '{PROCESSED_DIR}'")