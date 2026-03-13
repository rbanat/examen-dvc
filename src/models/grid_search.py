import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

PROCESSED_DIR = os.path.join("data", "processed_data")
PARAMS_DIR = os.path.join("models", "params")

PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}

# Chargement des jeux de données
X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train_scaled.csv"))
y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).squeeze()

# Grid Search
estimator = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(
    estimator=estimator,
    param_grid=PARAM_GRID,
    scoring="neg_mean_squared_error",
    cv=5,
    n_jobs=-1,
    verbose=2,
)

print("Lancement du GridSearchCV...")
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(f"Meilleurs paramètres trouvés : {best_params}")

# Sauvegarde
os.makedirs(PARAMS_DIR, exist_ok=True)
with open(os.path.join(PARAMS_DIR, "best_params.pkl"), "wb") as f:
    pickle.dump(best_params, f)

print(f"best_params.pkl sauvegardé dans '{PARAMS_DIR}'")