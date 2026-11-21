import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def grid_search(model, grid, X_train, y_train, X_val, y_val):
    best_mse = float("inf")
    for params in ParameterGrid(grid):
        candidate = clone(model).set_params(**params).fit(X_train, y_train)
        mse = mean_squared_error(y_val, candidate.predict(X_val))
        if mse < best_mse:
            best_model, best_params, best_mse = candidate, params, mse
    return best_model, best_params, best_mse


data = pd.read_csv("winequality-white.csv", sep=None, engine="python")
X, y = data.iloc[:, :-1], data["quality"]
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev, test_size=0.125, random_state=42
)

forest_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 3, 5, 10, 20],
    "min_samples_leaf": [1, 2, 3, 5, 10],
    "max_features": [0.3, 0.5, 0.7, 1.0],
    "bootstrap": [True, False],
}
while len(ParameterGrid(forest_grid)) <= len(y_val):
    forest_grid["n_estimators"].append(forest_grid["n_estimators"][-1] + 100)

max_neighbors = min(len(y_train), max(150, len(y_val) // 4 + 1))
knn_grid = {
    "kneighborsregressor__n_neighbors": range(1, max_neighbors + 1),
    "kneighborsregressor__weights": ["uniform", "distance"],
    "kneighborsregressor__p": [1, 2],
}
models = [
    ("Random Forest", RandomForestRegressor(random_state=42, n_jobs=4), forest_grid),
    ("KNN", make_pipeline(StandardScaler(), KNeighborsRegressor()), knn_grid),
]

print(f"Train: {len(y_train)}, validation: {len(y_val)}, test: {len(y_test)}")
for name, model, grid in models:
    best_model, best_params, val_mse = grid_search(
        model, grid, X_train, y_train, X_val, y_val
    )
    test_mse = mean_squared_error(y_test, best_model.predict(X_test))
    print(f"\n{name}: {len(ParameterGrid(grid))} configurations")
    print(f"Best parameters: {best_params}")
    print(f"Validation MSE: {val_mse:.6f}, test MSE: {test_mse:.6f}")
