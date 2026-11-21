import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split


for filename in ("dset0.csv", "dset1.csv"):
    data = np.loadtxt(filename, delimiter=",")
    X, y = data[:, 1:], data[:, 0].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    print(f"\n{filename}")
    print("Trees  Train error (%)  Test error (%)  Train log loss  Test log loss")
    for trees in (1, 10, 100, 300, 1000):
        model = GradientBoostingClassifier(
            n_estimators=trees,
            loss="log_loss",
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
        ).fit(X_train, y_train)
        train_error = 100 * np.mean(model.predict(X_train) != y_train)
        test_error = 100 * np.mean(model.predict(X_test) != y_test)
        train_loss = log_loss(y_train, model.predict_proba(X_train))
        test_loss = log_loss(y_test, model.predict_proba(X_test))
        print(f"{trees:5d} {train_error:15.2f} {test_error:15.2f}"
              f" {train_loss:15.4f} {test_loss:14.4f}")

