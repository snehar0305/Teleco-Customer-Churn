from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from preprocessing import prepare_data


RANDOM_STATE = 42
OUTPUT_DIR = Path("outputs")


def main():
    # prepare_data returns the preprocessed train/test data and target values.
    (
        X_train,
        X_test,
        y_train,
        y_test,
        _preprocessor,
        _numerical_features,
        _categorical_features,
        _demographic_features,
        _df,
    ) = prepare_data()

    # Train the simple baseline model.
    logistic_model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    logistic_model.fit(X_train, y_train)

    # Train the more complex tree-based model.
    random_forest_model = RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    random_forest_model.fit(X_train, y_train)

    models = {
        "Logistic Regression": logistic_model,
        "Random Forest": random_forest_model,
    }
    results = []

    for model_name, model in models.items():
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        test_accuracy = accuracy_score(y_test, predictions)
        training_accuracy = accuracy_score(y_train, model.predict(X_train))

        results.append(
            {
                "Model": model_name,
                "Accuracy": test_accuracy,
                "Precision": precision_score(y_test, predictions, zero_division=0),
                "Recall": recall_score(y_test, predictions, zero_division=0),
                "F1-score": f1_score(y_test, predictions, zero_division=0),
                "ROC-AUC": roc_auc_score(y_test, probabilities),
                "Training Accuracy": training_accuracy,
                "Train-Test Accuracy Gap": training_accuracy - test_accuracy,
            }
        )

    comparison_table = pd.DataFrame(results)

    print("\nMODEL COMPARISON")
    print(comparison_table.to_string(index=False, float_format="{:.4f}".format))

    # Save the metrics table and the trained Random Forest for later use.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison_table.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    joblib.dump(random_forest_model, OUTPUT_DIR / "random_forest_model.joblib")

    print(f"\nSaved model metrics to {OUTPUT_DIR / 'model_metrics.csv'}")
    print(f"Saved Random Forest model to {OUTPUT_DIR / 'random_forest_model.joblib'}")


if __name__ == "__main__":
    main()