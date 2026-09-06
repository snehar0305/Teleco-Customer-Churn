from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from preprocessing import prepare_data


MODEL_PATH = Path("outputs/random_forest_model.joblib")
OUTPUT_DIR = Path("outputs")


def main():
    # Reuse preprocessing to obtain the fitted OneHotEncoder feature names.
    (
        _X_train,
        _X_test,
        _y_train,
        _y_test,
        preprocessor,
        _numerical_features,
        _categorical_features,
        _demographic_features,
        _df,
    ) = prepare_data()

    # Load the Random Forest trained by the modeling script.
    random_forest_model = joblib.load(MODEL_PATH)
    feature_names = preprocessor.get_feature_names_out()
    importances = random_forest_model.feature_importances_

    if len(feature_names) != len(importances):
        raise ValueError("Feature names and feature importances have different lengths.")

    # Match each feature name with its importance and sort from highest to lowest.
    importance_table = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values("Importance", ascending=False, ignore_index=True)

    top_10_features = importance_table.head(10)
    top_3_features = importance_table.head(3)

    print("\nTOP 10 RANDOM FOREST FEATURES")
    print(top_10_features.to_string(index=False, float_format="{:.6f}".format))

    print("\nTOP 3 MOST IMPORTANT FEATURES")
    for rank, feature_name in enumerate(top_3_features["Feature"], start=1):
        print(f"{rank}. {feature_name}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    importance_table.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    # Create a horizontal bar chart so the feature names are easy to read.
    chart_data = top_10_features.sort_values("Importance")
    plt.figure(figsize=(10, 6))
    plt.barh(chart_data["Feature"], chart_data["Importance"])
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title("Top 10 Random Forest Feature Importances")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=300)
    plt.close()

    print(f"\nSaved all feature importances to {OUTPUT_DIR / 'feature_importance.csv'}")
    print(f"Saved top 10 feature chart to {OUTPUT_DIR / 'feature_importance.png'}")


if __name__ == "__main__":
    main()