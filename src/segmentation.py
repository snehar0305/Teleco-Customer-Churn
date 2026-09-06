from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from modeling import RANDOM_STATE
from preprocessing import prepare_data


MODEL_PATH = Path("outputs/random_forest_model.joblib")
OUTPUT_DIR = Path("outputs")


def create_segment_names(summary):
    """Create business-friendly names from the measured cluster characteristics."""
    risk_order = summary.sort_values("Average Churn Probability", ascending=False).index
    revenue_order = summary.sort_values("Estimated Monthly Revenue at Risk", ascending=False).index
    tenure_median = summary["Average Tenure"].median()
    charge_median = summary["Average MonthlyCharges"].median()

    names = {}
    highest_risk_cluster = risk_order[0]
    lowest_risk_cluster = risk_order[-1]
    highest_revenue_cluster = revenue_order[0]

    if summary.loc[highest_risk_cluster, "Average MonthlyCharges"] >= charge_median:
        names[highest_risk_cluster] = "High-Risk Valuable Customers"
    else:
        names[highest_risk_cluster] = "High-Risk Customers"

    if lowest_risk_cluster != highest_risk_cluster:
        if summary.loc[lowest_risk_cluster, "Average Tenure"] < tenure_median:
            names[lowest_risk_cluster] = "New/Low-Risk Customers"
        else:
            names[lowest_risk_cluster] = "Stable Customers"

    for cluster_id in summary.index:
        if cluster_id not in names:
            if cluster_id == highest_revenue_cluster:
                names[cluster_id] = "Highest Revenue-at-Risk Customers"
            elif summary.loc[cluster_id, "Average MonthlyCharges"] >= charge_median:
                names[cluster_id] = "Higher-Value Moderate-Risk Customers"
            else:
                names[cluster_id] = "Developing Customers"

    return names


def main():
    # prepare_data supplies the cleaned customer data and fitted preprocessing.
    (
        _X_train,
        _X_test,
        _y_train,
        _y_test,
        preprocessor,
        _numerical_features,
        _categorical_features,
        _demographic_features,
        customer_data,
    ) = prepare_data()

    # Load the Random Forest that was trained by modeling.py.
    random_forest_model = joblib.load(MODEL_PATH)
    customer_features = customer_data.drop(columns=["Churn"])
    customer_features_preprocessed = preprocessor.transform(customer_features)

    # Predict each customer's probability of belonging to the churn class.
    customer_data = customer_data.copy()
    customer_data["Predicted Churn Probability"] = random_forest_model.predict_proba(
        customer_features_preprocessed
    )[:, 1]

    # Use the three requested variables to build the customer segments.
    segmentation_features = customer_data[
        ["tenure", "MonthlyCharges", "Predicted Churn Probability"]
    ]
    scaled_features = StandardScaler().fit_transform(segmentation_features)
    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
    customer_data["Cluster"] = kmeans.fit_predict(scaled_features)

    # Summarize the size, customer characteristics, and revenue risk of each cluster.
    summary = (
        customer_data.groupby("Cluster")
        .agg(
            **{
                "Number of Customers": ("Cluster", "size"),
                "Average Tenure": ("tenure", "mean"),
                "Average MonthlyCharges": ("MonthlyCharges", "mean"),
                "Average Churn Probability": (
                    "Predicted Churn Probability",
                    "mean",
                ),
            }
        )
        .sort_index()
    )
    summary["Estimated Monthly Revenue at Risk"] = (
        summary["Number of Customers"]
        * summary["Average MonthlyCharges"]
        * summary["Average Churn Probability"]
    )
    segment_names = create_segment_names(summary)
    summary["Segment"] = summary.index.map(segment_names)
    summary = summary.reset_index()

    customer_data["Segment"] = customer_data["Cluster"].map(segment_names)
    summary = summary[
        [
            "Cluster",
            "Segment",
            "Number of Customers",
            "Average Tenure",
            "Average MonthlyCharges",
            "Average Churn Probability",
            "Estimated Monthly Revenue at Risk",
        ]
    ]

    print("\nCUSTOMER SEGMENT SUMMARY")
    print(summary.to_string(index=False, float_format="{:.4f}".format))

    highest_priority = summary.loc[
        summary["Estimated Monthly Revenue at Risk"].idxmax()
    ]
    print("\nHIGHEST RETENTION INVESTMENT PRIORITY")
    print(
        f"{highest_priority['Segment']} should receive the highest investment "
        f"(churn probability: {highest_priority['Average Churn Probability']:.4f}; "
        f"estimated monthly revenue at risk: "
        f"{highest_priority['Estimated Monthly Revenue at Risk']:.2f})."
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "customer_segments.csv", index=False)
    customer_data.to_csv(OUTPUT_DIR / "customer_segment_assignments.csv", index=False)

    # Plot estimated monthly revenue at risk so segments can be compared quickly.
    chart_data = summary.sort_values("Estimated Monthly Revenue at Risk")
    plt.figure(figsize=(10, 6))
    plt.barh(chart_data["Segment"], chart_data["Estimated Monthly Revenue at Risk"])
    plt.xlabel("Estimated Monthly Revenue at Risk")
    plt.ylabel("Customer Segment")
    plt.title("Estimated Monthly Revenue at Risk by Customer Segment")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "customer_segments.png", dpi=300)
    plt.close()

    print(f"\nSaved segment summary to {OUTPUT_DIR / 'customer_segments.csv'}")
    print(
        "Saved customer assignments to "
        f"{OUTPUT_DIR / 'customer_segment_assignments.csv'}"
    )
    print(f"Saved segment chart to {OUTPUT_DIR / 'customer_segments.png'}")


if __name__ == "__main__":
    main()