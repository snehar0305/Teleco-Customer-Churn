import os
import re
import json
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import joblib
import pandas as pd

from modeling import OUTPUT_DIR
from preprocessing import prepare_data
from retrieval import retrieve_retention_clause


MODEL_PATH = OUTPUT_DIR / "random_forest_model.joblib"

LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

FORBIDDEN_FEATURE_WORDS = (
    "gender",
    "seniorcitizen",
    "partner",
    "dependents",
)


def get_high_risk_customer():
    """Return one real customer summary selected from model predictions."""

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

    # Load the Random Forest model created by modeling.py.
    random_forest_model = joblib.load(MODEL_PATH)

    customer_features = customer_data.drop(columns=["Churn"])

    transformed_features = preprocessor.transform(customer_features)

    probabilities = random_forest_model.predict_proba(
        transformed_features
    )[:, 1]

    # Get the feature names used by the trained model.
    feature_names = preprocessor.get_feature_names_out()

    importance_table = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": random_forest_model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    # Remove demographic features from the features passed to the LLM.
    allowed_features = importance_table[
        ~importance_table["Feature"]
        .str.lower()
        .str.contains(
            "|".join(FORBIDDEN_FEATURE_WORDS),
            regex=True,
        )
    ]

    top_features = allowed_features.head(3)["Feature"].tolist()

    # Find customers with churn probability >= 0.70.
    high_risk_indices = [
        index
        for index, probability in enumerate(probabilities)
        if probability >= 0.70
    ]

    if not high_risk_indices:
        raise RuntimeError(
            "No customer with churn probability >= 0.70 was found."
        )

    # Select the highest-risk customer.
    selected_index = max(
        high_risk_indices,
        key=lambda index: probabilities[index],
    )

    return {
        "churn_probability": float(probabilities[selected_index]),
        "tenure": float(
            customer_data.iloc[selected_index]["tenure"]
        ),
        "top_features": top_features,
    }


def call_llm(api_key, system_prompt, user_prompt):
    """Send a request to the OpenAI API."""

    request_body = json.dumps(
        {
            "model": LLM_MODEL,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        }
    )

    request = Request(
        "https://api.openai.com/v1/chat/completions",
        data=request_body.encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=60) as response:
            response_data = json.loads(
                response.read().decode("utf-8")
            )

        return response_data["choices"][0]["message"]["content"].strip()

    except HTTPError as error:

        error_body = error.read().decode("utf-8")

        print("\nAPI ERROR:")
        print(error_body)

        # Do not retry if the account has no API credits.
        if "credit_balance_exhausted" in error_body:
            raise RuntimeError(
                "OpenAI API credits are exhausted. "
                "The ML and retrieval parts completed successfully, "
                "but the live LLM call cannot be completed until "
                "API credits are available."
            )

        # Handle other API errors.
        raise RuntimeError(
            f"OpenAI API request failed with HTTP {error.code}."
        )


def main():

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set."
        )

    print("=" * 60)
    print("TELCO CUSTOMER CHURN - LLM ADVISOR")
    print("=" * 60)

    # ---------------------------------------------------------
    # STEP 1: SELECT HIGH-RISK CUSTOMER
    # ---------------------------------------------------------

    customer = get_high_risk_customer()

    churn_probability = customer["churn_probability"]
    tenure = customer["tenure"]
    top_features = customer["top_features"]

    # ---------------------------------------------------------
    # STEP 2: RETRIEVE RETENTION PLAYBOOK CLAUSE
    # ---------------------------------------------------------

    retrieved_clause = retrieve_retention_clause(
        churn_probability,
        tenure,
    )

    clause_match = re.search(
        r"Clause\s+([1-3])",
        retrieved_clause,
        re.IGNORECASE,
    )

    correct_clause_number = (
        clause_match.group(1)
        if clause_match
        else "none"
    )

    # ---------------------------------------------------------
    # STEP 3: SYSTEM PROMPT
    # ---------------------------------------------------------

    system_prompt = (
        "You are a retention advisor writing for a retention agent. "
        "Write a short 3–4 sentence explanation. "
        "Use only the supplied customer risk probability, tenure, "
        "top 3 feature names, and retrieved Retention Playbook clause. "
        "Follow the retrieved clause and clearly state the recommended "
        "retention action. "
        "Never mention or imply that gender, SeniorCitizen, Partner, "
        "or Dependents contributed to the risk score. "
        "Never invent customer information."
    )

    user_prompt = (
        f"Customer churn probability: {churn_probability:.4f}\n"
        f"Customer tenure in months: {tenure:.0f}\n"
        f"Top 3 contributing feature names: "
        f"{', '.join(top_features)}\n"
        f"Retrieved Retention Playbook clause:\n"
        f"{retrieved_clause}"
    )

    # ---------------------------------------------------------
    # STEP 4: FIRST LLM CALL WITH RETRIEVAL
    # ---------------------------------------------------------

    print("\nCalling LLM with retrieval...")

    try:
        llm_response_with_retrieval = call_llm(
            api_key,
            system_prompt,
            user_prompt,
        )

    except RuntimeError as error:

        print("\n" + "=" * 60)
        print("LLM API COULD NOT BE COMPLETED")
        print("=" * 60)

        print(f"\nReason: {error}")

        print("\nML and retrieval results:")
        print(
            f"Customer risk probability: "
            f"{churn_probability:.4f}"
        )
        print(f"Tenure: {tenure:.0f} months")
        print(
            f"Top 3 contributing features: "
            f"{', '.join(top_features)}"
        )
        print(
            f"Retrieved clause number: "
            f"{correct_clause_number}"
        )

        print(
            "\nThe LLM response was not generated because "
            "the API account has no remaining credits."
        )

        return

    # ---------------------------------------------------------
    # STEP 5: SECOND LLM CALL WITHOUT RETRIEVAL
    # ---------------------------------------------------------

    no_retrieval_system_prompt = (
        "You are testing a rule-based retention policy classifier. "
        "Based only on the customer churn probability and tenure, "
        "identify which Retention Playbook clause number should apply. "
        "Reply with the clause number and a short reason. "
        "No playbook clause text is provided, so do not invent or "
        "quote policy wording."
    )

    no_retrieval_user_prompt = (
        f"Customer churn probability: {churn_probability:.4f}\n"
        f"Customer tenure in months: {tenure:.0f}\n"
        "Which Retention Playbook clause number should apply?"
    )

    print("\nCalling LLM without retrieval...")

    llm_response_without_retrieval = call_llm(
        api_key,
        no_retrieval_system_prompt,
        no_retrieval_user_prompt,
    )

    # ---------------------------------------------------------
    # STEP 6: CHECK SECOND LLM ANSWER
    # ---------------------------------------------------------

    no_retrieval_clause_match = re.search(
        r"Clause\s+([1-3])",
        llm_response_without_retrieval,
        re.IGNORECASE,
    )

    no_retrieval_clause_number = (
        no_retrieval_clause_match.group(1)
        if no_retrieval_clause_match
        else "none"
    )

    selected_correctly = (
        no_retrieval_clause_number == correct_clause_number
    )

    # ---------------------------------------------------------
    # STEP 7: DISPLAY RESULTS
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("LLM ADVISOR RESULTS")
    print("=" * 60)

    print(
        f"Customer risk probability: "
        f"{churn_probability:.4f}"
    )

    print(f"Tenure: {tenure:.0f} months")

    print(
        f"Top 3 contributing features: "
        f"{', '.join(top_features)}"
    )

    print(
        f"Retrieved clause number: "
        f"{correct_clause_number}"
    )

    print(
        "\nLLM response WITH retrieval:"
    )
    print(llm_response_with_retrieval)

    print(
        "\nLLM response WITHOUT retrieval:"
    )
    print(llm_response_without_retrieval)

    print(
        "\nNo-retrieval clause selected: "
        f"{no_retrieval_clause_number}"
    )

    print(
        "No-retrieval answer selected the correct clause: "
        f"{selected_correctly}"
    )


if __name__ == "__main__":
    main()