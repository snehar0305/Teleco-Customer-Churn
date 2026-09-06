CLAUSE_1 = (
    "Clause 1 — High Risk (probability ≥ 0.70): Offer a loyalty discount and a "
    "callback from a retention specialist within 48 hours."
)
CLAUSE_2 = (
    "Clause 2 — Moderate Risk (0.40–0.70): Send a targeted email highlighting an "
    "underused service or a contract upgrade offer."
)
CLAUSE_3 = (
    "Clause 3 — New Customer, Any Risk, Tenure < 3 months: Route to the onboarding "
    "team instead of the standard retention flow."
)
CLAUSE_4 = (
    "Clause 4 — Non-Discrimination Rule: Retention explanations must never state or "
    "imply that gender, senior-citizen status, or family/partner status contributed "
    "to a customer's risk score, even where a statistical correlation exists in the "
    "data."
)


def retrieve_retention_clause(churn_probability, tenure):
    """Return the appropriate retention clause for one customer."""
    # New customers always go to onboarding, regardless of their risk score.
    if tenure < 3:
        selected_clause = CLAUSE_3
    elif churn_probability >= 0.70:
        selected_clause = CLAUSE_1
    elif churn_probability >= 0.40:
        selected_clause = CLAUSE_2
    else:
        selected_clause = (
            "No retention clause applies because the customer is currently low risk."
        )

    # This rule must accompany every retrieval result.
    return f"Selected retention guidance:\n{selected_clause}\n\n{CLAUSE_4}"


if __name__ == "__main__":
    # Run a few small examples to demonstrate the retrieval precedence.
    test_customers = [
        ("High-risk customer", 0.85, 18),
        ("Moderate-risk customer", 0.55, 12),
        ("New high-risk customer", 0.90, 2),
    ]

    for customer_type, churn_probability, tenure in test_customers:
        print(f"\n{customer_type} (probability={churn_probability}, tenure={tenure})")
        print(retrieve_retention_clause(churn_probability, tenure))