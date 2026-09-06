import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

def prepare_data():

#load dataset4
    df = pd.read_csv(DATA_PATH)

  
    print("=" * 60)
    print("TELCO CUSTOMER CHURN - PREPROCESSING")
    print("=" * 60)

    print("\nOrginal Dataset Shape:")
    print(df.shape)

    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    print("\nMissing values in TotalCharges:")
    print(df['TotalCharges'].isna().sum())

    # Remove rows with missing TotalCharges
    rows_before = len(df)
    df = df.dropna(subset=['TotalCharges']).copy()
    rows_after = len(df)
    print(f"\nRows removed due to missing TotalCharges: {rows_before - rows_after}")
    print(f"\nDataset Shape after removing missing TotalCharges: {df.shape}")

    # Convert target variable
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Remove unnecessary customerID
    df = df.drop(columns=['customerID'])

    # Separate features and target
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    # Identify categorical and numerical columns
    numerical_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = [col for col in X.columns if col not in numerical_features]

    print("\nNumerical Features:")
    print(numerical_features)
    print("\nCategorical Features:")
    print(categorical_features)

    # Demographic features
    demographic_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']
    print("\nDemographic Features:")
    print(demographic_features)

#numerical preprocessing 

    numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
    ])

    # Categorical preprocessing
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Fit and transform the training data
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    print("\nTraining data shape:")
    print(X_train.shape)
    print("\nTest data shape:")
    print(X_test.shape)
    print("Preprocessed training data shape")
    print(X_train_preprocessed.shape)
    print("\nPreprocessed test data shape")
    print(X_test_preprocessed.shape)
    print("\nPreprocessing completed successfully.")

    return (X_train_preprocessed, X_test_preprocessed, y_train, y_test,
            preprocessor, numerical_features, categorical_features,
            demographic_features, df)



#run

if __name__ == "__main__":
    prepare_data()