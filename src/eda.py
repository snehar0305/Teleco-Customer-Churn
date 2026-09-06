import os
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH="data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_DIR="outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

#load dataset

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("TELCO CUSTOMER CHURN - PART 1")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nColumn :")
print(df.columns.tolist())

print("\First 5 rows:")
print(df.head())

#Overall Churn Rate
churn_rate = (df['Churn'].value_counts(normalize=True).get('Yes', 0) * 100)
print(f"\nOverall Churn Rate: {churn_rate:.2f}%")

#plot contract churn


contract_churn = (df.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).sort_values(ascending=False))
print("\nChurn Rate by Contract")
print(contract_churn)
plt.figure(figsize=(8, 5))


ax=contract_churn.plot(kind="bar")

ax.set_title("Churn Rate by Contract Type")
ax.set_xlabel("Contract Type")
ax.set_ylabel("Churn Rate (%)")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/churn_by_contract.png", dpi=300)

plt.close()

print("\n Contract churn values:")
print(contract_churn)
print("\n Data type:")
print(contract_churn.dtypes)



#Churn by internet service

internet_churn = (df.groupby('InternetService')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).sort_values(ascending=False))

print ("\nChurn Rate by Internet Service:")
print(internet_churn)

#plot internet service churn

plt.figure(figsize=(8, 5))
internet_churn.plot(kind="bar")
plt.title("Churn Rate by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/churn_by_internet_service.png", dpi=300)
plt.close()


#tenure vs churn

df["ChurnFlag"]=df["Churn"].map({"Yes": 1, "No": 0})
tenure_churn_correlation = df["tenure"].corr(df["ChurnFlag"])
print(f"\nCorrelation between Tenure and Churn: {tenure_churn_correlation:.4f}")

#tenure visualization

plt.figure(figsize=(8, 5))

df.boxplot(column="tenure", by="Churn")
plt.title("Tenure Distribution by Churn")
plt.xlabel("Churn")
plt.ylabel("Tenure")
plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/tenure_by_churn.png", dpi=300)
plt.close()

print("Part 1 completed sucessfully")
print("Charts saved in output folder")

