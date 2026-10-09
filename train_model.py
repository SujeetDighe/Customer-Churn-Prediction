import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ------------------------------------------------
# Load dataset
# ------------------------------------------------
df = pd.read_csv("churn_data.csv")

# ------------------------------------------------
# Data cleaning
# ------------------------------------------------
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# ------------------------------------------------
# Separate input and target
# ------------------------------------------------
X = df.drop(columns=["Churn", "customerID"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# ------------------------------------------------
# One-hot encoding
# ------------------------------------------------
X = pd.get_dummies(
    X,
    drop_first=True,
    dtype=int
)

feature_names = X.columns.tolist()

# ------------------------------------------------
# Train-test split
# ------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ------------------------------------------------
# Feature scaling
# ------------------------------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------
# Train Logistic Regression model
# ------------------------------------------------
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# ------------------------------------------------
# Evaluation
# ------------------------------------------------
predictions = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, predictions)

print("Model Training Completed")
print("------------------------")
print(f"Test Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# ------------------------------------------------
# Save model files
# ------------------------------------------------
joblib.dump(model, "churn_model.pkl")
joblib.dump(feature_names, "feature_names.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nSaved files:")
print("1. churn_model.pkl")
print("2. feature_names.pkl")
print("3. scaler.pkl")
