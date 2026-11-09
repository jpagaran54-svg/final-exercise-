import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

# Download the UCI red-wine dataset
try:
    df = pd.read_csv(URL, sep=";")
except Exception:
    df = pd.read_csv("wine_quality.csv")

df.to_csv("wine_quality.csv", index=False)

print("Rows and columns:", df.shape)
print("Missing values:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())
print("Quality distribution:\n", df["quality"].value_counts().sort_index())

# Remove duplicates and create binary target
# 1 = GOOD QUALITY, 0 = NEEDS IMPROVEMENT
df = df.drop_duplicates().copy()
y = (df["quality"] >= 6).astype(int)
X = df.drop(columns=["quality"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced"
    ))
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\nMODEL EVALUATION RESULTS")
print("=" * 40)
print("Accuracy :", round(accuracy_score(y_test, predictions), 4))
print("Precision:", round(precision_score(y_test, predictions, zero_division=0), 4))
print("Recall   :", round(recall_score(y_test, predictions, zero_division=0), 4))
print("F1-score :", round(f1_score(y_test, predictions, zero_division=0), 4))
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=["Needs Improvement", "Good Quality"], zero_division=0))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

joblib.dump(model, "wine_model.pkl")
X.head(5).to_csv("sample_wine_input.csv", index=False)
print("\nSaved wine_model.pkl")
print("Saved sample_wine_input.csv")
