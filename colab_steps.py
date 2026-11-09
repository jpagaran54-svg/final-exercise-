# GOOGLE COLAB STEP-BY-STEP CODE
# Run each section in a separate Colab cell.

# STEP 1-2: Get the CSV
import pandas as pd
from google.colab import files
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=";")
df.to_csv("wine_quality.csv", index=False)
print("CSV created successfully!")
print("Dataset shape:", df.shape)
display(df.head())
files.download("wine_quality.csv")

# STEP 3: Inspect
print(df.columns.tolist())
print(df.dtypes)
print(df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())
print(df["quality"].value_counts().sort_index())

# STEP 4-6: Train and evaluate
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

df = df.drop_duplicates().copy()
y = (df["quality"] >= 6).astype(int)
X = df.drop(columns=["quality"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced"))
])
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Accuracy :", round(accuracy_score(y_test, predictions), 4))
print("Precision:", round(precision_score(y_test, predictions, zero_division=0), 4))
print("Recall   :", round(recall_score(y_test, predictions, zero_division=0), 4))
print("F1-score :", round(f1_score(y_test, predictions, zero_division=0), 4))
print(classification_report(y_test, predictions, target_names=["Needs Improvement", "Good Quality"], zero_division=0))
print(confusion_matrix(y_test, predictions))

# STEP 7-8: Save/download model
import joblib
joblib.dump(model, "wine_model.pkl")
files.download("wine_model.pkl")

# STEP 9: Create/download sample input CSV
sample_data = df.drop(columns=["quality"]).head(5)
sample_data.to_csv("sample_wine_input.csv", index=False)
display(sample_data)
files.download("sample_wine_input.csv")
