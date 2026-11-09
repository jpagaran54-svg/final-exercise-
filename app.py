
import streamlit as st
import pandas as pd
import joblib
import json

st.set_page_config(
    page_title="Wine Quality Prediction System",
    page_icon="🍷",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("wine_model.pkl")

model = load_model()
data = pd.read_csv("wine_quality.csv")

st.title("🍷 Wine Quality Prediction System")
st.caption("Data Mining Final Project")

st.write(
    "This system predicts whether a wine sample is "
    "GOOD QUALITY or NEEDS IMPROVEMENT."
)

# DASHBOARD METRICS
c1, c2, c3 = st.columns(3)

c1.metric("Total Records", len(data))
c2.metric(
    "Average Quality",
    round(data["quality"].mean(), 2)
)
c3.metric(
    "Good Quality Records",
    int((data["quality"] >= 6).sum())
)

# DATASET PREVIEW
st.subheader("Dataset Preview")
st.dataframe(data.head(10), use_container_width=True)

# MODEL EVALUATION
st.subheader("Model Evaluation Results")

with open("model_metrics.json") as f:
    metrics = json.load(f)

m1, m2, m3, m4 = st.columns(4)

m1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
m2.metric("Precision", f"{metrics['precision_good_quality']:.2%}")
m3.metric("Recall", f"{metrics['recall_good_quality']:.2%}")
m4.metric("F1-Score", f"{metrics['f1_good_quality']:.2%}")

# SAMPLE CSV DOWNLOAD
st.subheader("Download Sample Input")

features = list(model.feature_names_in_)

sample = data[features].head(5)

st.download_button(
    "Download Sample CSV",
    sample.to_csv(index=False),
    "sample_wine_input.csv",
    "text/csv"
)

# PREDICTION
st.subheader("Wine Quality Prediction")

uploaded = st.file_uploader(
    "Upload Wine CSV File",
    type=["csv"]
)

if uploaded is not None:

    try:
        new_data = pd.read_csv(uploaded)

        missing = [
            col for col in features
            if col not in new_data.columns
        ]

        if missing:
            st.error(
                "Missing required columns: "
                + ", ".join(missing)
            )

        else:
            predictions = model.predict(
                new_data[features]
            )

            results = new_data.copy()

            results["Prediction"] = [
                "GOOD QUALITY" if p == 1
                else "NEEDS IMPROVEMENT"
                for p in predictions
            ]

            st.subheader("Prediction Results")

            st.dataframe(
                results,
                use_container_width=True
            )

            good = int((predictions == 1).sum())
            needs = int((predictions == 0).sum())

            c1, c2 = st.columns(2)

            c1.metric("Good Quality", good)
            c2.metric("Needs Improvement", needs)

            st.download_button(
                "Download Prediction Results",
                results.to_csv(index=False),
                "wine_quality_predictions.csv",
                "text/csv"
            )

    except Exception as e:
        st.error(f"Error: {e}")

st.warning(
    "This is a classroom prototype. Predictions are "
    "preliminary screening results and do not replace "
    "laboratory testing or professional quality assessment."
)
