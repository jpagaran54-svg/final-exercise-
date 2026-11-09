# Wine Quality Prediction System

## Project
A classroom prototype that classifies red-wine samples as GOOD QUALITY or NEEDS IMPROVEMENT using physicochemical measurements.

## Prediction rule
- quality >= 6 → GOOD QUALITY
- quality < 6 → NEEDS IMPROVEMENT

The original `quality` column is the target and is excluded from the input features to avoid target leakage.

## Model
Random Forest, 150 trees, `random_state=42`, `class_weight="balanced"`, with an 80/20 stratified train/test split. Median imputation and standard scaling are included in the pipeline.

## Your actual evaluation results
These are the results from the user's Colab run:

- Accuracy: 77.21%
- Precision: 79.29%
- Recall: 77.08%
- F1-score: 78.17%

Class-level results:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Needs Improvement | 0.75 | 0.77 | 0.76 | 128 |
| Good Quality | 0.79 | 0.77 | 0.78 | 144 |

Confusion matrix:

```text
[[99 29]
 [33 111]]
```

## Run

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

The local Streamlit address is normally `http://localhost:8501`.

## Limitation
The model is trained on a particular public red-wine dataset and may not generalize to other wine types, producers, regions, or testing conditions. It is a preliminary screening aid, not a replacement for laboratory testing, sensory evaluation, food-safety procedures, or professional quality-control decisions.
