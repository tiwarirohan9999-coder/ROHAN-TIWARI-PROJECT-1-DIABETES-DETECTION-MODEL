# Diabetes Prediction — ML Classification Project

Binary classification project predicting whether a patient has diabetes from demographic and clinical features, comparing Logistic Regression, K-Nearest Neighbors, and a Decision Tree.

## Dataset

[Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) (Kaggle, ~100,000 patient records).

Features: gender, age, hypertension, heart disease, smoking history, BMI, HbA1c level, blood glucose level.
Target: `diabetes` (0 = no, 1 = yes). The dataset is imbalanced — about 91.5% negative, 8.5% positive.

## Approach

1. **Encoding** — `gender` and `smoking_history` are label-encoded to numeric values.
2. **Train/test split** — 80/20, stratified on the target.
3. **Scaling** — features scaled with `StandardScaler`.
4. **Class imbalance** — `SMOTE` applied to the training set to balance the two classes before training.
5. **Models** — three classifiers trained and compared: Logistic Regression, KNN (k=5), and a Decision Tree (max depth 5).
6. **Evaluation** — accuracy and a full classification report (precision/recall/F1) for each model; the model with the highest accuracy is selected as "best" and gets a confusion matrix plot.

## Results

| Model | Accuracy |
|---|---|
| Logistic Regression | 88.78% |
| KNN (K=5) | 91.40% |
| Decision Tree | 86.95% |

**Best model (by accuracy): KNN (K=5)**

### A note on these numbers

Accuracy is the headline metric here, but worth knowing: since the dataset is ~91.5% non-diabetic, a model that predicted "no diabetes" for every single patient would already score close to 91.5% accuracy without learning anything. Looking at the full classification report is more informative — for example, the Decision Tree only hits ~39% precision on the diabetes class (meaning many of its positive predictions are false alarms) despite catching ~92% of actual diabetic cases (high recall). KNN has the best accuracy but a lower recall (~80%) than Logistic Regression or the Decision Tree, meaning it misses more true diabetic cases. Which model is "best" really depends on whether false alarms or missed cases are more costly in your use case — accuracy alone doesn't capture that trade-off.

## Repo structure

```
.
├── diabetes_prediction_dataset.csv      # input data (from Kaggle)
├── diabetes_model_training.py           # main training/eval script
├── requirements.txt
├── model_comparison.png                 # generated when you run the script
└── confusion_matrix.png                 # generated when you run the script
```

## How to run

```bash
pip install -r requirements.txt
python diabetes_model_training.py
```

This prints accuracy and a classification report for all three models to the console, and saves `model_comparison.png` and `confusion_matrix.png` in the same folder the script is run from.

## Limitations

- This is an educational project using a public dataset, not a validated clinical tool — predictions should not be used for real medical decisions.
- Model selection is based on accuracy alone, which can be misleading on imbalanced data (see note above). Precision/recall trade-offs aren't accounted for in the automatic "best model" choice.
- `gender` and `smoking_history` are label-encoded rather than one-hot encoded, which imposes an arbitrary numeric order on categories that don't actually have one.
- Hyperparameters (KNN's k=5, Decision Tree's max_depth=5) are fixed values, not tuned via cross-validation.

## AUTHOR
ROHAN TIWARI , RESEARCHER , TIET-UQ   
