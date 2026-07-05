# Diabetes Early Detection — ML Model Comparison

An interpretable machine learning comparison for early diabetes detection, built as the original technical contribution accompanying my independent research paper, *"Artificial Intelligence for Early Detection of Disease: A Study on Diabetes, Cardiovascular Disease, Skin Cancer, and Breast Cancer."*

## Why I Built This

I didn't want this project to be another literature review that just cites other people's models and moves on. I wanted to actually build one, get real numbers, and be honest about where those numbers fall short of what's already published — because that gap turned out to be one of the more interesting findings in the whole project, not just a footnote. This repository is that attempt: a small, transparent, fully reproducible piece of original research sitting alongside the broader paper.

## Project Highlights

- ✔ Literature Review (4 diseases, published AI performance benchmarked against traditional diagnosis)
- ✔ Primary Survey (47 respondents, public trust in AI-based diagnosis)
- ✔ Original ML Pipeline (3 models, trained and evaluated from scratch)
- ✔ Exploratory Data Analysis (correlation analysis, class distribution, feature distributions)
- ✔ Cross-Validation Robustness Check (not just a single train/test split)
- ✔ Full Research Paper (11 chapters, methodology through discussion and conclusion)
- ✔ Future Deployment Directions (see Future Work below)

## Quick Start

```bash
git clone https://github.com/SG975310/AI_DETECTION_DIABETES-.git
cd AI_DETECTION_DIABETES-
pip install -r requirements.txt
python src/train_models.py
python src/make_charts.py
```

That's it — running those three commands reproduces every number and chart in this README.

## How This Repository Connects to the Research Paper

This repository isn't a standalone coding exercise — it's one piece of a larger research project, and the pieces feed into each other:

```
Research Question (paper, Ch. 1)
        ↓
Literature Review — what traditional diagnosis looks like (paper, Ch. 3)
        ↓
This Repository — an original model, built to test that literature directly (paper, Ch. 8)
        ↓
Results + Cross-Validation — a real, independently-produced data point (below)
        ↓
Survey — public trust in AI-based diagnosis (paper, Ch. 9)
        ↓
Discussion — do the model results and the survey findings agree? (paper, Ch. 10)
        ↓
Conclusion — does AI actually improve early detection? (paper, Ch. 11)
```

The full paper is the best place to see how this specific model fits into that bigger picture.

## Research Question

This project supports the paper's central research question: **can artificial intelligence improve the early detection of disease compared with traditional methods?** For diabetes specifically, the traditional method is a single blood test (HbA1c or fasting glucose) applied after a person seeks testing. This project asks whether a machine learning model, using the same kind of routine health data, can flag risk earlier or more consistently than that single-threshold approach — and, just as importantly, whether one ML algorithm is actually more trustworthy than another once you look past a single accuracy number.

## Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 70.78% | 60.00% | 50.00% | 54.55% | 0.8130 |
| Random Forest | 74.03% | 65.22% | 55.56% | 60.00% | 0.8161 |
| **XGBoost** | **77.27%** | **70.21%** | **61.11%** | **65.35%** | 0.8154 |

![Model comparison](images/chart_model_comparison.png)

XGBoost had the highest performance across every metric on this particular train/test split, which is typical of the algorithm on structured data, according to the literature reviewed. Recall is worth flagging specifically: even XGBoost only caught about 61% of actual diabetic cases in the test set, meaning close to 4 in 10 diabetic patients in this test set would have been missed. That's the kind of detail that gets hidden if you only report accuracy — which is exactly why this project reports all five metrics, not just the one that looks best.

### Cross-Validation: The Most Important Result in This Repo

A single 80/20 split can make a model look better or worse than it really is, just based on which patients happened to land in the test set. To check this, all 3 models were also run through 5-fold stratified cross-validation:

![Cross-validation results](images/chart_cross_validation.png)

**On the single train/test split, XGBoost had the highest accuracy (77.27%). But under 5-fold cross-validation, Logistic Regression actually came out on top (77.34% average, with the lowest variance across folds), while XGBoost's average cross-validated accuracy (74.21%) was noticeably lower than its single-split result.**

This suggests XGBoost's single-split performance may have been partly a good draw of which patients ended up in that particular test set, rather than a consistently superior model. It's a useful reminder — for this project and in general — that a single train/test split isn't enough to declare a "winner," and that cross-validation is what actually tells you how stable a model's performance is.

### Confusion Matrices

![Confusion matrices](images/chart_confusion_matrices.png)

All three models produce more false negatives (missed diabetic cases) than false positives, which is the less safe direction to err in for a screening tool — a missed case delays treatment, while a false alarm just means an extra follow-up test.

### ROC Curves

![ROC curves](images/chart_roc_curves.png)

## Feature Importance

![Feature importance](images/chart_feature_importance.png)

The most important variable was glucose, followed by BMI and Diabetes Pedigree Function (which is related to genetics). This validates the claims made in Section 4.3 ("Common Algorithms Used in Detection Models") of the paper, as evidenced by the emphasis placed on blood sugar levels and hemoglobin A1c as diagnostic criteria for diabetes. It also matches the correlation heatmap below — the feature most correlated with the outcome is also the feature the model leaned on most.

## Sample Output

Running `python src/train_models.py` produces logged output like this:

```
17:32:05 [INFO] Dataset shape: (768, 9)
17:32:05 [INFO] Diabetic cases: 268 (34.9%)
17:32:05 [INFO] Non-diabetic cases: 500 (65.1%)
17:32:05 [INFO] Zero-value ('missing') counts per column: {'Glucose': 5, 'BloodPressure': 35, 'SkinThickness': 227, 'Insulin': 374, 'BMI': 11}
17:32:05 [INFO] Training set: 614 patients
17:32:05 [INFO] Test set: 154 patients
17:32:05 [INFO] --- Train/Test Split Evaluation ---
17:32:05 [INFO] Logistic Regression: {'accuracy': 70.78, 'precision': 60.0, 'recall': 50.0, 'f1_score': 54.55, 'roc_auc': 0.813}
17:32:06 [INFO] Random Forest: {'accuracy': 74.03, 'precision': 65.22, 'recall': 55.56, 'f1_score': 60.0, 'roc_auc': 0.8161}
17:32:06 [INFO] XGBoost: {'accuracy': 77.27, 'precision': 70.21, 'recall': 61.11, 'f1_score': 65.35, 'roc_auc': 0.8154}
17:32:06 [INFO] --- 5-Fold Cross-Validation (robustness check) ---
17:32:06 [INFO] Logistic Regression — 5-fold CV accuracy: 77.34% (+/- 1.56%)
17:32:08 [INFO] Random Forest — 5-fold CV accuracy: 76.04% (+/- 2.06%)
17:32:08 [INFO] XGBoost — 5-fold CV accuracy: 74.21% (+/- 2.74%)
17:32:08 [INFO] Results saved to results/results.json
```

For an interactive walkthrough with all the EDA plots inline, see `notebooks/analysis.ipynb`.

## Dataset

Source: Pima Indians Diabetes Dataset (originally collected by the National Institute of Diabetes and Digestive and Kidney Diseases)

- Size: 768 patients (all female, of Pima Indian heritage, age 21+)
- Features: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age
- Target: Outcome (1 = diabetic, 0 = non-diabetic)
- Class balance: 268 diabetic cases (34.9%), 500 non-diabetic cases (65.1%)

![Class distribution](images/chart_class_distribution.png)

The dataset is imbalanced (roughly 2:1 non-diabetic to diabetic), which is part of the reason this project reports precision, recall, and F1 alongside accuracy rather than accuracy alone — a model can score a deceptively high accuracy just by leaning toward the majority class.

## Exploratory Data Analysis

![Correlation heatmap](images/chart_correlation_heatmap.png)

Glucose has the strongest correlation with the diabetes outcome of any single feature, which foreshadows the feature importance result above — the model and the raw correlation numbers agree with each other. A full interactive version of this analysis, including feature distributions split by outcome, is in `notebooks/analysis.ipynb`.

## Preprocessing

Several columns in this dataset use 0 as a placeholder for missing data (a documented quirk of this specific dataset — a value of 0 for Blood Pressure or BMI is not physiologically possible in a living patient). These zero-values were treated as missing and imputed with the column median:

| Column | Missing values (encoded as 0) |
| --- | --- |
| Glucose | 5 |
| Blood Pressure | 35 |
| Skin Thickness | 227 |
| Insulin | 374 |
| BMI | 11 |

## Models Compared

1. Logistic Regression
2. Random Forest
3. XGBoost (Extreme Gradient Boosting)

All models were trained on an 80-20 train-test split (with stratification applied to the split to reduce potential bias). The data for the Logistic Regression was standardized. Each model was picked for a specific reason rather than just "trying a few things": Logistic Regression as a simple, interpretable baseline; Random Forest as a stronger ensemble method that's still fairly resistant to overfitting; and XGBoost because it's the algorithm most consistently reported as top-performing on structured medical data in the literature this project is built on.

The performance of all 3 algorithms was compared using accuracy, precision, recall, F1 score, and ROC-AUC as evaluation metrics — not accuracy alone, since a single metric can hide how a model is actually failing (see Results above).

## Comparison to Published Literature

This project's 77.27% accuracy (XGBoost, single split) is significantly lower than the 99.41% accuracy achieved by Iparraguirre-Villanueva et al. (2023).

Reasons for this discrepancy could include:

1. **Different Datasets:** It's entirely possible that the research of Iparraguirre-Villanueva et al. was conducted on a larger dataset than the famous Pima Indians dataset used for this project.
2. **Class Imbalance:** Another reason could be the imbalance between the number of samples of each class (768 patients, of which only 268 were diabetic). This is especially likely given that the test set only had 154 patients (25.8% of the dataset). There is obviously a possibility of inaccuracy due to a small sample size.
3. **No Hyperparameter Tuning:** The 3 models weren't tuned for optimal performance. The default parameters (with some minor tweaking) were sufficient for the purpose of this project, which is primarily academic.

However, this is an academic exercise meant to contribute to the body of knowledge on AI/ML applications in early diagnosis, and this particular result actually serves to further one of the theses of this paper: namely, that there is a significant amount of bias in data science. This result can be used to support the claims made in Section 6.1 ("Bias in Training Data") of the paper, as it shows how different results can come from similar methods due to differences in data and hyperparameters.

## Repository Structure

```
diabetes-ml-detection/
├── data/
│   └── diabetes.csv                    # Pima Indians Diabetes Dataset
├── src/
│   ├── train_models.py                 # Main pipeline: load, clean, train, evaluate, cross-validate
│   ├── make_charts.py                  # Generates all comparison and EDA charts
│   └── make_workflow_diagram.py        # Generates the workflow diagram
├── notebooks/
│   └── analysis.ipynb                  # Interactive EDA + results walkthrough
├── results/
│   └── results.json                    # Full numerical results (metrics, CV, feature importance)
├── images/
│   ├── workflow_diagram.png
│   ├── chart_model_comparison.png
│   ├── chart_roc_curves.png
│   ├── chart_confusion_matrices.png
│   ├── chart_cross_validation.png
│   ├── chart_correlation_heatmap.png
│   ├── chart_class_distribution.png
│   └── chart_feature_importance.png
├── requirements.txt                    # Pinned Python dependencies
├── LICENSE                             # MIT License
└── README.md
```

## Workflow

![Workflow diagram](images/workflow_diagram.png)

## How to Reproduce

```bash
pip install -r requirements.txt
python src/train_models.py            # trains all 3 models, runs 5-fold CV, saves results/results.json
python src/make_charts.py             # generates all comparison and EDA charts into images/
python src/make_workflow_diagram.py   # generates the workflow diagram into images/
```

All random operations (train/test split, model initialization, cross-validation folds) use a fixed random seed (42), so re-running the pipeline reproduces the exact numbers reported in this README.

## Limitations

1. **Small dataset** (768 patients total, 154 in the test set), which can lead to less stable results — the cross-validation results above suggest this project's single train/test split may not be fully representative of true model performance.
2. **Non-representative population** — this dataset consists exclusively of Pima Indian women, age 21+, from an American population sample. It is not representative of the broader population this project's paper is ultimately concerned with (an Indian population), which is one of the more important limitations of this specific project — the model's real-world validity outside this narrow population is untested. This ties directly into the bias discussion in Section 6.1 of the paper.
3. **Lack of hyperparameter tuning** — all 3 models were run close to default settings, so these results represent a baseline rather than each model's best possible performance.
4. **Potential bias from dataset age and collection method** — this dataset is decades old and was collected under specific clinical criteria; newer diagnostic criteria or a more diverse collection process could change these results.
5. This should be taken as an academic exercise and not implemented anywhere close to a real-world clinical setting.

## Future Work

- **Hyperparameter tuning** (e.g. grid search or Bayesian optimization) for all 3 models, to see whether the gap to the published 99.41% closes at all once each model is actually optimized rather than run near-default.
- **Deep learning approaches** (e.g. a simple feedforward neural network) as a fourth comparison point alongside the three classical ML models used here.
- **A larger, more diverse dataset**, ideally including male patients and a wider age range, to test whether the cross-validation vs. single-split discrepancy found here holds up, shrinks, or reverses with more data.
- **An Indian population dataset** specifically, to directly address this project's most significant limitation and test whether these findings generalize beyond the Pima Indian population this dataset represents.
- **SHAP value analysis** for a more granular, per-prediction explanation of feature contributions, beyond the global feature importance shown here.
- **Mobile or web deployment** of the best-validated model as a lightweight risk-screening tool, to explore what a real-world, non-clinical-setting application of this kind of model might look like — with the explicit caveat that this would require far more validation than this project provides before being clinically appropriate.

## Author

Soham - Class 12 (PCB), independent research project, 2026.

## Citation

If you're going to use this, here's the citation: "Artificial Intelligence for Early Detection of Disease: A Study on Diabetes, Cardiovascular Disease, Skin Cancer, and Breast Cancer."
