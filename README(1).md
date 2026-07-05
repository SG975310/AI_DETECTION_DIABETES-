# Diabetes Early Detection - ML Model Comparison

The code for this project comes from a paper comparing the efficacy of 3 classification algorithms for early diabetes detection: Logistic Regression, Random Forest, and XGBoost. The paper, "Artificial Intelligence for Early Detection of Disease: A Study on Diabetes, Cardiovascular Disease, Skin Cancer, and Breast Cancer," is referenced throughout this README.

## Project Overview

This project attempts to create 3 machine learning models that can determine if a patient has diabetes based on 8 indicators (variables) in order to predict the likelihood of developing the disease at an early stage. The code for this project implements the methodology used in the published literature (Iparraguirre-Villanueva et al., 2023; Ullah et al., 2022) and builds upon it in an effort to contribute to research regarding the benefits of artificial intelligence in early diagnosis of disease.

## Dataset

Source: Pima Indians Diabetes Dataset (originally collected by the National Institute of Diabetes and Digestive and Kidney Diseases)

- Size: 768 patients (all female, of Pima Indian heritage, age 21+)
- Features: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age
- Target: Outcome (1 = diabetic, 0 = non-diabetic)
- Class balance: 268 diabetic cases (34.9%), 500 non-diabetic cases (65.1%)

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

All models were trained on an 80-20 train-test split (with stratification applied to the split to reduce potential bias). The data for the Logistic Regression was standardized.

The performance of all 3 algorithms was compared using accuracy, precision, recall, F1 score, and ROC-AUC as evaluation metrics.

XGBoost had the highest performance across every metric, which is typical of the algorithm on structured data, according to the literature reviewed.

## Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 70.78% | 60.00% | 50.00% | 54.55% | 0.8130 |
| Random Forest | 74.03% | 65.22% | 55.56% | 60.00% | 0.8161 |
| XGBoost | 77.27% | 70.21% | 61.11% | 65.35% | 0.8154 |

This particular project's accuracy (77.27%, achieved using the XGBoost model) pales in comparison to the 99.41% accuracy achieved by Iparraguirre-Villanueva et al. (2023) - but it is expected, for reasons outlined in detail in this project's "Comparison to Published Literature" section.

## Feature Importance (according to the Random Forest model)

The most important variable was glucose, followed by BMI and Diabetes Pedigree Function (which is related to genetics). This validates the claims made in Section 4.3 ("Common Algorithms Used in Detection Models") of the paper, as evidenced by the emphasis placed on blood sugar levels and hemoglobin A1c as diagnostic criteria for diabetes.

## Comparison to Published Literature

As mentioned earlier, this project's 77.27% accuracy is significantly lower than the 99.41% accuracy achieved by Iparraguirre-Villanueva et al. (2023).

Reasons for this discrepancy could include:

1. **Different Datasets:** It's entirely possible that the research of Iparraguirre-Villanueva et al. was conducted on a larger dataset than the famous Pima Indians dataset used for this project.
2. **Class Imbalance:** Another reason could be the imbalance between the number of samples of each class (768 patients, of which only 268 were diabetic). This is especially likely given that the test set only had 154 patients (25.8% of the dataset). There is obviously a possibility of inaccuracy due to a small sample size.
3. **No Hyperparameter Tuning:** The 3 models weren't tuned for optimal performance. The default parameters (with some minor tweaking) were sufficient for the purpose of this project, which is primarily academic.

However, this is an academic exercise meant to contribute to the body of knowledge on AI/ML applications in early diagnosis, and this particular result actually serves to further one of the theses of this paper: namely, that there is a significant amount of bias in data science. This result can be used to support the claims made in Section 6.1 ("Bias in Training Data") of the paper, as it shows how different results can come from similar methods due to differences in data and hyperparameters.

## Repository Structure

```
diabetes-ml-detection/
├── diabetes.csv                    # Pima Indians Diabetes Dataset
├── train_models.py                 # Main pipeline: load, clean, train, evaluate
├── make_charts.py                  # Generates comparison charts
├── results.json                    # Full numerical results
├── chart_model_comparison.png      # Bar chart: accuracy/precision/recall/F1 by model
├── chart_roc_curves.png            # ROC curves for all three models
├── chart_feature_importance.png    # Feature importance (Random Forest)
├── requirements.txt                # Python dependencies
└── README.md
```

## How to Reproduce

```bash
pip install -r requirements.txt
python train_models.py    # trains all 3 models, prints metrics, saves results.json
python make_charts.py     # generates all comparison charts from results.json
```

## Limitations

1. Small dataset, which can lead to less accurate results
2. Non-representative sample (only Pima Indian women of age 21+) - this limitation ties into the potential sources of bias mentioned in Section 6.1
3. Lack of hyperparameter tuning
4. This should be taken as an academic exercise and not implemented anywhere close to a real-world setting

## Author

Soham - Class 12 (PCB), independent research project, 2026.

## Citation

If you're going to use this, here's the citation: "Artificial Intelligence for Early Detection of Disease: A Study on Diabetes, Cardiovascular Disease, Skin Cancer, and Breast Cancer."
