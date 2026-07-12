# Data Dictionary - COVID Drug Effect Analysis

## Author: Prathap Ramachandra
## Last Updated: July 2026

---

## Patients Table

| Field Name | Data Type | Description | Example |
|---|---|---|---|
| patient_id | STRING | Unique patient identifier | PAT_001 |
| age | INTEGER | Patient age in years | 45 |
| gender | STRING | Patient gender | M, F |
| blood_type | STRING | Patient blood type | A+, O-, B+ |
| comorbidities | STRING | Existing health conditions | Diabetes, Hypertension |

---

## Treatments Table

| Field Name | Data Type | Description | Example |
|---|---|---|---|
| treatment_id | STRING | Unique treatment identifier | TRT_001 |
| patient_id | STRING | Patient identifier (FK) | PAT_001 |
| treatment_type | STRING | Type of treatment given | DRUG_A, DRUG_B |
| treatment_start_date | DATE | Treatment start date | 2021-01-15 |
| treatment_end_date | DATE | Treatment end date | 2021-01-30 |
| dosage | DOUBLE | Treatment dosage amount | 500.00 |

---

## Outcomes Table

| Field Name | Data Type | Description | Example |
|---|---|---|---|
| outcome_id | STRING | Unique outcome identifier | OUT_001 |
| patient_id | STRING | Patient identifier (FK) | PAT_001 |
| recovery_status | STRING | Patient recovery status | RECOVERED, NOT_RECOVERED |
| recovery_days | INTEGER | Days taken to recover | 14 |
| hospital_stay_days | INTEGER | Days spent in hospital | 7 |
| outcome_date | DATE | Date of outcome recorded | 2021-02-01 |

---

## Engineered Features

| Feature Name | Data Type | Description | Values |
|---|---|---|---|
| age_group | STRING | Patient age category | Under18, 18-35, 36-50, 51-65, Over65 |
| gender_encoded | INTEGER | Encoded gender | 0=Female, 1=Male |
| treatment_encoded | INTEGER | Encoded treatment type | 0, 1, 2, 3 |
| treatment_duration | INTEGER | Length of treatment in days | Number of days |
| has_comorbidity | INTEGER | Comorbidity flag | 0=No, 1=Yes |
| outcome_label | INTEGER | Target variable | 0=Not Recovered, 1=Recovered |

---

## ML Models Used

| Model | Type | Key Parameters |
|---|---|---|
| Logistic Regression | Classification | max_iter=1000, random_state=42 |
| Naive Bayes | Probabilistic | GaussianNB |
| Gradient Boosting | Ensemble | n_estimators=100, learning_rate=0.1 |
| XGBoost | Boosting | n_estimators=100, max_depth=4 |

---

## Model Evaluation Metrics

| Metric | Description | Formula |
|---|---|---|
| Accuracy | Overall correct predictions | Correct / Total * 100 |
| Precision | True positives out of predicted positives | TP / (TP + FP) |
| Recall | True positives out of actual positives | TP / (TP + FN) |
| F1 Score | Balance between precision and recall | 2 * (P * R) / (P + R) |
| CV Score | Cross validation score across 5 folds | Mean of 5 fold scores |

---

## Treatment Types

| Treatment | Description |
|---|---|
| DRUG_A | First line treatment option |
| DRUG_B | Second line treatment option |
| DRUG_C | Third line treatment option |
| PLACEBO | Control group — no active treatment |

---

## Recovery Status

| Status | Description |
|---|---|
| RECOVERED | Patient fully recovered from COVID-19 |
| NOT_RECOVERED | Patient did not recover within study period |

---

## Age Groups

| Group | Age Range | Description |
|---|---|---|
| Under18 | 0-17 years | Minor patients |
| 18-35 | 18-35 years | Young adults |
| 36-50 | 36-50 years | Middle aged adults |
| 51-65 | 51-65 years | Older adults |
| Over65 | 65+ years | Senior patients |
