# 💰 Loan Eligibility Prediction — Machine Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

*End-to-end machine learning pipeline for predicting loan eligibility — from raw data to deployed model.*

</div>

---

## 📌 Project Overview

This project builds an end-to-end **binary classification pipeline** to predict whether a loan applicant will be approved or rejected. It covers the complete ML workflow: **data exploration**, **preprocessing**, **feature engineering**, **model training and comparison**, and **performance evaluation**.

The dataset contains applicant demographics, financial attributes, and credit history — reflecting real-world lending scenarios.

---

## 📊 Dataset Features

| Feature | Type | Description |
|---------|------|-------------|
| `Gender` | Categorical | Male / Female |
| `Married` | Categorical | Marital status |
| `Dependents` | Numerical | Number of dependents |
| `Education` | Categorical | Graduate / Not Graduate |
| `Self_Employed` | Categorical | Employment type |
| `ApplicantIncome` | Numerical | Monthly income (₹) |
| `CoapplicantIncome` | Numerical | Co-applicant income (₹) |
| `LoanAmount` | Numerical | Requested loan amount |
| `Loan_Amount_Term` | Numerical | Repayment term (months) |
| `Credit_History` | Binary | 1 = Good, 0 = Bad |
| `Property_Area` | Categorical | Urban / Semiurban / Rural |
| **`Loan_Status`** | **Target** | **Y = Approved, N = Rejected** |

---

## 🔬 ML Pipeline

```
Raw Data → EDA → Preprocessing → Feature Engineering → Model Training → Evaluation → Best Model
```

### 1. Exploratory Data Analysis (EDA)
- Distribution plots for income, loan amount, credit history
- Correlation heatmap
- Class imbalance analysis

### 2. Preprocessing
- Handle missing values: median imputation for numerics, mode for categoricals
- Outlier treatment: IQR-based capping for `ApplicantIncome` and `LoanAmount`
- Encoding: Label encoding for binary features, one-hot for multi-class

### 3. Feature Engineering
- `Total_Income` = `ApplicantIncome` + `CoapplicantIncome`
- `EMI` = `LoanAmount` / `Loan_Amount_Term`
- `Income_Balance` = `Total_Income` − `EMI`
- Log transformation on skewed features

### 4. Models Compared

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 80.5% | 0.81 | 0.95 | 0.87 |
| Decision Tree | 76.4% | 0.79 | 0.90 | 0.84 |
| Random Forest | 82.1% | 0.83 | 0.94 | 0.88 |
| **Gradient Boosting** | **83.7%** | **0.85** | **0.93** | **0.89** |

> ✅ **Best Model: Gradient Boosting Classifier** with `n_estimators=200`, `learning_rate=0.05`

### 5. Key Finding
**Credit History** is the most influential feature — applicants with a good credit history have a **5× higher approval rate**.

---

## 📁 Repository Structure

```
Loan-Prediction/
│
├── data/
│   ├── train.csv                 # Training dataset
│   └── test.csv                  # Test dataset
│
├── notebooks/
│   ├── 01_EDA.ipynb              # Exploratory analysis
│   ├── 02_Preprocessing.ipynb    # Data cleaning & feature engineering
│   └── 03_Modelling.ipynb        # Model training & evaluation
│
├── src/
│   ├── preprocess.py             # Preprocessing pipeline
│   └── model.py                  # Training and prediction functions
│
├── outputs/
│   ├── model.pkl                 # Saved best model
│   └── predictions.csv           # Test set predictions
│
└── README.md
```

---

## ⚙️ How to Run

```bash
# Clone the repository
git clone https://github.com/pradeepnalawade00/Loan-Prediction.git
cd Loan-Prediction

# Install dependencies
pip install -r requirements.txt

# Run notebooks in order
jupyter notebook notebooks/01_EDA.ipynb
```

### Requirements
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
```

---

## 🎯 Learning Outcomes

- Complete end-to-end ML project lifecycle
- Feature engineering for financial datasets
- Model selection and hyperparameter tuning with cross-validation
- Interpreting business metrics (precision vs. recall trade-off in lending)

---

## 👤 Author

**Pradeep Nalawade** | ECE Student | Data Science Enthusiast

[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-A78BFA?style=flat-square)](https://pradeepnalawade00.github.io/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/pradeep-nalawade-950244314/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github)](https://github.com/pradeepnalawade00)
