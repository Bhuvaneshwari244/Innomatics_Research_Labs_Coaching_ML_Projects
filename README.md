# Telco Customer Churn Prediction Using Machine Learning

An end-to-end supervised machine learning project that predicts whether a telecommunications customer is likely to **churn (leave the service)** or **stay**, using customer demographics, account details, subscribed services, and billing information.

## Project Overview

Customer churn can affect revenue and long-term business growth. This project uses historical customer data to identify patterns associated with churn and build a binary classification model that estimates the likelihood of a customer leaving.

- **Problem type:** Supervised Learning — Binary Classification
- **Target column:** `Churn`
- **Target encoding:** `1 = Yes (churn)`, `0 = No (stay)`
- **Dataset size:** 7,043 customer records and 21 original columns
- **Predictor features:** 19 (after removing `customerID` and separating the target)

## Project Workflow

1. Business understanding and problem definition
2. Dataset loading and data understanding
3. Data cleaning and quality checks
4. Exploratory Data Analysis (EDA)
5. Feature and target preparation
6. Stratified train-test split
7. Numerical and categorical preprocessing
8. Model building and training
9. Model evaluation and comparison
10. Best model selection
11. Model and feature-column serialization
12. Interactive prediction interface using Gradio

## Dataset

The project uses the Telco Customer Churn dataset. Each row represents a customer and includes demographic, account, service-subscription, and billing attributes.

The original dataset contains 7,043 rows and 21 columns. The `customerID` field is removed because it is an identifier, not a behavioural predictor. The remaining 19 columns are used as model inputs, with `Churn` as the target.

### Feature groups

| Group | Example features |
|---|---|
| Demographic | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Account | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod` |
| Phone services | `PhoneService`, `MultipleLines` |
| Internet services | `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| Billing | `MonthlyCharges`, `TotalCharges` |

**Dataset file expected by the notebook:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`

Place the CSV file in the project root (the same folder as the notebook). If you have the dataset in Excel format, export/save it as CSV with the filename above before running the notebook.

## Data Cleaning and Preprocessing

- Converted `TotalCharges` to numeric values; whitespace-only entries become missing values.
- Detected 11 missing values in `TotalCharges` after conversion.
- Checked for duplicate records; no complete duplicate rows were found.
- Inspected numerical feature distributions, outliers, and correlations.
- Used an **80:20 stratified train-test split** with `random_state=42`:
  - Training records: 5,634
  - Testing records: 1,409
- Numerical features: median imputation followed by `StandardScaler`.
- Categorical features: most-frequent-value imputation followed by `OneHotEncoder(drop='first', handle_unknown='ignore')`.
- Combined the transformations with `ColumnTransformer` and included preprocessing inside each model pipeline to avoid fitting preprocessing on the test data.

Outlier inspection was descriptive; the notebook does not remove the inspected observations.

## Machine Learning Models

Four classification algorithms were trained using the same predictors, preprocessing approach, and held-out test set:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier (`n_estimators=100`)
4. K-Nearest Neighbors (`n_neighbors=5`)

## Model Evaluation

The models were compared using Accuracy, ROC-AUC, and classification-report metrics (Precision, Recall, and F1-score).

| Model | Accuracy | ROC-AUC |
|---|---:|---:|
| Logistic Regression | **0.806** | **0.842** |
| Decision Tree | 0.741 | 0.661 |
| Random Forest | 0.786 | 0.824 |
| K-Nearest Neighbors | 0.767 | 0.793 |

**Selected model: Logistic Regression**

It achieved the highest accuracy and ROC-AUC among the four models in this experiment. On the held-out test set, it achieved **80.6% accuracy** and **0.842 ROC-AUC**.

## Interactive Demo with Gradio

A Gradio interface allows a user to enter the 19 customer attributes and receive:

- Predicted customer status: **likely to Churn** or **likely to Stay**
- Model-estimated churn probability

The notebook launches the interface with `demo.launch(share=True)`. The generated `gradio.live` URL is temporary and should be treated as a demonstration link, not permanent hosting.

## Project Files

A typical repository structure:

```text
Telco-Customer-Churn-Prediction/
│
├── Telco Customer Churn Prediction.ipynb
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── churn_model.pkl
├── model_columns.pkl
├── requirements.txt
└── README.md
```

- `Telco Customer Churn Prediction.ipynb` — complete analysis, preprocessing, model training, evaluation, and Gradio demo.
- `WA_Fn-UseC_-Telco-Customer-Churn.csv` — dataset file required by the notebook.
- `churn_model.pkl` — saved best-performing fitted preprocessing and Logistic Regression pipeline.
- `model_columns.pkl` — saved input feature-column order expected by the model.
- `requirements.txt` — Python dependencies.

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
```

### 2. Create and activate a virtual environment (recommended)

**Windows PowerShell**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you do not have a `requirements.txt` yet, install the libraries used in the notebook:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter ipykernel gradio openpyxl
```

### 4. Add the dataset

Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the repository root. The notebook reads this exact relative path:

```python
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
```

### 5. Run the notebook

```bash
jupyter notebook
```

Open `Telco Customer Churn Prediction.ipynb` and run the cells in order. The notebook trains and evaluates the models, saves `churn_model.pkl` and `model_columns.pkl`, then launches the Gradio interface.

## Saved Model Artifacts

The selected pipeline and feature-column list are saved with Python's `pickle` module:

- `churn_model.pkl`
- `model_columns.pkl`

The fitted pipeline includes the preprocessing transformations and Logistic Regression estimator. Keep the model artifacts and the environment's compatible package versions together when loading the model.

> **Security note:** Only load pickle files from sources you trust. Pickle deserialization can execute arbitrary code.

## Key Learnings

- Translating a business problem into a binary classification task.
- Exploring customer data and checking data quality.
- Building separate preprocessing pipelines for numerical and categorical features.
- Preventing preprocessing leakage by fitting transformations within model pipelines.
- Comparing multiple classification algorithms with consistent evaluation.
- Saving a complete fitted pipeline and reusing it for new predictions.
- Building an interactive ML demonstration with Gradio.

## Future Improvements

- Tune model hyperparameters using cross-validation.
- Evaluate business-relevant churn recall, precision, and threshold trade-offs.
- Investigate class imbalance and calibration where appropriate.
- Add explainability to help understand the factors behind individual predictions.
- Deploy the application to a permanent hosting platform.

## Author

**Bhuvaneshwari Rebba**

If you find this project useful, feel free to ⭐ the repository and share your feedback!

---

*This is an educational machine learning project. Predictions are model estimates based on the dataset and should not be treated as guaranteed customer behaviour.*
