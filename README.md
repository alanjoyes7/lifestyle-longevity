# Lifestyle-Based Health Pattern Discovery

A data-mining project that applies the Knowledge Discovery in Databases (KDD) process to public **NHANES 2017–2018** survey data. The project integrates demographic, physical activity, smoking, sleep, diet, body measurement, and health questionnaire data to explore lifestyle patterns and classify self-reported general health.

> **Scope:** The classification target is self-reported health—**“Good or better”** versus **“Fair or poor.”** This project does not predict life expectancy, diagnose disease, establish causation, or provide medical advice.

## What the project does

The pipeline in `kdd_pipeline.py`:

1. Downloads any missing NHANES XPT files from the CDC.
2. Selects relevant columns and integrates seven datasets using `SEQN` (participant ID).
3. Cleans selected refusal/don't-know codes, invalid values, and duplicate records.
4. Engineers derived features such as age and BMI groups, activity level, smoking status, sleep category, and health label.
5. Runs three data-mining methods:
   - **Apriori** for frequent itemsets and association rules.
   - **K-Means** for participant clusters and cluster profiles.
   - **Random Forest** classification for self-reported health.

The Streamlit interface in `app.py` provides data previews and data-quality summaries, exploratory charts, the Apriori and K-Means controls, classification evaluation, and an interactive prediction form.

## Project structure

```text
lifestyle_longevity_kdd/
├── app.py                 # Streamlit user interface
├── kdd_pipeline.py        # Data preparation and mining algorithms
├── view_datasets.py       # Converts raw XPT files to CSV
├── requirements.txt       # Python dependencies
└── data/
    ├── raw/               # NHANES XPT source files
    └── csv/               # CSV exports created by view_datasets.py
```

The seven source files are `DEMO_J`, `PAQ_J`, `SMQ_J`, `SLQ_J`, `DBQ_J`, `BMX_J`, and `HUQ_J`. They are stored under `data/raw/`; if a required XPT file is missing, the pipeline attempts to download it from the official CDC NHANES 2017 data files.

## Setup and run

Run these commands in PowerShell from the project directory:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

Streamlit opens the app in your browser. The first run may take longer because the pipeline reads and integrates the source datasets. The app caches the loaded dataset for the current Streamlit session.

If PowerShell prevents virtual-environment activation, use the environment's Python executable directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### Export the source datasets to CSV

To convert every XPT file in `data/raw/` to a CSV in `data/csv/`:

```powershell
py view_datasets.py
```

### Print or save the integrated dataset

The **Overview** tab shows a readable preview and provides a button to download the complete integrated dataset as CSV. To inspect it from Python:

```powershell
py -c "from kdd_pipeline import load_data; df = load_data(); print(df.shape); print(df.head(20).to_string(index=False))"
```

## Using the app

- **Overview:** KDD stages, integrated dataset preview, and full CSV download.
- **Data Quality:** Data types, missing counts and percentages, unique values, and duplicate participant IDs.
- **Explore Data:** Numeric distributions, descriptive statistics, and selected categorical distributions.
- **Apriori:** Adjust minimum support and confidence, then inspect frequent itemsets and association rules.
- **K-Means:** Choose the number of clusters; inspect cluster sizes, profiles, and silhouette score.
- **Classification:** Train and evaluate the Random Forest model; inspect accuracy, classification report, and confusion matrix.
- **Health Predictor:** Enter feature values to estimate the model's self-reported health category. The model is retrained when a prediction is submitted.

## Classification and evaluation

The target is derived from `HUQ010`:

| NHANES response | Project class |
|---|---|
| Excellent, very good, or good | Good or better |
| Fair or poor | Fair or poor |

The classifier uses demographic and lifestyle features. It excludes the direct health response (`HUQ010`) and its derived label (`general_health`) from the input features to avoid target leakage. Numeric missing values are imputed with the median and scaled; categorical missing values are imputed with the most frequent value and one-hot encoded. These preprocessing steps and the `RandomForestClassifier` are combined in a scikit-learn pipeline.

The code uses an 80/20 train/test split, `random_state=42`, 200 trees, and `class_weight="balanced"`. The app calculates the confusion matrix and classification report from the held-out test set. **Your results may vary if the source data or code changes.**

One run with the included data produced:

| Actual \ Predicted | Fair or poor | Good or better |
|---|---:|---:|
| Fair or poor | 172 | 147 |
| Good or better | 183 | 1,348 |

Accuracy for that run was approximately **82.16%** on **1,850** test records. The classes are imbalanced, and the weaker recall for “Fair or poor” means accuracy alone does not describe model performance; consult the classification report in the app as well.

## Interpreting the results

- **Confusion matrix:** Rows are actual classes; columns are predicted classes. The diagonal contains correct predictions.
- **Precision:** Of records predicted to be a class, the fraction that truly belong to it.
- **Recall:** Of records that truly belong to a class, the fraction the model identifies.
- **F1-score:** Harmonic mean of precision and recall.
- **Apriori support:** Fraction of transactions containing an itemset.
- **Apriori confidence:** Fraction of transactions with the antecedent that also contain the consequent.
- **Apriori lift:** Strength of co-occurrence relative to what would be expected if the items were independent. Association rules show co-occurrence, not causation.
- **Silhouette score:** A measure of how well-separated the K-Means clusters are; higher is generally better, but cluster usefulness also requires interpretation.

## Data and methodological limitations

- NHANES is observational survey data. Associations found by the project do not demonstrate cause and effect.
- The target is a participant's self-reported general health, not a clinical outcome or a measure of lifespan.
- This implementation evaluates one random train/test split and does not provide external validation or clinical validation.
- The model does not use NHANES survey weights, strata, or primary sampling units; results should not be interpreted as population-weighted estimates.
- Missing-value handling and derived feature definitions affect results. Review `clean_data()` and `engineer_features()` in `kdd_pipeline.py` before interpreting them.
- Model probabilities are model estimates, not calibrated clinical probabilities.

## Data source

National Health and Nutrition Examination Survey (NHANES), 2017–2018, National Center for Health Statistics, Centers for Disease Control and Prevention. The project downloads public XPT data from the official [NHANES 2017–2018 data files](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017).
