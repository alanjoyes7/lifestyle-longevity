from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


# ---------------------------------------------------------
# PROJECT CONFIGURATION
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"

NHANES_BASE_URL = (
    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/"
)

# Each file contains raw NHANES survey data.
# SEQN is the participant identifier used for integration.

DATASETS = {
    "DEMO_J": [
        "SEQN",
        "RIAGENDR",
        "RIDAGEYR",
        "RIDRETH3",
        "DMDEDUC2",
        "INDFMPIR",
    ],
    "PAQ_J": [
        "SEQN",
        "PAQ605",
        "PAQ620",
        "PAQ635",
        "PAQ650",
        "PAQ665",
        "PAD680",
    ],
    "SMQ_J": [
        "SEQN",
        "SMQ020",
        "SMQ040",
    ],
    "SLQ_J": [
        "SEQN",
        "SLD012",
        "SLD013",
    ],
    "DBQ_J": [
        "SEQN",
        "DBQ197",
        "DBQ700",
    ],
    "BMX_J": [
        "SEQN",
        "BMXBMI",
        "BMXWT",
        "BMXHT",
        "BMXWAIST",
    ],
    "HUQ_J": [
        "SEQN",
        "HUQ010",
    ],
}


# ---------------------------------------------------------
# STEP 1: DOWNLOAD RAW DATASETS
# ---------------------------------------------------------

def download_dataset(dataset_name: str) -> Path:
    """
    Download one raw NHANES XPT file if it is not already
    present in the project's data/raw directory.
    """

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    file_path = RAW_DIR / f"{dataset_name}.XPT"

    if file_path.exists() and file_path.stat().st_size > 0:
        return file_path

    url = f"{NHANES_BASE_URL}{dataset_name}.XPT"

    request = Request(
        url,
        headers={"User-Agent": "LifestyleLongevityKDDProject/1.0"},
    )

    try:
        with urlopen(request, timeout=90) as response:
            content = response.read()
    except Exception as exc:
        raise RuntimeError(
            f"Could not download {dataset_name}.XPT.\n"
            f"Check your internet connection and the official "
            f"NHANES dataset page.\nOriginal error: {exc}"
        ) from exc

    if not content:
        raise RuntimeError(f"The downloaded file {dataset_name}.XPT is empty.")

    file_path.write_bytes(content)

    return file_path


# ---------------------------------------------------------
# STEP 2: LOAD AND INTEGRATE RAW DATASETS
# ---------------------------------------------------------

def load_raw_datasets() -> pd.DataFrame:
    """
    Download, read, and integrate the seven NHANES datasets.

    Integration key: SEQN
    """

    integrated_df = None

    for dataset_name, expected_columns in DATASETS.items():

        file_path = download_dataset(dataset_name)

        try:
            raw_df = pd.read_sas(
                file_path,
                format="xport",
                encoding="utf-8",
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not read {file_path.name}: {exc}"
            ) from exc

        # Normalize column names.
        raw_df.columns = [
            str(column).upper() for column in raw_df.columns
        ]

        # Keep only the columns needed by this project.
        available_columns = [
            column
            for column in expected_columns
            if column in raw_df.columns
        ]

        if "SEQN" not in available_columns:
            raise ValueError(
                f"SEQN is missing from {dataset_name}.XPT"
            )

        raw_df = raw_df[available_columns].copy()

        # Remove duplicate participant records before merging.
        raw_df = raw_df.drop_duplicates(subset="SEQN")

        if integrated_df is None:
            integrated_df = raw_df
        else:
            integrated_df = integrated_df.merge(
                raw_df,
                on="SEQN",
                how="left",
                validate="one_to_one",
            )

    if integrated_df is None or integrated_df.empty:
        raise RuntimeError("No data was loaded from NHANES.")

    return integrated_df


# ---------------------------------------------------------
# STEP 3: DATA CLEANING
# ---------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean missing values and questionnaire response codes.

    NHANES uses special codes such as 7, 9, 77 and 99
    for refusal or don't-know responses in certain variables.

    These codes are handled only for the relevant variables.
    """

    df = df.copy()

    # Convert relevant columns to numeric.
    numeric_columns = [
        "SEQN",
        "RIAGENDR",
        "RIDAGEYR",
        "RIDRETH3",
        "DMDEDUC2",
        "INDFMPIR",
        "PAQ605",
        "PAQ620",
        "PAQ635",
        "PAQ650",
        "PAQ665",
        "PAD680",
        "SMQ020",
        "SMQ040",
        "SLD012",
        "SLD013",
        "DBQ197",
        "DBQ700",
        "BMXBMI",
        "BMXWT",
        "BMXHT",
        "BMXWAIST",
        "HUQ010",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    # Refused / don't know codes for relevant categorical variables.
    special_79_columns = [
        "DMDEDUC2",
        "PAQ605",
        "PAQ620",
        "PAQ635",
        "PAQ650",
        "PAQ665",
        "SMQ020",
        "SMQ040",
        "DBQ700",
        "HUQ010",
    ]

    for column in special_79_columns:
        if column in df.columns:
            df[column] = df[column].replace(
                {7: np.nan, 9: np.nan}
            )

    # Sleep duration questionnaire uses special 77/99 codes.
    for column in ["SLD012", "SLD013"]:
        if column in df.columns:
            df[column] = df[column].replace(
                {77: np.nan, 99: np.nan}
            )

    # Remove impossible or out-of-range values.
    if "RIDAGEYR" in df.columns:
        df.loc[
            (df["RIDAGEYR"] < 0) | (df["RIDAGEYR"] > 120),
            "RIDAGEYR",
        ] = np.nan

    if "BMXBMI" in df.columns:
        df.loc[
            (df["BMXBMI"] <= 0) | (df["BMXBMI"] > 100),
            "BMXBMI",
        ] = np.nan

    for column in ["SLD012", "SLD013"]:
        if column in df.columns:
            df.loc[
                (df[column] < 0) | (df[column] > 24),
                column,
            ] = np.nan

    if "PAD680" in df.columns:
        # PAD680 represents daily sedentary time in minutes.
        df.loc[
            (df["PAD680"] < 0) | (df["PAD680"] > 1440),
            "PAD680",
        ] = np.nan

    # Drop rows without participant identifiers.
    df = df.dropna(subset=["SEQN"])

    # Remove duplicate participants after integration.
    df = df.drop_duplicates(subset="SEQN")

    return df.reset_index(drop=True)


# ---------------------------------------------------------
# STEP 4: FEATURE ENGINEERING
# ---------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create meaningful features from the cleaned survey data.
    """

    df = df.copy()

    # Gender labels.
    if "RIAGENDR" in df.columns:
        df["gender"] = df["RIAGENDR"].map(
            {
                1: "Male",
                2: "Female",
            }
        )

    # Race / ethnicity labels.
    if "RIDRETH3" in df.columns:
        race_labels = {
            1: "Mexican American",
            2: "Other Hispanic",
            3: "Non-Hispanic White",
            4: "Non-Hispanic Black",
            6: "Non-Hispanic Asian",
            7: "Other or multiracial",
        }

        df["race_ethnicity"] = df["RIDRETH3"].map(
            race_labels
        )

    # Education labels.
    if "DMDEDUC2" in df.columns:
        education_labels = {
            1: "Less than 9th grade",
            2: "9th-11th grade",
            3: "High school graduate",
            4: "Some college or associate degree",
            5: "College graduate or above",
        }

        df["education"] = df["DMDEDUC2"].map(
            education_labels
        )

    # Body mass index categories.
    if "BMXBMI" in df.columns:
        df["bmi_group"] = pd.cut(
            df["BMXBMI"],
            bins=[0, 18.5, 25, 30, np.inf],
            labels=[
                "Underweight",
                "Normal",
                "Overweight",
                "Obese",
            ],
            right=False,
        ).astype("object")

    # Age categories.
    if "RIDAGEYR" in df.columns:
        df["age_group"] = pd.cut(
            df["RIDAGEYR"],
            bins=[0, 18, 30, 45, 60, np.inf],
            labels=[
                "Under 18",
                "18-29",
                "30-44",
                "45-59",
                "60+",
            ],
            right=False,
        ).astype("object")

    # Physical activity indicators.
    activity_columns = [
        "PAQ605",
        "PAQ620",
        "PAQ635",
        "PAQ650",
        "PAQ665",
    ]

    indicator_names = {
        "PAQ605": "vigorous_work",
        "PAQ620": "moderate_work",
        "PAQ635": "walking_transport",
        "PAQ650": "vigorous_recreation",
        "PAQ665": "moderate_recreation",
    }

    created_activity_columns = []

    for source_column, feature_name in indicator_names.items():
        if source_column in df.columns:
            df[feature_name] = df[source_column].map(
                {
                    1: 1,
                    2: 0,
                }
            )

            created_activity_columns.append(feature_name)

    if created_activity_columns:
        df["activity_score"] = df[
            created_activity_columns
        ].sum(axis=1, min_count=1)

        df["activity_level"] = np.select(
            [
                df["activity_score"].eq(0),
                df["activity_score"].between(1, 2),
                df["activity_score"].ge(3),
            ],
            [
                "Low",
                "Moderate",
                "Higher",
            ],
            default="Unknown",
        )

        df.loc[
            df["activity_score"].isna(),
            "activity_level",
        ] = np.nan

    # Smoking categories.
    if "SMQ020" in df.columns:
        df["ever_smoked_100_cigarettes"] = df[
            "SMQ020"
        ].map(
            {
                1: 1,
                2: 0,
            }
        )

    if "SMQ020" in df.columns and "SMQ040" in df.columns:

        df["current_smoker"] = np.nan

        # Never smoked 100 cigarettes.
        df.loc[
            df["SMQ020"] == 2,
            "current_smoker",
        ] = 0

        # Previously smoked 100 cigarettes:
        # 1 = every day, 2 = some days, 3 = not at all.
        df.loc[
            (df["SMQ020"] == 1)
            & (df["SMQ040"].isin([1, 2])),
            "current_smoker",
        ] = 1

        df.loc[
            (df["SMQ020"] == 1)
            & (df["SMQ040"] == 3),
            "current_smoker",
        ] = 0

        df["smoking_group"] = df[
            "current_smoker"
        ].map(
            {
                0: "Not currently smoking",
                1: "Currently smoking",
            }
        )

    # Sleep categories.
    if "SLD012" in df.columns:
        df["sleep_group"] = pd.cut(
            df["SLD012"],
            bins=[0, 6, 7, 9, 24],
            labels=[
                "Under 6 hours",
                "6 to under 7 hours",
                "7 to under 9 hours",
                "9 hours or more",
            ],
            right=False,
        ).astype("object")

    # Diet quality response.
    if "DBQ700" in df.columns:
        diet_labels = {
            1: "Excellent",
            2: "Very good",
            3: "Good",
            4: "Fair",
            5: "Poor",
        }

        df["diet_quality"] = df["DBQ700"].map(
            diet_labels
        )

    # General self-reported health.
    if "HUQ010" in df.columns:
        health_labels = {
            1: "Excellent",
            2: "Very good",
            3: "Good",
            4: "Fair",
            5: "Poor",
        }

        df["general_health"] = df["HUQ010"].map(
            health_labels
        )

        # Binary classification target.
        # This predicts reported health category, NOT lifespan.
        df["health_label"] = np.select(
            [
                df["HUQ010"].isin([1, 2, 3]),
                df["HUQ010"].isin([4, 5]),
            ],
            [
                "Good or better",
                "Fair or poor",
            ],
            default=None,
        )

        df["health_label"] = df[
            "health_label"
        ].replace("None", np.nan)

    return df


# ---------------------------------------------------------
# STEP 5: COMPLETE KDD DATA PREPARATION
# ---------------------------------------------------------

def load_data() -> pd.DataFrame:
    """
    Run the initial KDD stages:
    raw data collection -> integration -> cleaning
    -> feature engineering.
    """

    raw_df = load_raw_datasets()
    cleaned_df = clean_data(raw_df)
    final_df = engineer_features(cleaned_df)

    return final_df


# ---------------------------------------------------------
# STEP 6: APRIORI FREQUENT PATTERN MINING
# ---------------------------------------------------------

def run_apriori(
    df: pd.DataFrame,
    min_support: float = 0.05,
    min_confidence: float = 0.30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply Apriori to lifestyle and health categories.

    Returns:
        frequent_itemsets
        association_rules_table
    """

    from mlxtend.frequent_patterns import (
        apriori,
        association_rules,
    )
    from mlxtend.preprocessing import TransactionEncoder

    candidate_columns = [
        "age_group",
        "bmi_group",
        "sleep_group",
        "activity_level",
        "smoking_group",
        "diet_quality",
        "general_health",
    ]

    available_columns = [
        column
        for column in candidate_columns
        if column in df.columns
    ]

    if len(available_columns) < 2:
        raise ValueError(
            "Not enough categorical features are available "
            "for Apriori."
        )

    transactions_df = df[available_columns].copy()

    transactions = []

    for _, row in transactions_df.iterrows():
        items = []

        for column, value in row.items():
            if pd.notna(value) and str(value) != "Unknown":
                items.append(f"{column}={value}")

        transactions.append(items)

    # Exclude rows with no usable items.
    transactions = [
        transaction
        for transaction in transactions
        if transaction
    ]

    if not transactions:
        return pd.DataFrame(), pd.DataFrame()

    encoder = TransactionEncoder()
    encoded_array = encoder.fit(transactions).transform(
        transactions
    )

    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoder.columns_,
    )

    frequent_itemsets = apriori(
        encoded_df,
        min_support=min_support,
        use_colnames=True,
    )

    if frequent_itemsets.empty:
        return frequent_itemsets, pd.DataFrame()

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence,
    )

    if rules.empty:
        return frequent_itemsets, rules

    rules = rules.copy()

    rules["antecedents"] = rules[
        "antecedents"
    ].apply(lambda items: ", ".join(sorted(items)))

    rules["consequents"] = rules[
        "consequents"
    ].apply(lambda items: ", ".join(sorted(items)))

    rules = rules.sort_values(
        ["lift", "confidence"],
        ascending=False,
    )

    return frequent_itemsets, rules


# ---------------------------------------------------------
# STEP 7: K-MEANS CLUSTERING
# ---------------------------------------------------------

def run_clustering(
    df: pd.DataFrame,
    n_clusters: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, float | None]:
    """
    Group participants using numeric lifestyle and
    demographic features.

    Returns:
        clustered participant data
        cluster profile table
        silhouette score
    """

    candidate_features = [
        "RIDAGEYR",
        "INDFMPIR",
        "BMXBMI",
        "BMXWAIST",
        "SLD012",
        "SLD013",
        "PAD680",
        "vigorous_work",
        "moderate_work",
        "walking_transport",
        "vigorous_recreation",
        "moderate_recreation",
        "current_smoker",
    ]

    available_features = [
        column
        for column in candidate_features
        if column in df.columns
    ]

    if len(available_features) < 2:
        raise ValueError(
            "Not enough numeric features are available "
            "for clustering."
        )

    X = df[available_features].copy()

    # Remove columns that are entirely missing or constant.
    X = X.dropna(axis=1, how="all")

    variable_columns = [
        column
        for column in X.columns
        if X[column].nunique(dropna=True) > 1
    ]

    X = X[variable_columns]

    if X.shape[1] < 2:
        raise ValueError(
            "At least two useful numeric features are "
            "needed for clustering."
        )

    # Remove rows with no information across selected features.
    valid_rows = X.notna().any(axis=1)

    X = X.loc[valid_rows]

    if len(X) <= n_clusters:
        raise ValueError(
            "Not enough participants for the selected "
            "number of clusters."
        )

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    )

    cluster_labels = model.fit_predict(X_scaled)

    result_df = df.loc[X.index].copy()
    result_df["cluster"] = cluster_labels

    score = None

    if len(np.unique(cluster_labels)) > 1:
        score = float(
            silhouette_score(X_scaled, cluster_labels)
        )

    profile_features = [
        column
        for column in [
            "RIDAGEYR",
            "BMXBMI",
            "BMXWAIST",
            "SLD012",
            "SLD013",
            "PAD680",
            "INDFMPIR",
            "current_smoker",
            "activity_score",
        ]
        if column in result_df.columns
    ]

    profile = result_df.groupby(
        "cluster",
        observed=False,
    )[profile_features].mean(numeric_only=True)

    profile.insert(
        0,
        "participants",
        result_df.groupby(
            "cluster",
            observed=False,
        ).size(),
    )

    return result_df, profile, score


# ---------------------------------------------------------
# STEP 8: CLASSIFICATION
# ---------------------------------------------------------

def run_classification(
    df: pd.DataFrame,
) -> dict:
    """
    Train a Random Forest classifier to predict the
    self-reported health category.

    IMPORTANT:
    This is NOT a life expectancy prediction model.
    """

    if "health_label" not in df.columns:
        raise ValueError(
            "The health_label target is unavailable."
        )

    # Exclude direct health-response variables to avoid leakage.
    candidate_features = [
        "RIDAGEYR",
        "gender",
        "race_ethnicity",
        "education",
        "INDFMPIR",
        "BMXBMI",
        "BMXWAIST",
        "SLD012",
        "SLD013",
        "PAD680",
        "vigorous_work",
        "moderate_work",
        "walking_transport",
        "vigorous_recreation",
        "moderate_recreation",
        "current_smoker",
        "diet_quality",
    ]

    available_features = [
        column
        for column in candidate_features
        if column in df.columns
        and df[column].notna().any()
    ]

    if len(available_features) < 2:
        raise ValueError(
            "Not enough features are available "
            "for classification."
        )

    model_df = df[
        available_features + ["health_label"]
    ].copy()

    model_df = model_df.dropna(
        subset=["health_label"]
    )

    # The target is self-reported health.
    # Do not include HUQ010/general_health as model inputs.
    X = model_df[available_features]
    y = model_df["health_label"].astype(str)

    if y.nunique() < 2:
        raise ValueError(
            "Classification requires at least two target classes."
        )

    if len(model_df) < 30:
        raise ValueError(
            "Too few labelled participants for a meaningful "
            "train/test experiment."
        )

    numeric_features = [
        column
        for column in available_features
        if pd.api.types.is_numeric_dtype(X[column])
    ]

    categorical_features = [
        column
        for column in available_features
        if column not in numeric_features
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    transformers = []

    if numeric_features:
        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    # Stratify when every class has at least two records.
    class_counts = y.value_counts()

    stratify_target = (
        y if class_counts.min() >= 2 else None
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=stratify_target,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    report_dict = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report_dict).transpose()

    labels = sorted(y.unique())

    confusion_df = pd.DataFrame(
        confusion_matrix(
            y_test,
            predictions,
            labels=labels,
        ),
        index=labels,
        columns=labels,
    )

    return {
        "accuracy": float(accuracy),
        "report": report_df,
        "confusion_matrix": confusion_df,
        "model": model,
        "features": available_features,
        "test_rows": len(y_test),
        "train_rows": len(y_train),
    }