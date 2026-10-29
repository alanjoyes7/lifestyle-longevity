import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from kdd_pipeline import (
    load_data,
    run_apriori,
    run_classification,
    run_clustering,
)


# ---------------------------------------------------------
# HUMAN-READABLE DATA DICTIONARY / DISPLAY HELPERS
# ---------------------------------------------------------

COLUMN_LABELS = {
    "SEQN": "Participant ID",
    "RIAGENDR": "Gender",
    "RIDAGEYR": "Age (years)",
    "RIDRETH3": "Race / ethnicity",
    "DMDEDUC2": "Education level",
    "INDFMPIR": "Family income-to-poverty ratio",
    "PAQ605": "Vigorous work activity",
    "PAQ620": "Moderate work activity",
    "PAQ635": "Walking or bicycle transportation",
    "PAQ650": "Vigorous recreational activity",
    "PAQ665": "Moderate recreational activity",
    "PAD680": "Sedentary time (minutes/day)",
    "SMQ020": "Smoked at least 100 cigarettes in life",
    "SMQ040": "Current smoking frequency",
    "SLD012": "Weekday/workday sleep (hours)",
    "SLD013": "Weekend sleep (hours)",
    "DBQ197": "Meal away from home frequency",
    "DBQ700": "Perceived diet quality",
    "BMXBMI": "Body Mass Index (BMI)",
    "BMXWT": "Weight (kg)",
    "BMXHT": "Height (cm)",
    "BMXWAIST": "Waist circumference (cm)",
    "HUQ010": "Self-reported general health",
    "gender": "Gender",
    "race_ethnicity": "Race / ethnicity",
    "education": "Education level",
    "bmi_group": "BMI category",
    "age_group": "Age group",
    "vigorous_work": "Vigorous work activity (yes/no)",
    "moderate_work": "Moderate work activity (yes/no)",
    "walking_transport": "Walking/cycling for transport (yes/no)",
    "vigorous_recreation": "Vigorous recreational activity (yes/no)",
    "moderate_recreation": "Moderate recreational activity (yes/no)",
    "activity_score": "Physical activity score",
    "activity_level": "Physical activity level",
    "ever_smoked_100_cigarettes": "Ever smoked 100 cigarettes (yes/no)",
    "current_smoker": "Currently smoking (yes/no)",
    "smoking_group": "Smoking status",
    "sleep_group": "Sleep duration category",
    "diet_quality": "Perceived diet quality",
    "general_health": "Self-reported general health",
    "health_label": "Self-reported health category",
    "cluster": "Cluster",
    "support": "Support",
    "confidence": "Confidence",
    "lift": "Lift",
}

VALUE_LABELS = {
    "RIAGENDR": {1: "Male", 2: "Female"},
    "RIDRETH3": {
        1: "Mexican American", 2: "Other Hispanic",
        3: "Non-Hispanic White", 4: "Non-Hispanic Black",
        6: "Non-Hispanic Asian", 7: "Other or multiracial",
    },
    "DMDEDUC2": {
        1: "Less than 9th grade", 2: "9th–11th grade",
        3: "High school graduate", 4: "Some college/associate degree",
        5: "College graduate or above",
    },
    "PAQ605": {1: "Yes", 2: "No"},
    "PAQ620": {1: "Yes", 2: "No"},
    "PAQ635": {1: "Yes", 2: "No"},
    "PAQ650": {1: "Yes", 2: "No"},
    "PAQ665": {1: "Yes", 2: "No"},
    "SMQ020": {1: "Yes", 2: "No"},
    "SMQ040": {1: "Every day", 2: "Some days", 3: "Not at all"},
    "DBQ700": {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"},
    "HUQ010": {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"},
}

def display_frame(frame):
    """Create a display-friendly DataFrame with unique column names."""

    shown = frame.copy()

    # Convert coded values into readable labels where mappings exist.
    for column, value_labels in VALUE_LABELS.items():
        if column in shown.columns:
            original = shown[column]
            mapped = original.map(value_labels)
            shown[column] = mapped.where(mapped.notna(), original)

    # Convert technical column names into human-readable names.
    unique_labels = []
    label_counts = {}

    for column in shown.columns:
        label = COLUMN_LABELS.get(
            column,
            str(column).replace("_", " ").capitalize()
        )

        # Ensure every displayed column name is unique.
        base_label = label
        count = label_counts.get(base_label, 0) + 1

        while label in unique_labels:
            count += 1
            label = f"{base_label} ({count})"

        label_counts[base_label] = count
        unique_labels.append(label)

    shown.columns = unique_labels

    return shown

def friendly_name(column):
    return COLUMN_LABELS.get(column, str(column).replace("_", " ").capitalize())



# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Lifestyle and Longevity Data Mining",
    page_icon="📊",
    layout="wide",
)

st.title(
    "Lifestyle-Based Health and Longevity "
    "Pattern Discovery"
)

st.markdown(
    """
    **Project:** Lifestyle-Based Health and Longevity Pattern
    Discovery Using Integrated Data Mining Techniques

    This application demonstrates the Knowledge Discovery in
    Databases (KDD) process using integrated public NHANES data.
    """
)

st.info(
    "This project explores associations between lifestyle and "
    "self-reported health. It does not predict actual lifespan "
    "or establish cause and effect."
)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_project_data():
    return load_data()


with st.spinner(
    "Loading and integrating raw NHANES datasets..."
):
    try:
        df = get_project_data()
    except Exception as exc:
        st.error(
            "The dataset could not be loaded. "
            "Check your internet connection and the project "
            "dependencies."
        )
        st.exception(exc)
        st.stop()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.header("Project Information")

st.sidebar.write(
    "Dataset: NHANES 2017–2018"
)

st.sidebar.write(
    "Data integration key: SEQN"
)

st.sidebar.metric(
    "Participants",
    f"{len(df):,}",
)

st.sidebar.metric(
    "Integrated variables",
    f"{len(df.columns):,}",
)

st.sidebar.caption(
    "Use the tabs to explore the different KDD stages."
)


# ---------------------------------------------------------
# NAVIGATION
# ---------------------------------------------------------

tabs = st.tabs(
    [
        "Overview",
        "Data Quality",
        "Explore Data",
        "Apriori",
        "K-Means",
        "Classification",
        "Health Predictor",
    ]
)


# ---------------------------------------------------------
# TAB 1: OVERVIEW
# ---------------------------------------------------------

with tabs[0]:

    st.header("KDD Project Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Participants",
        f"{len(df):,}",
    )

    col2.metric(
        "Variables",
        f"{len(df.columns):,}",
    )

    col3.metric(
        "Duplicate participant IDs",
        f"{df['SEQN'].duplicated().sum():,}",
    )

    st.subheader("Knowledge Discovery Pipeline")

    pipeline_steps = [
        "1. Data collection",
        "2. Data integration",
        "3. Data cleaning",
        "4. Data transformation",
        "5. Data mining",
        "6. Pattern evaluation",
        "7. Visualization and interpretation",
    ]

    for step in pipeline_steps:
        st.write(f"✓ {step}")

    st.subheader("Integrated Dataset Preview")

    st.dataframe(
        display_frame(df.head(20)),
        use_container_width=True,
    )

    st.download_button(
        label="Download integrated dataset as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="integrated_lifestyle_health_data.csv",
        mime="text/csv",
    )

    st.caption(
        "The CSV contains integrated survey records. "
        "Review the codebook before interpreting coded variables."
    )


# ---------------------------------------------------------
# TAB 2: DATA QUALITY
# ---------------------------------------------------------

with tabs[1]:

    st.header("Data Quality and Cleaning")

    st.write(
        "This section shows missing values, data types, "
        "and duplicate participant records."
    )

    quality_df = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": [
                str(df[column].dtype)
                for column in df.columns
            ],
            "Missing Values": [
                int(df[column].isna().sum())
                for column in df.columns
            ],
            "Missing Percentage": [
                round(
                    df[column].isna().mean() * 100,
                    2,
                )
                for column in df.columns
            ],
            "Unique Values": [
                int(df[column].nunique(dropna=True))
                for column in df.columns
            ],
        }
    )

    st.dataframe(
        display_frame(quality_df),
        use_container_width=True,
    )

    st.subheader("Missing Values")

    missing_df = quality_df[
        quality_df["Missing Values"] > 0
    ].sort_values(
        "Missing Percentage",
        ascending=False,
    )

    if missing_df.empty:
        st.success(
            "No missing values were detected."
        )
    else:
        st.dataframe(
            display_frame(missing_df),
            use_container_width=True,
        )

    st.subheader("Data Types")

    st.dataframe(
        df.dtypes.astype(str).rename("Data Type"),
        use_container_width=True,
    )

    st.write(
        "**Duplicate participant IDs:**",
        int(df["SEQN"].duplicated().sum()),
    )


# ---------------------------------------------------------
# TAB 3: EXPLORE DATA
# ---------------------------------------------------------

with tabs[2]:

    st.header("Exploratory Data Analysis")

    numeric_columns = [
        column
        for column in df.select_dtypes(
            include="number"
        ).columns
        if df[column].notna().any()
    ]

    if numeric_columns:

        selected_column = st.selectbox(
            "Select a numeric variable",
            numeric_columns,
            format_func=friendly_name,
        )

        chart_data = df[
            selected_column
        ].dropna()

        st.subheader(
            f"Distribution of {friendly_name(selected_column)}"
        )

        fig, ax = plt.subplots()

        ax.hist(
            chart_data,
            bins=25,
            edgecolor="black",
        )

        ax.set_xlabel(friendly_name(selected_column))
        ax.set_ylabel("Number of participants")
        ax.set_title(
            f"Distribution of {friendly_name(selected_column)}"
        )

        st.pyplot(fig)

        plt.close(fig)

        st.subheader("Descriptive Statistics")

        st.dataframe(
            display_frame(chart_data.describe().to_frame(
                name="Value"
            )),
            use_container_width=True,
        )

    else:
        st.warning(
            "No numeric variables are available."
        )

    categorical_columns = [
        column
        for column in [
            "gender",
            "age_group",
            "bmi_group",
            "activity_level",
            "sleep_group",
            "smoking_group",
            "diet_quality",
            "general_health",
        ]
        if column in df.columns
    ]

    if categorical_columns:

        st.subheader("Categorical Distribution")

        selected_category = st.selectbox(
            "Select a category",
            categorical_columns,
            format_func=friendly_name,
            key="category_distribution",
        )

        counts = (
            df[selected_category]
            .value_counts(dropna=False)
            .head(15)
        )

        st.bar_chart(counts)


# ---------------------------------------------------------
# TAB 4: APRIORI
# ---------------------------------------------------------

with tabs[3]:

    st.header("Frequent Pattern Mining — Apriori")

    st.write(
        """
        Apriori identifies frequently occurring combinations
        of lifestyle and health categories.

        - **Support:** How frequently an item combination occurs.
        - **Confidence:** How often the consequent occurs when
          the antecedent occurs.
        - **Lift:** How strongly the items are associated relative
          to their independent occurrence.
        """
    )

    support = st.slider(
        "Minimum support",
        min_value=0.01,
        max_value=0.50,
        value=0.05,
        step=0.01,
    )

    confidence = st.slider(
        "Minimum confidence",
        min_value=0.10,
        max_value=1.00,
        value=0.30,
        step=0.05,
    )

    if st.button(
        "Run Apriori",
        key="run_apriori",
    ):

        try:
            with st.spinner(
                "Mining frequent patterns..."
            ):

                itemsets, rules = run_apriori(
                    df,
                    min_support=support,
                    min_confidence=confidence,
                )

            st.subheader("Frequent Itemsets")

            if itemsets.empty:
                st.warning(
                    "No frequent itemsets were found. "
                    "Try lowering minimum support."
                )
            else:
                display_itemsets = itemsets.copy()

                display_itemsets["itemsets"] = (
                    display_itemsets["itemsets"]
                    .apply(
                        lambda items: ", ".join(
                            sorted(items)
                        )
                    )
                )

                st.dataframe(
                    display_itemsets.sort_values(
                        "support",
                        ascending=False,
                    ),
                    use_container_width=True,
                )

            st.subheader("Association Rules")

            if rules.empty:
                st.warning(
                    "No association rules met the selected "
                    "confidence threshold."
                )
            else:
                display_rules = rules[
                    [
                        "antecedents",
                        "consequents",
                        "support",
                        "confidence",
                        "lift",
                    ]
                ].copy()

                st.dataframe(
                    display_rules,
                    use_container_width=True,
                )

                st.download_button(
                    "Download association rules",
                    data=display_rules.to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name="apriori_association_rules.csv",
                    mime="text/csv",
                )

                st.caption(
                    "Association rules describe co-occurrence, "
                    "not causation."
                )

        except Exception as exc:
            st.error(
                "Apriori could not be completed."
            )
            st.exception(exc)


# ---------------------------------------------------------
# TAB 5: K-MEANS CLUSTERING
# ---------------------------------------------------------

with tabs[4]:

    st.header("Clustering — K-Means")

    st.write(
        """
        K-Means groups participants according to similarities
        in selected numeric lifestyle and demographic variables.

        Cluster labels are mathematical group identifiers.
        They do not automatically represent good or bad health.
        """
    )

    n_clusters = st.slider(
        "Number of clusters (K)",
        min_value=2,
        max_value=6,
        value=3,
        step=1,
    )

    if st.button(
        "Run K-Means Clustering",
        key="run_clustering",
    ):

        try:
            with st.spinner(
                "Preparing features and running K-Means..."
            ):

                clustered_df, profile, score = run_clustering(
                    df,
                    n_clusters=n_clusters,
                )

            if score is not None:
                st.metric(
                    "Silhouette Score",
                    f"{score:.3f}",
                )

                st.caption(
                    "Silhouette scores range approximately from "
                    "-1 to 1. Higher values generally indicate "
                    "better-separated clusters."
                )

            st.subheader("Cluster Profiles")

            st.dataframe(
                profile,
                use_container_width=True,
            )

            st.subheader("Participants by Cluster")

            cluster_counts = (
                clustered_df["cluster"]
                .value_counts()
                .sort_index()
            )

            st.bar_chart(cluster_counts)

            st.subheader("Clustered Data Preview")

            st.dataframe(
                clustered_df.head(30),
                use_container_width=True,
            )

            st.download_button(
                "Download clustered data",
                data=clustered_df.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="kmeans_clustered_data.csv",
                mime="text/csv",
            )

        except Exception as exc:
            st.error(
                "Clustering could not be completed."
            )
            st.exception(exc)


# ---------------------------------------------------------
# TAB 6: CLASSIFICATION
# ---------------------------------------------------------

with tabs[5]:

    st.header("Classification — Random Forest")

    st.write(
        """
        This experiment predicts whether a participant's
        self-reported health falls into the "Good or better"
        or "Fair or poor" category.

        The target comes from the NHANES general-health
        questionnaire. This is not an estimate of lifespan.
        """
    )

    st.warning(
        "The reported accuracy is an experimental result for "
        "this dataset and split. It is not a medical diagnosis "
        "or proof that lifestyle causes a particular outcome."
    )

    if st.button(
        "Train and Evaluate Model",
        key="run_classification",
    ):

        try:
            with st.spinner(
                "Training the Random Forest classifier..."
            ):

                result = run_classification(df)

            st.subheader("Model Evaluation")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Accuracy",
                f"{result['accuracy']:.2%}",
            )

            col2.metric(
                "Training Records",
                f"{result['train_rows']:,}",
            )

            col3.metric(
                "Testing Records",
                f"{result['test_rows']:,}",
            )

            st.subheader("Classification Report")

            st.dataframe(
                result["report"],
                use_container_width=True,
            )

            st.subheader("Confusion Matrix")

            st.dataframe(
                result["confusion_matrix"],
                use_container_width=True,
            )

            st.write(
                "**Input features:**",
                ", ".join(friendly_name(name) for name in result["features"]),
            )

        except Exception as exc:
            st.error(
                "Classification could not be completed."
            )
            st.exception(exc)



# ---------------------------------------------------------
# TAB 7: INTERACTIVE HEALTH CONDITION PREDICTOR
# ---------------------------------------------------------

with tabs[6]:
    st.header("Health Condition Predictor")
    st.write(
        "Enter lifestyle and demographic information to estimate the "
        "participant's self-reported health category using the project's "
        "Random Forest model."
    )
    st.warning(
        "Educational data-mining demonstration only. This is not a medical "
        "diagnosis, a clinical risk score, or a prediction of life expectancy. "
        "The model was trained on U.S. NHANES 2017–2018 survey data."
    )

    st.subheader("Enter participant details")
    st.caption(
        "Inputs are limited to features used by the existing classification "
        "pipeline. The model target is 'Good or better' versus 'Fair or poor' "
        "self-reported health."
    )

    # The same feature list and preprocessing returned by run_classification()
    # are used for prediction, so input columns and encoding remain consistent.
    with st.form("health_prediction_form"):
        form_values = {}
        input_features = [
            "RIDAGEYR", "gender", "race_ethnicity", "education",
            "INDFMPIR", "BMXBMI", "BMXWAIST", "SLD012", "SLD013",
            "PAD680", "vigorous_work", "moderate_work",
            "walking_transport", "vigorous_recreation",
            "moderate_recreation", "current_smoker", "diet_quality",
        ]
        available_input_features = [
            feature for feature in input_features
            if feature in df.columns and df[feature].notna().any()
        ]
        left_col, right_col = st.columns(2)

        # Friendly labels with understandable values or ranges.
        input_labels = {
            "RIDAGEYR": "Age (years) (0–80)",
            "gender": "Gender",
            "race_ethnicity": "Race / ethnicity",
            "education": "Education level",
            "INDFMPIR": "Family income-to-poverty ratio (0–5)",
            "BMXBMI": "Body Mass Index (BMI) (kg/m²)",
            "BMXWAIST": "Waist circumference (cm)",
            "SLD012": "Weekday/workday sleep (hours)",
            "SLD013": "Weekend sleep (hours)",
            "PAD680": "Sedentary time (minutes/day)",
            "vigorous_work": "Vigorous work activity (Yes=1, No=0)",
            "moderate_work": "Moderate work activity (Yes=1, No=0)",
            "walking_transport": "Walking/cycling for transport (Yes=1, No=0)",
            "vigorous_recreation": "Vigorous recreational activity (Yes=1, No=0)",
            "moderate_recreation": "Moderate recreational activity (Yes=1, No=0)",
            "current_smoker": "Currently smoking (Yes/No)",
            "diet_quality": "Perceived diet quality (Excellent–Poor)",
        }

        for index, feature in enumerate(available_input_features):
            target_col = left_col if index % 2 == 0 else right_col
            series = df[feature].dropna()

            if series.empty:
                continue

            label = input_labels.get(feature, friendly_name(feature))

            with target_col:
                if pd.api.types.is_numeric_dtype(series):
                    low = float(series.quantile(0.01))
                    high = float(series.quantile(0.99))
                    default = float(series.median())

                    if low == high:
                        low = float(series.min())
                        high = float(series.max())

                    if low == high:
                        form_values[feature] = st.number_input(
                            label,
                            value=default,
                            key=f"predict_{feature}",
                        )
                    else:
                        form_values[feature] = st.number_input(
                            label,
                            min_value=float(low),
                            max_value=float(high),
                            value=float(min(max(default, low), high)),
                            step=1.0 if feature in {"RIDAGEYR", "PAD680"} else 0.1,
                            key=f"predict_{feature}",
                        )
                else:
                    options = sorted(series.unique().tolist(), key=str)

                    mode_values = series.mode()
                    default_value = (
                        mode_values.iloc[0]
                        if not mode_values.empty
                        else options[0]
                    )

                    default_index = (
                        options.index(default_value)
                        if default_value in options
                        else 0
                    )

                    form_values[feature] = st.selectbox(
                        label,
                        options=options,
                        index=default_index,
                        format_func=lambda value: str(value),
                        key=f"predict_{feature}",
                    )
        submitted = st.form_submit_button(
            "Predict Health Condition",
            type="primary",
        )

    if submitted:
        try:
            with st.spinner("Training the model and generating the prediction..."):
                model_result = run_classification(df)

            # Keep only the features actually used by this fitted model.
            prediction_row = {
                feature: form_values[feature]
                for feature in model_result["features"]
                if feature in form_values
            }

            missing_features = [
                feature for feature in model_result["features"]
                if feature not in prediction_row
            ]
            if missing_features:
                st.error(
                    "Some model inputs are unavailable in the form: "
                    + ", ".join(friendly_name(feature) for feature in missing_features)
                )
            else:
                prediction_input = pd.DataFrame([prediction_row])
                predicted_label = model_result["model"].predict(prediction_input)[0]

                st.subheader("Prediction Result")
                if predicted_label == "Good or better":
                    st.success(f"Predicted self-reported health category: **{predicted_label}**")
                else:
                    st.warning(f"Predicted self-reported health category: **{predicted_label}**")

                if hasattr(model_result["model"], "predict_proba"):
                    probabilities = model_result["model"].predict_proba(prediction_input)[0]
                    classes = model_result["model"].named_steps["classifier"].classes_
                    probability_df = pd.DataFrame({
                        "Health category": classes,
                        "Model probability": probabilities,
                    })
                    probability_df["Model probability"] = (
                        probability_df["Model probability"] * 100
                    ).round(1).astype(str) + "%"
                    st.write("**Model probabilities**")
                    st.dataframe(probability_df, use_container_width=True, hide_index=True)
                    st.caption(
                        "These are model-estimated class probabilities, not medically "
                        "calibrated probabilities or a personal prognosis."
                    )

                st.write("**Summary of entered information**")
                summary = pd.DataFrame({
                    "Feature": [friendly_name(feature) for feature in prediction_row],
                    "Entered value": list(prediction_row.values()),
                })
                st.dataframe(summary, use_container_width=True, hide_index=True)
                st.caption(
                    "The prediction estimates a self-reported survey category. "
                    "It does not diagnose disease or predict lifespan."
                )
        except Exception as exc:
            st.error("The health condition prediction could not be completed.")
            st.exception(exc)


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "Academic demonstration of the KDD process. "
    "Dataset: U.S. NHANES 2017–2018. "
    "Interpret findings cautiously; observational associations "
    "do not establish causation."
)