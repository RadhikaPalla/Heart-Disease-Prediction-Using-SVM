import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    classification_report
)

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Heart Disease ML Dashboard",
    page_icon="❤️",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

.hero {
    padding: 1.5rem;
    border-radius: 18px;
    background: linear-gradient(120deg, #153e75, #3977c3);
    margin-bottom: 1.2rem;
}

.hero h1 {
    color: white !important;
    margin: 0;
    font-size: 2.2rem;
}

.hero p {
    color: #eef5ff !important;
}

div[data-testid="stMetric"] {
    background-color: #172554 !important;
    border: 1px solid #3b5998 !important;
    padding: 18px !important;
    border-radius: 14px !important;
}

div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] p {
    color: #dbeafe !important;
}

div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] div,
div[data-testid="stMetricValue"] p {
    color: white !important;
    font-weight: 700 !important;
}

button[data-baseweb="tab"] {
    color: #dbeafe !important;
}

.stButton button,
.stFormSubmitButton button {
    border-radius: 10px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================

st.markdown("""
<div class="hero">
    <h1>❤️ Heart Disease Prediction System</h1>
    <p>Machine Learning Dashboard | SVM, Logistic Regression and KNN</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# EXPECTED FEATURES
# ==========================================

FEATURES = [
    "age",
    "sex",
    "chest pain type",
    "resting bp s",
    "cholesterol",
    "fasting blood sugar",
    "resting ecg",
    "max heart rate",
    "exercise angina",
    "oldpeak",
    "ST slope"
]

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:
    st.header("⚙️ Dashboard Settings")

    uploaded_file = st.file_uploader(
        "Upload your ML Dataset.csv",
        type=["csv"]
    )

    st.divider()

    st.markdown("### Target Values")
    st.write("0 = No Heart Disease")
    st.write("1 = Heart Disease")

    st.info(
        "This is an educational ML project. "
        "Predictions are not a medical diagnosis."
    )

# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def load_csv(file_bytes):
    return pd.read_csv(BytesIO(file_bytes))


# ==========================================
# TRAIN MODELS
# ==========================================

@st.cache_resource
def train_models(dataframe):

    df = dataframe.copy()

    # Check target column
    if "target" not in df.columns:
        raise ValueError(
            "Dataset must contain a column named 'target'."
        )

    # Check feature columns
    missing_features = [
        col for col in FEATURES if col not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing feature columns: {missing_features}"
        )

    # Select the expected features in the correct order
    X = df[FEATURES].copy()

    # Convert values to numeric
    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df["target"], errors="coerce")

    # Remove rows with missing or invalid values
    valid_rows = X.notna().all(axis=1) & y.notna()

    X = X.loc[valid_rows].copy()
    y = y.loc[valid_rows].astype(int).copy()

    df = df.loc[valid_rows].copy()
    df["target"] = y

    if len(df) < 10:
        raise ValueError(
            "Not enough valid dataset rows to train the models."
        )

    if set(y.unique()) != {0, 1}:
        raise ValueError(
            "Target must contain both 0 and 1 values."
        )

    # Check class counts
    class_counts = y.value_counts()

    if class_counts.min() < 2:
        raise ValueError(
            "At least two rows are required for each target class."
        )

    # Split dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Scale using training data only
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Models
    models = {
        "SVM": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=42
        ),

        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=5
        )
    }

    results = {}

    # Train and evaluate each model
    for name, model in models.items():

        model.fit(X_train_scaled, y_train)

        predictions = model.predict(X_test_scaled)

        probabilities = model.predict_proba(
            X_test_scaled
        )[:, 1]

        results[name] = {
            "model": model,
            "pred": predictions,
            "prob": probabilities,

            "accuracy": accuracy_score(
                y_test, predictions
            ),

            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),

            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),

            "f1": f1_score(
                y_test, predictions, zero_division=0
            )
        }

    # SVM evaluation
    svm_result = results["SVM"]

    svm_result["cm"] = confusion_matrix(
        y_test,
        svm_result["pred"],
        labels=[0, 1]
    )

    svm_result["roc_auc"] = roc_auc_score(
        y_test,
        svm_result["prob"]
    )

    svm_result["roc"] = roc_curve(
        y_test,
        svm_result["prob"]
    )

    svm_result["report"] = classification_report(
        y_test,
        svm_result["pred"],
        labels=[0, 1],
        target_names=[
            "No Heart Disease",
            "Heart Disease"
        ],
        output_dict=True,
        zero_division=0
    )

    return (
        df,
        X,
        y,
        X_test,
        y_test,
        scaler,
        results
    )


# ==========================================
# UPLOAD SCREEN
# ==========================================

if uploaded_file is None:

    st.info(
        "👈 Upload your ML Dataset.csv from the sidebar."
    )

    st.subheader("Dashboard Features")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        ### 📊 Model Comparison
        - Accuracy
        - Precision
        - Recall
        - F1-Score
        """)

    with c2:
        st.markdown("""
        ### 🎯 SVM Evaluation
        - Confusion Matrix
        - ROC Curve
        - ROC-AUC
        - Actual vs Predicted
        """)

    with c3:
        st.markdown("""
        ### 🩺 Patient Prediction
        - Enter 11 patient details
        - Get SVM prediction
        - View class probabilities
        """)

    st.stop()


# ==========================================
# LOAD AND TRAIN
# ==========================================

try:

    df, X, y, X_test, y_test, scaler, results = train_models(
        load_csv(uploaded_file.getvalue())
    )

except Exception as e:

    st.error(f"Dataset or model error: {e}")
    st.stop()


svm_result = results["SVM"]

st.caption(
    f"Dataset: {len(df):,} valid rows | "
    f"{len(FEATURES)} features | "
    "80% training / 20% testing"
)

# ==========================================
# ACTUAL TARGET DISTRIBUTION
# ==========================================

st.subheader("📌 Actual Dataset Target Distribution")

target_counts = y.value_counts().reindex(
    [0, 1], fill_value=0
)

c1, c2, c3 = st.columns(3)

c1.metric(
    "Total Valid Rows",
    f"{len(df):,}"
)

c2.metric(
    "No Heart Disease (0)",
    f"{target_counts[0]:,}"
)

c3.metric(
    "Heart Disease (1)",
    f"{target_counts[1]:,}"
)

st.caption(
    "These counts show the actual labels in the uploaded dataset."
)

# ==========================================
# MODEL PERFORMANCE OVERVIEW
# ==========================================

st.subheader("📊 Model Performance Overview")

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "SVM Accuracy",
    f"{svm_result['accuracy'] * 100:.2f}%"
)

m2.metric(
    "SVM Precision",
    f"{svm_result['precision'] * 100:.2f}%"
)

m3.metric(
    "SVM Recall",
    f"{svm_result['recall'] * 100:.2f}%"
)

m4.metric(
    "SVM F1-Score",
    f"{svm_result['f1'] * 100:.2f}%"
)

# ==========================================
# TABS
# ==========================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Algorithm Comparison",
    "🎯 SVM Evaluation",
    "🩺 Patient Prediction",
    "📄 Dataset Details"
])

# ==========================================
# TAB 1: ALGORITHM COMPARISON
# ==========================================

with tab1:

    st.subheader("Algorithm Performance Comparison")

    comparison = pd.DataFrame([
        {
            "Algorithm": name,
            "Accuracy (%)": values["accuracy"] * 100,
            "Precision (%)": values["precision"] * 100,
            "Recall (%)": values["recall"] * 100,
            "F1-Score (%)": values["f1"] * 100
        }
        for name, values in results.items()
    ])

    st.dataframe(
        comparison.style.format({
            "Accuracy (%)": "{:.2f}",
            "Precision (%)": "{:.2f}",
            "Recall (%)": "{:.2f}",
            "F1-Score (%)": "{:.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    left, right = st.columns(2)

    with left:

        fig, ax = plt.subplots(figsize=(7, 4))

        bars = ax.bar(
            comparison["Algorithm"],
            comparison["Accuracy (%)"]
        )

        ax.set_title("Algorithm Accuracy")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.tick_params(axis="x", rotation=12)

        ax.bar_label(bars, fmt="%.2f%%", padding=3)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:

        metric_names = [
            "Precision (%)",
            "Recall (%)",
            "F1-Score (%)"
        ]

        x = np.arange(len(metric_names))
        width = 0.24

        fig, ax = plt.subplots(figsize=(7, 4))

        for i, row in comparison.iterrows():

            values_to_plot = [
                row[metric] for metric in metric_names
            ]

            ax.bar(
                x + (i - 1) * width,
                values_to_plot,
                width,
                label=row["Algorithm"]
            )

        ax.set_xticks(x)
        ax.set_xticklabels([
            "Precision",
            "Recall",
            "F1-Score"
        ])

        ax.set_ylim(0, 100)
        ax.set_ylabel("Score (%)")
        ax.set_title("Precision, Recall and F1-Score")
        ax.legend()

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ==========================================
# TAB 2: SVM EVALUATION
# ==========================================

with tab2:

    st.subheader("🎯 SVM Confusion Matrix")

    left, right = st.columns(2)

    with left:

        fig, ax = plt.subplots(figsize=(5.5, 4.5))

        disp = ConfusionMatrixDisplay(
            confusion_matrix=svm_result["cm"],
            display_labels=[
                "No Heart Disease",
                "Heart Disease"
            ]
        )

        disp.plot(
            ax=ax,
            colorbar=False,
            cmap="Blues"
        )

        ax.set_title("SVM Confusion Matrix")

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:

        fpr, tpr, thresholds = svm_result["roc"]

        fig, ax = plt.subplots(figsize=(5.5, 4.5))

        ax.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"SVM (AUC = {svm_result['roc_auc']:.3f})"
        )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            label="Random baseline"
        )

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.metric(
            "SVM ROC-AUC",
            f"{svm_result['roc_auc']:.4f}"
        )

    # Prediction distribution
    st.subheader("SVM Prediction Distribution")

    pred_counts = pd.Series(
        svm_result["pred"]
    ).value_counts().reindex(
        [0, 1], fill_value=0
    )

    fig, ax = plt.subplots(figsize=(7, 3.5))

    bars = ax.bar(
        ["No Heart Disease", "Heart Disease"],
        pred_counts.values
    )

    ax.set_ylabel("Number of test rows")
    ax.set_title("SVM Predicted Classes")
    ax.bar_label(bars, padding=3)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Classification report
    st.subheader("SVM Classification Report")

    report_df = pd.DataFrame(
        svm_result["report"]
    ).transpose()

    st.dataframe(
        report_df.style.format(precision=3),
        use_container_width=True
    )

    # Actual vs predicted
    st.subheader("Actual vs Predicted - SVM")

    actual_predicted = pd.DataFrame({
        "Actual Target": y_test.to_numpy(),
        "Predicted Target": svm_result["pred"],
        "Predicted Probability (Target 1)": svm_result["prob"]
    })

    actual_predicted["Result"] = np.where(
        actual_predicted["Actual Target"]
        == actual_predicted["Predicted Target"],
        "Correct",
        "Incorrect"
    )

    st.dataframe(
        actual_predicted,
        use_container_width=True,
        hide_index=True
    )

    wrong_predictions = actual_predicted[
        actual_predicted["Actual Target"]
        != actual_predicted["Predicted Target"]
    ]

    st.subheader("Incorrect SVM Predictions")

    if wrong_predictions.empty:

        st.success(
            "SVM predicted every row in this test split correctly."
        )

    else:

        st.warning(
            f"{len(wrong_predictions)} test rows were misclassified."
        )

        st.dataframe(
            wrong_predictions,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "These are test-set mistakes, not proof that the "
            "dataset's actual target labels are wrong."
        )


# ==========================================
# TAB 3: PATIENT PREDICTION
# ==========================================

with tab3:

    st.subheader("🩺 Enter Patient Details")

    st.warning(
        "Educational demonstration only. "
        "This is not a medical diagnosis."
    )

    feature_help = {
        "age": (
            "Age",
            1, 100, 50, 1
        ),

        "sex": (
            "Sex (0 or 1 as encoded in dataset)",
            0, 1, 1, 1
        ),

        "chest pain type": (
            "Chest Pain Type (1–4)",
            1, 4, 2, 1
        ),

        "resting bp s": (
            "Resting Blood Pressure",
            50, 250, 120, 1
        ),

        "cholesterol": (
            "Cholesterol",
            50, 700, 200, 1
        ),

        "fasting blood sugar": (
            "Fasting Blood Sugar (0/1)",
            0, 1, 0, 1
        ),

        "resting ecg": (
            "Resting ECG (0–2)",
            0, 2, 0, 1
        ),

        "max heart rate": (
            "Maximum Heart Rate",
            50, 250, 150, 1
        ),

        "exercise angina": (
            "Exercise-Induced Angina (0/1)",
            0, 1, 0, 1
        ),

        "oldpeak": (
            "Oldpeak",
            0.0, 10.0, 1.0, 0.1
        ),

        "ST slope": (
            "ST Slope (check dataset encoding)",
            0, 3, 1, 1
        )
    }

    values = {}

    with st.form("patient_form"):

        cols = st.columns(3)

        for idx, feature in enumerate(FEATURES):

            label, minimum, maximum, default, step = (
                feature_help[feature]
            )

            with cols[idx % 3]:

                if isinstance(default, float):

                    values[feature] = st.number_input(
                        label,
                        min_value=float(minimum),
                        max_value=float(maximum),
                        value=None,
                        step=float(step),
                        key=f"patient_{feature}"
                    )

                else:

                    values[feature] = st.number_input(
                        label,
                        min_value=int(minimum),
                        max_value=int(maximum),
                        value=None,
                        step=int(step),
                        key=f"patient_{feature}"
                    )

        submitted = st.form_submit_button(
            "🔍 Predict Heart Disease",
            use_container_width=True
        )

    if submitted:

        if any(value is None for value in values.values()):

            st.warning(
                "Please enter all 11 patient details before prediction."
            )

        else:

            patient_df = pd.DataFrame(
                [[values[feature] for feature in FEATURES]],
                columns=FEATURES
            )

            patient_scaled = scaler.transform(patient_df)

            # SVM prediction
            model = svm_result["model"]

            prediction = int(
                model.predict(patient_scaled)[0]
            )

            probabilities = model.predict_proba(
                patient_scaled
            )[0]

            class_index = list(
                model.classes_
            ).index(1)

            probability_1 = probabilities[class_index]

            probability_0 = 1 - probability_1

            st.subheader("Prediction Result")

            if prediction == 1:

                st.error(
                    "⚠️ Model Prediction: Heart Disease (1)"
                )

                st.warning(
                    "Please consult a qualified healthcare professional "
                    "for medical assessment."
                )

            else:

                st.success(
                    "Model Prediction: No Heart Disease (0)"
                )

                st.info(
                    "A negative model prediction does not rule out "
                    "heart disease."
                )

            p1, p2 = st.columns(2)

            p1.metric(
                "Probability: No Heart Disease",
                f"{probability_0 * 100:.2f}%"
            )

            p2.metric(
                "Probability: Heart Disease",
                f"{probability_1 * 100:.2f}%"
            )

            st.caption(
                "Probabilities are estimates from the trained model; "
                "they are not clinical risk estimates."
            )

            # Compare all three models for this patient
            st.subheader("Compare All Three Models")

            patient_predictions = []

            for name, model_result in results.items():

                patient_prediction = int(
                    model_result["model"].predict(
                        patient_scaled
                    )[0]
                )

                patient_predictions.append({
                    "Algorithm": name,
                    "Predicted Target": patient_prediction,
                    "Prediction": (
                        "Heart Disease"
                        if patient_prediction == 1
                        else "No Heart Disease"
                    )
                })

            st.dataframe(
                pd.DataFrame(patient_predictions),
                use_container_width=True,
                hide_index=True
            )


# ==========================================
# TAB 4: DATASET DETAILS
# ==========================================

with tab4:

    st.subheader("📄 Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Dataset Rows",
        f"{len(df):,}"
    )

    c2.metric(
        "Input Features",
        str(len(FEATURES))
    )

    c3.metric(
        "Missing Values After Cleaning",
        str(int(df[FEATURES + ["target"]].isnull().sum().sum()))
    )

    st.subheader("Target Distribution")

    target_counts = y.value_counts().reindex(
        [0, 1], fill_value=0
    )

    fig, ax = plt.subplots(figsize=(6, 3.5))

    bars = ax.bar(
        ["No Heart Disease (0)", "Heart Disease (1)"],
        target_counts.values
    )

    ax.set_ylabel("Number of rows")
    ax.set_title("Actual Dataset Target Distribution")
    ax.bar_label(bars, padding=3)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Feature Column Order")

    st.dataframe(
        pd.DataFrame({
            "Position": range(1, len(FEATURES) + 1),
            "Feature": FEATURES
        }),
        use_container_width=True,
        hide_index=True
    )