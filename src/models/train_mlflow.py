import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "students.csv"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

os.makedirs(MODEL_DIR, exist_ok=True)

# MLflow experiment
mlflow.set_experiment("Student Placement Prediction")

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

# ============================================================
# REMOVE DUPLICATES
# ============================================================

df = df.drop_duplicates().reset_index(drop=True)

# ============================================================
# TARGET
# ============================================================

TARGET = "Placement Status"

# Remove rows where target is missing
df = df.dropna(subset=[TARGET]).reset_index(drop=True)

# Convert target to binary
target_mapping = {
    "Placed": 1,
    "Not Placed": 0,
    "Yes": 1,
    "No": 0,
    "1": 1,
    "0": 0,
    1: 1,
    0: 0,
}

y = df[TARGET].map(target_mapping)

# Check target conversion
if y.isna().any():
    print("Unique target values found:")
    print(df[TARGET].unique())

    raise ValueError(
        "Target contains values that were not recognized."
    )

y = y.astype(int)

# ============================================================
# REMOVE DATA LEAKAGE
# ============================================================

LEAKAGE_COLUMNS = [
    "Student ID",
    "Name",
    "Placement Domain",
    "CTC (LPA)",
    TARGET,
]

X = df.drop(
    columns=[
        col for col in LEAKAGE_COLUMNS
        if col in df.columns
    ]
)

print("\nInput features:")
print(X.columns.tolist())

print("\nNumber of features:", X.shape[1])

# ============================================================
# IDENTIFY NUMERIC AND CATEGORICAL FEATURES
# ============================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ============================================================
# PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        ),
    ]
)

# ============================================================
# MODELS
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),
}

# ============================================================
# TRAIN AND LOG EACH MODEL
# ============================================================

results = []

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print("Training:", model_name)
    print("=" * 60)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    with mlflow.start_run(run_name=model_name):

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        pipeline.fit(X_train, y_train)

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        y_pred = pipeline.predict(X_test)

        # ----------------------------------------------------
        # Probability for ROC-AUC
        # ----------------------------------------------------

        if hasattr(pipeline, "predict_proba"):
            y_probability = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_probability = None

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        if y_probability is not None:
            roc_auc = roc_auc_score(
                y_test,
                y_probability
            )
        else:
            roc_auc = 0.0

        # ----------------------------------------------------
        # Log parameters
        # ----------------------------------------------------

        mlflow.log_param(
            "model",
            model_name
        )

        mlflow.log_param(
            "test_size",
            0.20
        )

        mlflow.log_param(
            "random_state",
            42
        )

        mlflow.log_param(
            "num_features",
            X.shape[1]
        )

        mlflow.log_param(
            "train_samples",
            len(X_train)
        )

        mlflow.log_param(
            "test_samples",
            len(X_test)
        )

        # ----------------------------------------------------
        # Log metrics
        # ----------------------------------------------------

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "precision",
            precision
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_metric(
            "f1_score",
            f1
        )

        mlflow.log_metric(
            "roc_auc",
            roc_auc
        )

        # ----------------------------------------------------
        # Log model
        # ----------------------------------------------------
        #
        # NOTE: Newer MLflow versions default to the "skops"
        # serialization format for sklearn models. skops runs a
        # security audit on every type it finds inside the
        # pickled object and refuses to save anything it doesn't
        # explicitly recognize as safe — here it rejects
        # numpy.dtype, which shows up via the OneHotEncoder /
        # ColumnTransformer metadata.
        #
        # Using cloudpickle instead avoids that audit entirely
        # and behaves the same way plain pickle/joblib did in
        # older MLflow versions.
        # ----------------------------------------------------

        mlflow.sklearn.log_model(
            pipeline,
            name="placement_model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print("Accuracy :", round(accuracy, 4))
        print("Precision:", round(precision, 4))
        print("Recall   :", round(recall, 4))
        print("F1 Score :", round(f1, 4))
        print("ROC-AUC  :", round(roc_auc, 4))

        results.append(
            {
                "model": model_name,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": roc_auc,
            }
        )

# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("MLFLOW MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))

# ============================================================
# SELECT BEST MODEL BASED ON F1
# ============================================================

best_model_name = results_df.loc[
    results_df["f1_score"].idxmax(),
    "model"
]

print("\nBest model based on F1:", best_model_name)

# ============================================================
# TRAIN FINAL MODEL
# ============================================================

final_model = models[best_model_name]

final_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", final_model),
    ]
)

final_pipeline.fit(
    X_train,
    y_train
)

# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = os.path.join(
    MODEL_DIR,
    "placement_pipeline.pkl"
)

joblib.dump(
    final_pipeline,
    final_model_path
)

print("\nFinal model saved to:")
print(final_model_path)

print("\n✓ MLFLOW TRAINING COMPLETED")