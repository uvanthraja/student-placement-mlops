import os
import sys
import pandas as pd


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

REQUIRED_COLUMNS = [
    "Student ID",
    "Name",
    "Age",
    "Gender",
    "Branch",
    "Average GPA",
    "Backlogs",
    "Attendance (%)",
    "Clubs",
    "Skills",
    "Internship Done",
    "Internship Domain",
    "Placement Status",
    "Placement Domain",
    "CTC (LPA)",
    "Alumni Path",
    "Sem1 GPA",
    "Sem2 GPA",
    "Sem3 GPA",
    "Sem4 GPA",
    "Sem5 GPA",
    "Sem6 GPA",
    "Sem7 GPA",
    "Sem8 GPA",
]


# ============================================================
# VALIDATE DATASET
# ============================================================

def validate_dataset():

    print("\n" + "=" * 60)
    print("DATA VALIDATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Check file
    # --------------------------------------------------------

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    print(f"Dataset found: {DATA_PATH}")

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing_columns)
        )

    print("✓ Required columns present")

    # --------------------------------------------------------
    # Check target
    # --------------------------------------------------------

    target = "Placement Status"

    if df[target].isna().any():
        missing_target = df[target].isna().sum()

        print(
            f"⚠ Warning: {missing_target} rows "
            "have missing target values"
        )

    # --------------------------------------------------------
    # Check duplicate rows
    # --------------------------------------------------------

    duplicates = df.duplicated().sum()

    print(f"Duplicate rows: {duplicates}")

    # Duplicates are not automatically an error because
    # the training pipeline handles them.

    # --------------------------------------------------------
    # Check target values
    # --------------------------------------------------------

    print("\nTarget distribution:")

    print(
        df[target]
        .value_counts(dropna=False)
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    print("\n✓ DATA VALIDATION PASSED")

    return True


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:

        validate_dataset()

    except Exception as error:

        print("\n✗ DATA VALIDATION FAILED")
        print(f"Reason: {error}")

        sys.exit(1)