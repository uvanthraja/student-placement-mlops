import os
import sys
import subprocess
from datetime import datetime


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

VALIDATION_SCRIPT = os.path.join(
    PROJECT_ROOT,
    "src",
    "data",
    "validate.py"
)

TRAINING_SCRIPT = os.path.join(
    PROJECT_ROOT,
    "src",
    "models",
    "train_mlflow.py"
)


# ============================================================
# HELPER FUNCTION
# ============================================================

def run_stage(stage_name, script_path):

    print("\n")
    print("=" * 70)
    print(f"STARTING STAGE: {stage_name}")
    print("=" * 70)

    start_time = datetime.now()

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=PROJECT_ROOT
    )

    end_time = datetime.now()

    duration = end_time - start_time

    print("\n" + "-" * 70)
    print(f"STAGE: {stage_name}")
    print(f"Duration: {duration}")
    print("-" * 70)

    if result.returncode != 0:

        print(f"✗ {stage_name} FAILED")

        return False

    print(f"✓ {stage_name} COMPLETED")

    return True


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    pipeline_start = datetime.now()

    print("\n")
    print("=" * 70)
    print(" STUDENT PLACEMENT MLOps PIPELINE")
    print("=" * 70)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Started at  : {pipeline_start}")

    # --------------------------------------------------------
    # STAGE 1 — DATA VALIDATION
    # --------------------------------------------------------

    validation_success = run_stage(
        "DATA VALIDATION",
        VALIDATION_SCRIPT
    )

    if not validation_success:

        print("\n" + "=" * 70)
        print("PIPELINE STOPPED")
        print("Reason: Data validation failed")
        print("=" * 70)

        sys.exit(1)

    # --------------------------------------------------------
    # STAGE 2 — MODEL TRAINING + MLFLOW
    # --------------------------------------------------------

    training_success = run_stage(
        "MODEL TRAINING + MLFLOW",
        TRAINING_SCRIPT
    )

    if not training_success:

        print("\n" + "=" * 70)
        print("PIPELINE FAILED")
        print("Reason: Model training failed")
        print("=" * 70)

        sys.exit(1)

    # --------------------------------------------------------
    # PIPELINE COMPLETE
    # --------------------------------------------------------

    pipeline_end = datetime.now()

    total_duration = pipeline_end - pipeline_start

    print("\n")
    print("=" * 70)
    print(" STUDENT PLACEMENT MLOps PIPELINE COMPLETED")
    print("=" * 70)

    print(f"Started : {pipeline_start}")
    print(f"Finished: {pipeline_end}")
    print(f"Duration: {total_duration}")

    print("\nPipeline stages:")
    print("  ✓ Data validation")
    print("  ✓ Model training")
    print("  ✓ MLflow experiment tracking")
    print("  ✓ Model evaluation")
    print("  ✓ Final model saved")

    print("\n✓ PIPELINE SUCCESSFUL")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()