"""PhishLite: entry point. Menu that runs data check, EDA and baseline stages."""
from pathlib import Path
import traceback

from load import DEFAULT_LABEL, FULL_FILE, load_dataset, train_val_test
import eda
import baseline_v2 as baseline   # the file is baseline_v2.py, so import it under that name

EDA_DIR = Path("./reports/eda")
BASELINE_DIR = Path("./reports/baseline")


def print_header() -> None:
    print("\n" + "=" * 68)
    print("PhishLite: PHISHING DETECTION PROJECT")
    print("=" * 68)
    print(f"Dataset : {Path(FULL_FILE).resolve()}")
    print(f"Target  : {DEFAULT_LABEL}")
    print("-" * 68)


def check_dataset() -> None:
    """Load, deduplicate, and show how the data is split."""
    print("\n[1] DATA LOADING AND VALIDATION")
    df = load_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test(df)

    print(f"Rows after deduplication : {len(df):,}")
    print(f"Number of features       : {df.shape[1] - 1:,}")
    print(f"Train / Val / Test rows  : {len(X_train):,} / {len(X_val):,} / {len(X_test):,}")
    print("\nTarget distribution (0 = legitimate, 1 = phishing):")
    print(df[DEFAULT_LABEL].value_counts().sort_index().to_string())
    print("\nFeature values are kept as provided, including -1 placeholders.")


def run_eda() -> None:
    print("\n[2] EXPLORATORY DATA ANALYSIS")
    eda.main()
    print(f"\nEDA finished. Outputs are in {EDA_DIR}/")


def run_baseline() -> None:
    print("\n[3] BASELINE MODEL COMPARISON")
    baseline.main()
    print(f"\nBaseline finished. Results are in {BASELINE_DIR}/ (see cv_results.csv)")


def run_full_pipeline() -> None:
    print("\nSTARTING COMPLETE PIPELINE")
    check_dataset()
    run_eda()
    run_baseline()
    print("\nCOMPLETE PIPELINE FINISHED")


# menu choice -> (label, function)
ACTIONS = {
    "1": ("Run complete pipeline (Load + EDA + Baseline)", run_full_pipeline),
    "2": ("Load and validate dataset only", check_dataset),
    "3": ("Run EDA only", run_eda),
    "4": ("Run baseline models only", run_baseline),
}


def show_menu() -> None:
    print("\nPlease choose an operation:")
    for key, (label, _) in ACTIONS.items():
        print(f"  {key}. {label}")
    print("  5. Exit")


def main() -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    while True:
        print_header()
        show_menu()
        choice = input("\nEnter your choice (1-5): ").strip()

        if choice == "5":
            print("\nExiting. Goodbye!")
            break
        if choice not in ACTIONS:
            print("\nInvalid choice. Please enter a number from 1 to 5.")
            continue

        try:
            ACTIONS[choice][1]()
        except FileNotFoundError as exc:
            print(f"\n[FILE ERROR] {exc}")
            print("Check that the dataset exists at the path shown above.")
        except Exception as exc:
            print(f"\n[PIPELINE ERROR] {type(exc).__name__}: {exc}")
            traceback.print_exc()

        input("\nPress Enter to return to the main menu...")


if __name__ == "__main__":
    main()