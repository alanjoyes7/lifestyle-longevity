import pandas as pd
from pathlib import Path
from tabulate import tabulate

# Folder containing your CSV datasets
DATA_FOLDER = Path("data/csv")

# Number of rows to display from each dataset
ROWS_TO_PRINT = 5

# Find all CSV files
csv_files = sorted(DATA_FOLDER.glob("*.csv"))

if not csv_files:
    print(f"No CSV files found in {DATA_FOLDER.resolve()}")
else:
    print(f"\nFound {len(csv_files)} CSV datasets\n")

    for file in csv_files:
        try:
            df = pd.read_csv(file)

            print("\n" + "=" * 100)
            print(f" DATASET: {file.name}")
            print(f" Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
            print("=" * 100)

            # Display the first 5 rows in a readable table
            print(
                tabulate(
                    df.head(ROWS_TO_PRINT),
                    headers="keys",
                    tablefmt="rounded_grid",
                    showindex=False,
                    numalign="right",
                    stralign="left",
                    maxcolwidths=20
                )
            )

        except Exception as e:
            print(f"Error reading {file.name}: {e}")

    print("\nFinished displaying all datasets.")