from pathlib import Path
import pandas as pd

# Project folders
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
CSV_DIR = BASE_DIR / "data" / "csv"

CSV_DIR.mkdir(parents=True, exist_ok=True)

if not RAW_DIR.exists() or not list(RAW_DIR.glob("*.XPT")):
    print("No raw datasets found.")
    print("Run your Streamlit application first:")
    print("python -m streamlit run app.py")
    raise SystemExit(1)

for file_path in RAW_DIR.glob("*.XPT"):
    print(f"Converting {file_path.name}...")

    try:
        df = pd.read_sas(
            file_path,
            format="xport",
            encoding="utf-8",
        )

        output_path = CSV_DIR / f"{file_path.stem}.csv"
        df.to_csv(output_path, index=False)

        print(f"Saved: {output_path}")
        print(f"Rows: {len(df):,} | Columns: {len(df.columns)}")

    except Exception as error:
        print(f"Error converting {file_path.name}: {error}")

print("\nConversion complete!")
print(f"CSV files are located in: {CSV_DIR}")