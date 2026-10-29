# Lifestyle Longevity KDD - UI Update

## What changed

- Added a `Health Predictor` tab to the existing Streamlit app.
- The predictor uses the existing `run_classification()` Random Forest pipeline.
- It predicts the existing target: `Good or better` or `Fair or poor` self-reported health.
- Added human-readable column labels and known value labels for NHANES coded fields.
- Kept `kdd_pipeline.py` unchanged so existing data processing and analysis remain intact.

## Apply the update

1. Back up your current `app.py`.
2. Replace the project's `app.py` with the `app.py` from this package.
3. Keep your existing `kdd_pipeline.py` and all `data/raw/*.XPT` files.
4. In PowerShell, from the project directory, run:
   `python -m streamlit run app.py`

The included `kdd_pipeline.py` is the supplied version for reference; you do not need to replace your existing pipeline if it is the same.

## Notes

- The form uses features already selected by the classification pipeline.
- The model is retrained when a prediction is submitted, so prediction may take a short while.
- Model probabilities are not clinical probabilities.
- The project predicts a self-reported health category, not life expectancy or a medical diagnosis.
