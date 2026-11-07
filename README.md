# House Price Prediction System Using Regression

A Python + Streamlit academic project adapted from the structure and workflow of the reference Credit Card Fraud Detection System. It changes the task from binary classification to continuous-value regression.

## Features
- Housing dataset exploration and summary
- Data preprocessing with median/mode imputation, scaling and one-hot encoding
- Comparison of Linear Regression, Random Forest Regressor and Gradient Boosting Regressor
- Model selection using test RMSE
- MAE, RMSE and R² evaluation
- Single-property prediction and bulk CSV prediction/download

## Project structure
```text
house_price_prediction/
├── app.py
├── generate_dataset.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── house_prices.csv       # generated demo data
├── models/
│   └── house_price_model.pkl  # generated after training
└── results/
    └── metrics.json           # generated after training
```

## Run on Windows / VS Code
Open the terminal in this folder and run:

```bash
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
python generate_dataset.py
python train_model.py
streamlit run app.py
```

If PowerShell blocks activation, use `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again. You can also run `venv\Scripts\python.exe -m pip install -r requirements.txt` without activation.

## Dataset format
The CSV must contain these columns:

- `area_sqft`: built-up area in square feet
- `bedrooms`: number of bedrooms
- `bathrooms`: number of bathrooms
- `floors`: number of floors
- `age_years`: property age
- `parking`: number of parking spaces
- `location`: location category
- `price_lakhs`: target price in Indian lakhs

The included generator creates synthetic data for demonstration and testing only. It is not real estate market data and should not be used for actual buying, selling, lending or investment decisions. For a credible academic evaluation, replace `data/house_prices.csv` with a documented, licensed dataset from the region being studied, then rerun training.

## Algorithms
1. **Linear Regression** — baseline model that estimates a linear relationship between house features and price.
2. **Random Forest Regressor** — ensemble of decision trees that can learn nonlinear relationships.
3. **Gradient Boosting Regressor** — sequential ensemble that improves predictions by fitting residual errors.

## Evaluation
- **MAE:** average absolute difference between actual and predicted prices.
- **RMSE:** square root of mean squared error; penalizes large errors more strongly.
- **R²:** goodness-of-fit measure; values closer to 1 indicate more variance explained, though test performance and data quality matter.

## Reference adaptation
The reference repository uses separate data exploration, model training and Streamlit application scripts, saves a trained model and metrics, and offers Home, Dataset Overview, Model Performance and prediction pages. This project retains that organization and replaces classification-specific pieces (fraud labels, confusion matrix, precision/recall/F1) with regression models, continuous price output and MAE/RMSE/R².
