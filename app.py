from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "house_price_model.pkl"
DATA_PATH = ROOT / "data" / "house_prices.csv"
METRICS_PATH = ROOT / "results" / "metrics.json"

st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.title("🏠 House Price Prediction System")
st.caption("Estimate residential property prices using supervised machine learning regression.")

if not MODEL_PATH.exists() or not METRICS_PATH.exists():
    st.warning("The model has not been trained yet. Run `python generate_dataset.py` and then `python train_model.py` in the project folder.")
    st.stop()

model = load_model()
metrics = json.loads(METRICS_PATH.read_text())
page = st.sidebar.radio("Navigation", ["Home", "Dataset Overview", "Model Performance", "Price Prediction"])

if page == "Home":
    st.header("Welcome")
    st.write("This application estimates a house's price from its size, rooms, age, location, floors and parking availability. It follows a workflow of data exploration, preprocessing, regression model comparison, evaluation and prediction.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Selected model", metrics["best_model"])
    c2.metric("Test records", metrics["test_rows"])
    c3.metric("Regression models", len(metrics["metrics"]))
    st.subheader("Project features")
    st.markdown("- Explore housing data and missing values\n- Compare Linear Regression, Random Forest and Gradient Boosting\n- Predict an estimated price from user-entered property details\n- Upload a CSV to predict multiple properties and download results")
    st.info("The included dataset is synthetic demo data. Replace it with reliable, recent local property-sale data before using predictions for real decisions.")

elif page == "Dataset Overview":
    st.header("Dataset Overview")
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        a, b, c = st.columns(3)
        a.metric("Properties", f"{len(df):,}")
        b.metric("Features", len(df.columns) - 1)
        c.metric("Missing values", int(df.isna().sum().sum()))
        st.subheader("Sample records")
        st.dataframe(df.head(20), use_container_width=True)
        st.subheader("Price distribution (INR lakhs)")
        st.bar_chart(df["price_lakhs"].round(0).value_counts().sort_index())
        st.subheader("Average price by location")
        st.bar_chart(df.groupby("location")["price_lakhs"].mean().sort_values())
    else:
        st.error("Dataset not found at data/house_prices.csv")

elif page == "Model Performance":
    st.header("Regression Model Performance")
    st.write(f"**Selected model:** {metrics['best_model']}")
    table = pd.DataFrame(metrics["metrics"]).T.rename(columns={"mae": "MAE (lakhs)", "rmse": "RMSE (lakhs)", "r2": "R² score"})
    st.dataframe(table.style.format({"MAE (lakhs)": "{:.2f}", "RMSE (lakhs)": "{:.2f}", "R² score": "{:.3f}"}), use_container_width=True)
    st.markdown("**MAE:** average absolute prediction error. **RMSE:** error measure that penalizes larger mistakes. **R²:** proportion of target variation explained by the model; closer to 1 is generally better.")

elif page == "Price Prediction":
    st.header("Predict House Price")
    st.write("Enter the property details below. The result is an estimated price in Indian rupees (lakhs).")
    with st.form("prediction_form"):
        left, right = st.columns(2)
        area = left.number_input("Area (square feet)", min_value=200, max_value=20000, value=1200, step=50)
        bedrooms = right.number_input("Bedrooms", min_value=1, max_value=10, value=2)
        bathrooms = left.number_input("Bathrooms", min_value=1, max_value=10, value=2)
        floors = right.number_input("Floors", min_value=1, max_value=30, value=2)
        age = left.number_input("Property age (years)", min_value=0, max_value=150, value=5)
        parking = right.selectbox("Parking spaces", [0, 1, 2, 3, 4])
        location = st.selectbox("Location", ["Hyderabad", "Bengaluru", "Chennai", "Pune", "Mumbai", "Delhi", "Other"])
        submitted = st.form_submit_button("Predict Price", type="primary")
    if submitted:
        row = pd.DataFrame([{"area_sqft": area, "bedrooms": bedrooms, "bathrooms": bathrooms,
                             "floors": floors, "age_years": age, "parking": parking, "location": location}])
        estimate = float(model.predict(row)[0])
        st.success(f"Estimated house price: ₹{estimate:,.2f} lakhs")
        st.caption("This is a model estimate, not a property valuation or guaranteed selling price.")
    st.divider()
    st.subheader("Bulk prediction using CSV")
    st.caption("Upload a CSV containing area_sqft, bedrooms, bathrooms, floors, age_years, parking and location.")
    uploaded = st.file_uploader("Choose property CSV", type=["csv"])
    if uploaded:
        try:
            batch = pd.read_csv(uploaded)
            expected = metrics["feature_columns"]
            missing = [col for col in expected if col not in batch.columns]
            if missing:
                st.error("Missing required columns: " + ", ".join(missing))
            else:
                batch["predicted_price_lakhs"] = model.predict(batch[expected])
                st.dataframe(batch, use_container_width=True)
                st.download_button("Download predictions CSV", batch.to_csv(index=False).encode("utf-8"), "house_price_predictions.csv", "text/csv")
        except Exception as exc:
            st.error(f"Could not process the file: {exc}")
