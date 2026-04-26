import streamlit as st
from ui import inject_css, sidebar_guide, hero, section_title, muted, callout
import requests
import pandas as pd
import os

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://fresh-retail-copilot-api-837696130499.us-central1.run.app"
)

st.set_page_config(page_title="Forecast Demo | Fresh Retail Copilot", page_icon="🔮", layout="wide")
inject_css()
sidebar_guide("Forecast Demo")

hero(
    "🔮 Forecast Demo",
    "Choose a scenario and run the next-demand forecast.",
    badges=["FastAPI", "Prediction", "Feature payload", "Business insight"],
)

callout(
    "What happens when you click Predict",
    "The selected feature row goes to the API. The forecast comes back and is plotted with recent sales."
)

# -----------------------
# Load real data
# -----------------------
@st.cache_data
def load_data():
    df = pd.read_parquet("data/processed/model_data.parquet")
    lost_df = pd.read_parquet("data/processed/lost_sales.parquet")

    df["dt"] = pd.to_datetime(df["dt"])
    lost_df["dt"] = pd.to_datetime(lost_df["dt"])

    return df, lost_df


df, lost_df = load_data()

# -----------------------
# Scenario selection
# -----------------------
section_title("Select a scenario", "Choose a store and product, then run inference.")

stores = sorted(df["store_id"].unique().tolist())
selected_store = st.selectbox("Select Store", stores)

store_products = sorted(
    df[df["store_id"] == selected_store]["product_id"].unique().tolist()
)
selected_product = st.selectbox("Select Product", store_products)

subset = df[
    (df["store_id"] == selected_store) &
    (df["product_id"] == selected_product)
].sort_values("dt")

if subset.empty:
    st.error("No data found for this store-product combination.")
    st.stop()

latest = subset.iloc[-1]

lost_subset = lost_df[
    (lost_df["store_id"] == selected_store) &
    (lost_df["product_id"] == selected_product)
].sort_values("dt")

# -----------------------
# Current selected context
# -----------------------
st.subheader("Selected context")

col1, col2, col3 = st.columns(3)
col1.metric("Latest Observed Sales", f"{latest['sale_amount']:.2f}")
col2.metric("In-Stock Ratio", f"{latest['in_stock_ratio']:.2f}")
col3.metric("Discount", f"{latest['discount']:.2f}")

# -----------------------
# Build API payload from latest real row
# -----------------------
payload = {
    "lag_1": float(latest["lag_1"]),
    "lag_7": float(latest["lag_7"]),
    "rolling_mean_7": float(latest["rolling_mean_7"]),
    "discount": float(latest["discount"]),
    "holiday_flag": int(latest["holiday_flag"]),
    "activity_flag": int(latest["activity_flag"]),
    "precpt": float(latest["precpt"]),
    "avg_temperature": float(latest["avg_temperature"]),
    "avg_humidity": float(latest["avg_humidity"]),
    "avg_wind_level": float(latest["avg_wind_level"]),
    "day_of_week": int(latest["day_of_week"]),
    "month": int(latest["month"]),
    "avg_sales_when_available": float(latest["avg_sales_when_available"]),
    "stockout_hours": int(latest["stockout_hours"]),
    "demand_std": float(latest["demand_std"]),
}

with st.expander("View API payload"):
    st.json(payload)

# -----------------------
# Prediction
# -----------------------
st.subheader("Prediction")

if st.button("Predict Next Demand", width="stretch"):
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            timeout=10,
        )
        response.raise_for_status()
        result = response.json()

        st.success("Prediction complete")
        pred_value = float(result["predicted_demand"])

        col1, col2 = st.columns(2)
        col1.metric("Predicted Next Demand", f"{pred_value:.4f}")

        if not lost_subset.empty:
            impact_pct = (
                lost_subset["lost_sales"].sum() /
                (lost_subset["sale_amount"].sum() + 1e-6)
            ) * 100
            col2.metric("Estimated Demand Loss (%)", f"{impact_pct:.2f}%")
        else:
            col2.metric("Estimated Demand Loss (%)", "N/A")

        # -------------------------
        # Business Insight
        # -------------------------
        st.subheader("Quick read")

        insight = []

        if latest["in_stock_ratio"] < 0.5:
            insight.append("⚠️ Low stock availability may be suppressing sales.")

        if latest["discount"] < 0.8:
            insight.append("💸 Discount is low — potential demand may not be fully activated.")

        if latest["holiday_flag"] == 1:
            insight.append("🎉 Holiday effect may increase demand.")

        if latest["avg_temperature"] > 25:
            insight.append("🌡 High temperature may affect fresh product demand.")

        if not insight:
            insight.append("✅ No major demand constraints detected.")

        for item in insight:
            st.write(item)

        # -------------------------
        # Prediction vs Trend
        # -------------------------
        st.subheader("Prediction vs. recent sales")

        pred_plot_df = subset[["dt", "sale_amount"]].tail(30).copy()
        pred_plot_df["dt"] = pd.to_datetime(pred_plot_df["dt"])

        new_row = pred_plot_df.iloc[-1:].copy()
        new_row.loc[:, "dt"] = new_row["dt"] + pd.Timedelta(days=1)
        new_row.loc[:, "sale_amount"] = pred_value

        combined = pd.concat([pred_plot_df, new_row], ignore_index=True)
        combined["dt"] = pd.to_datetime(combined["dt"])

        st.line_chart(combined.set_index("dt")[["sale_amount"]])

    except Exception as e:
        st.error(f"API request failed: {e}")
        st.info(f"Make sure the API is reachable at {API_BASE_URL}")

st.sidebar.write("API Base URL:", API_BASE_URL)
