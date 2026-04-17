import streamlit as st
import requests
import pandas as pd
import os

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8080")

st.set_page_config(page_title="Fresh Retail Copilot", layout="wide")

st.title("Fresh Retail Copilot")
st.write("Stockout-aware demand forecasting and retail intelligence")

# -----------------------
# Load real data
# -----------------------
df = pd.read_parquet("data/processed/model_data.parquet")
lost_df = pd.read_parquet("data/processed/lost_sales.parquet")

df["dt"] = pd.to_datetime(df["dt"])
lost_df["dt"] = pd.to_datetime(lost_df["dt"])

# -----------------------
# Scenario selection
# -----------------------
st.subheader("Select Scenario")

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

# -----------------------
# Current context
# -----------------------
st.subheader("Current Context")

col1, col2, col3 = st.columns(3)
col1.metric("Latest Observed Sales", f"{latest['sale_amount']:.2f}")
col2.metric("In-Stock Ratio", f"{latest['in_stock_ratio']:.2f}")
col3.metric("Discount", f"{latest['discount']:.2f}")

context_df = pd.DataFrame([{
    "dt": pd.to_datetime(latest["dt"]),
    "sale_amount": float(latest["sale_amount"]),
    "in_stock_ratio": float(latest["in_stock_ratio"]),
    "stockout_hours": int(latest["stockout_hours"]),
    "avg_temperature": float(latest["avg_temperature"]),
    "avg_humidity": float(latest["avg_humidity"]),
    "avg_wind_level": float(latest["avg_wind_level"]),
    "holiday_flag": int(latest["holiday_flag"]),
    "activity_flag": int(latest["activity_flag"]),
}])

st.dataframe(context_df, width="stretch")

# -----------------------
# Recent demand trend
# -----------------------
st.subheader("Recent Demand Trend")
plot_df = subset[["dt", "sale_amount"]].tail(30).copy()
plot_df["dt"] = pd.to_datetime(plot_df["dt"])
st.line_chart(plot_df.set_index("dt")[["sale_amount"]])

# -----------------------
# Lost sales insight
# -----------------------
st.subheader("Lost Sales Insight")

lost_subset = lost_df[
    (lost_df["store_id"] == selected_store) &
    (lost_df["product_id"] == selected_product)
].sort_values("dt")

if not lost_subset.empty:
    total_lost = float(lost_subset["lost_sales"].sum())
    avg_lost = float(lost_subset["lost_sales"].mean())

    col1, col2 = st.columns(2)
    col1.metric("Total Estimated Lost Sales", f"{total_lost:.2f}")
    col2.metric("Average Lost Sales", f"{avg_lost:.4f}")

    display_lost = lost_subset[
        ["dt", "sale_amount", "predicted_true_demand", "lost_sales"]
    ].tail(10).copy()

    display_lost["dt"] = pd.to_datetime(display_lost["dt"])
    display_lost["sale_amount"] = pd.to_numeric(display_lost["sale_amount"], errors="coerce")
    display_lost["predicted_true_demand"] = pd.to_numeric(display_lost["predicted_true_demand"], errors="coerce")
    display_lost["lost_sales"] = pd.to_numeric(display_lost["lost_sales"], errors="coerce")

    st.dataframe(display_lost, width="stretch")
else:
    st.info("No lost-sales records found for this selection.")

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

# -----------------------
# Prediction
# -----------------------
st.subheader("Next-Demand Prediction")

if st.button("Predict Next Demand"):
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            timeout=10,
        )
        response.raise_for_status()
        result = response.json()

        st.success("Prediction complete")
        st.metric("Predicted Next Demand", f"{result['predicted_demand']:.4f}")

        # -------------------------
        # Impact Summary
        # -------------------------
        st.subheader("Impact Summary")

        if not lost_subset.empty:
            impact_pct = (
                lost_subset["lost_sales"].sum() /
                (lost_subset["sale_amount"].sum() + 1e-6)
            ) * 100

            st.metric("Estimated Demand Loss (%)", f"{impact_pct:.2f}%")

        # -------------------------
        # Business Insight
        # -------------------------
        st.subheader("Business Insight")

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
        st.subheader("Prediction vs Recent Trend")

        st.sidebar.write("API Base URL:", API_BASE_URL)

        pred_plot_df = subset[["dt", "sale_amount"]].tail(30).copy()
        pred_plot_df["dt"] = pd.to_datetime(pred_plot_df["dt"])

        pred_value = float(result["predicted_demand"])

        new_row = pred_plot_df.iloc[-1:].copy()
        new_row.loc[:, "dt"] = new_row["dt"] + pd.Timedelta(days=1)
        new_row.loc[:, "sale_amount"] = pred_value

        combined = pd.concat([pred_plot_df, new_row], ignore_index=True)
        combined["dt"] = pd.to_datetime(combined["dt"])

        st.line_chart(combined.set_index("dt")[["sale_amount"]])

    except Exception as e:
        st.error(f"API request failed: {e}")
        st.info(f"Make sure the API is reachable at {API_BASE_URL}")