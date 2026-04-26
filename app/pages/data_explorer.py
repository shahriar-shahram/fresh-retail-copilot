import streamlit as st
from ui import inject_css, sidebar_guide, hero, section_title, muted, callout
import pandas as pd

st.set_page_config(page_title="Data Explorer | Fresh Retail Copilot", page_icon="📊", layout="wide")
inject_css()
sidebar_guide("Data Explorer")

hero(
    "📊 Data Explorer",
    "Pick a store-product case and inspect sales, availability, and missed demand.",
    badges=["Sales history", "Availability", "Lost sales", "Demand trend"],
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
section_title("Select a scenario", "Choose a store and product.")

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

callout(
    "How to read this page",
    "Low sales need availability context."
)

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["Current Context", "Recent Demand Trend", "Lost Sales Insight"])

# -----------------------
# Current context
# -----------------------
with tab1:
    st.subheader("Current Context")

    col1, col2, col3 = st.columns(3)
    col1.metric("Latest Observed Sales", f"{latest['sale_amount']:.2f}")
    col2.metric("In-Stock Ratio", f"{latest['in_stock_ratio']:.2f}")
    col3.metric("Discount", f"{latest['discount']:.2f}")

    st.caption(
        "In-stock ratio helps explain whether low sales may come from limited availability."
    )

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

    st.dataframe(context_df, use_container_width=True)

# -----------------------
# Recent demand trend
# -----------------------
with tab2:
    st.subheader("Recent Demand Trend")

    plot_df = subset[["dt", "sale_amount"]].tail(30).copy()
    plot_df["dt"] = pd.to_datetime(plot_df["dt"])

    st.line_chart(plot_df.set_index("dt")[["sale_amount"]])

    st.caption(
        "Recent observed sales for the selected product."
    )

# -----------------------
# Lost sales insight
# -----------------------
with tab3:
    st.subheader("Lost Sales Insight")

    if not lost_subset.empty:
        total_lost = float(lost_subset["lost_sales"].sum())
        avg_lost = float(lost_subset["lost_sales"].mean())

        col1, col2 = st.columns(2)
        col1.metric("Total Estimated Lost Sales", f"{total_lost:.2f}")
        col2.metric("Average Lost Sales", f"{avg_lost:.4f}")

        st.caption(
            "Lost sales estimate the gap between what was sold and what demand may have been during constrained availability."
        )

        display_lost = lost_subset[
            ["dt", "sale_amount", "predicted_true_demand", "lost_sales"]
        ].tail(10).copy()

        display_lost["dt"] = pd.to_datetime(display_lost["dt"])
        display_lost["sale_amount"] = pd.to_numeric(display_lost["sale_amount"], errors="coerce")
        display_lost["predicted_true_demand"] = pd.to_numeric(display_lost["predicted_true_demand"], errors="coerce")
        display_lost["lost_sales"] = pd.to_numeric(display_lost["lost_sales"], errors="coerce")

        st.dataframe(display_lost, use_container_width=True)
    else:
        st.info("No lost-sales records found for this selection.")
