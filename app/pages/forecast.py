import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

from ui import inject_css, sidebar_guide, hero, section_title, callout, product_card

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://fresh-retail-copilot-api-837696130499.us-central1.run.app",
)

MONTH_FEATURES = [
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "discount",
    "holiday_flag",
    "activity_flag",
    "precpt",
    "avg_temperature",
    "avg_humidity",
    "avg_wind_level",
    "day_of_week",
    "month",
    "avg_sales_when_available",
    "stockout_hours",
    "demand_std",
]

HISTORY_COLUMNS = [
    "store_id",
    "product_id",
    "dt",
    "sale_amount",
    "in_stock_ratio",
    "stockout_hours",
    "discount",
    "holiday_flag",
    "activity_flag",
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "precpt",
    "avg_temperature",
    "avg_humidity",
    "avg_wind_level",
    "day_of_week",
    "month",
    "avg_sales_when_available",
    "demand_std",
]

HORIZON_MAP = {
    "1 day ahead": 1,
    "1 week ahead": 7,
    "1 month ahead": 30,
}


inject_css()
sidebar_guide("Forecast")

hero(
    "🔮 Forecast",
    (
        "Forecast demand for a selected store-product pair. "
        "This view shows the recent demand history together with a forward forecast path "
        "for 1 day, 1 week, or 1 month."
    ),
    badges=["Recent history", "Forward forecast path", "1 / 7 / 30 periods"],
    eyebrow="Demand forecast workspace",
)


@st.cache_data(show_spinner=False)
def load_history():
    train_df = pd.read_parquet("data/processed/model_train_rich.parquet", columns=HISTORY_COLUMNS)
    eval_df = pd.read_parquet("data/processed/model_eval_rich.parquet", columns=HISTORY_COLUMNS)

    df = pd.concat([train_df, eval_df], ignore_index=True)
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values(["store_id", "product_id", "dt"]).reset_index(drop=True)
    return df


@st.cache_data(ttl=300, show_spinner=False)
def load_month_model_metadata():
    try:
        r = requests.get(f"{API_BASE_URL}/month-horizon-model", timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def make_month_payload(row):
    payload = {}
    int_cols = {"holiday_flag", "activity_flag", "day_of_week", "month"}

    for col in MONTH_FEATURES:
        if col in int_cols:
            payload[col] = int(row[col])
        else:
            payload[col] = float(row[col])

    return payload


def predict_month_horizon(payload):
    r = requests.post(
        f"{API_BASE_URL}/predict-month-horizon",
        json=payload,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def pct_change(current_value, reference_value):
    if abs(reference_value) < 1e-9:
        return 0.0
    return (current_value - reference_value) / reference_value * 100.0


history_df = load_history()
month_meta = load_month_model_metadata()
month_metrics = month_meta.get("metrics", {})

section_title(
    "1. Forecast setup",
    "This page uses a direct multi-horizon model. It predicts a full forward path, and the selected business duration shows the first 1, 7, or 30 forecasted periods."
)

m1, m2, m3, m4 = st.columns(4)
with m1:
    product_card("Model", month_meta.get("model_name", "Direct multi-horizon model"))
with m2:
    product_card("Day +1 validation MAE", f"{month_metrics.get('h1_mae', 0):.4f}")
with m3:
    product_card("Day +7 validation MAE", f"{month_metrics.get('h7_mae', 0):.4f}")
with m4:
    product_card("Day +30 validation MAE", f"{month_metrics.get('h30_mae', 0):.4f}")

callout(
    "How to read this page",
    "A 1-week forecast means a 7-period forecast path. A 1-month forecast means a 30-period forecast path. "
    "So the chart shows the full path over that selected duration, not just one single future point."
)

st.markdown("---")

section_title(
    "2. Choose the forecast scenario",
    "Select the store, product, and planning duration."
)

c1, c2, c3 = st.columns([1.1, 1.1, 1.2])

with c1:
    store_options = sorted(history_df["store_id"].unique().tolist())
    selected_store = st.selectbox("Store", store_options)

with c2:
    product_options = sorted(
        history_df.loc[history_df["store_id"] == selected_store, "product_id"].unique().tolist()
    )
    selected_product = st.selectbox("Product", product_options)

with c3:
    selected_duration = st.radio(
        "Forecast duration",
        ["1 day ahead", "1 week ahead", "1 month ahead"],
        index=1,
        help="This controls how many future periods are shown in the forecast path.",
    )

history_window = st.selectbox(
    "Historical window shown",
    [14, 30, 60, 90],
    index=3,
    help="How many most recent observed records to show before the forecast starts.",
)

subset = history_df[
    (history_df["store_id"] == selected_store) &
    (history_df["product_id"] == selected_product)
].sort_values("dt").reset_index(drop=True)

if subset.empty:
    st.error("No data found for this store-product combination.")
    st.stop()

available_history = len(subset)
recent = subset.tail(history_window).copy()
latest = subset.iloc[-1].copy()

selected_steps = HORIZON_MAP[selected_duration]
last_observed_date = pd.to_datetime(latest["dt"])
recent_avg_7 = float(subset.tail(7)["sale_amount"].mean()) if len(subset) >= 7 else float(subset["sale_amount"].mean())
latest_sales = float(latest["sale_amount"])

section_title(
    "3. Forecast definition",
    "What the selected forecast means."
)

d1, d2, d3 = st.columns(3)
with d1:
    product_card(
        "What are we forecasting?",
        "Expected future demand for the selected store-product pair."
    )
with d2:
    product_card(
        "Forecast horizon",
        f"{selected_steps} future period(s)."
    )
with d3:
    product_card(
        "Forecast unit",
        "Demand in `sale_amount` units."
    )

section_title(
    "4. Current context",
    "Recent demand and latest observed business context."
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Latest observed sales", f"{latest_sales:.2f}")
k2.metric("Recent 7-period average", f"{recent_avg_7:.2f}")
k3.metric("In-stock ratio", f"{float(latest['in_stock_ratio']):.2f}")
k4.metric("Stockout hours", f"{float(latest['stockout_hours']):.0f}")

with st.expander("Features used by the forecast model"):
    st.write(
        "The forecast is based on demand lags, rolling demand statistics, discount/activity signals, "
        "weather variables, calendar variables, stockout context, and recent average sales when the item was available."
    )
    st.json(make_month_payload(latest))

section_title("5. Run forecast")

if not st.button("Generate forecast", use_container_width=True):
    st.info("Click the button to generate the forecast path.")
    st.stop()

payload = make_month_payload(latest)

try:
    result = predict_month_horizon(payload)
except Exception as e:
    st.error(f"API request failed: {e}")
    st.info(f"Make sure the API is reachable at {API_BASE_URL}")
    st.stop()

forecast_df = pd.DataFrame(result["forecasts"]).copy()
forecast_df["horizon"] = forecast_df["horizon"].astype(int)
forecast_df["predicted_demand"] = forecast_df["predicted_demand"].astype(float)
forecast_df["forecast_date"] = last_observed_date + pd.to_timedelta(forecast_df["horizon"], unit="D")

path_df = forecast_df[forecast_df["horizon"] <= selected_steps].copy().reset_index(drop=True)

if path_df.empty:
    st.error("No forecast path was returned by the API.")
    st.stop()

final_point = float(path_df.iloc[-1]["predicted_demand"])
horizon_total = float(path_df["predicted_demand"].sum())
horizon_avg = float(path_df["predicted_demand"].mean())
avg_vs_recent = pct_change(horizon_avg, recent_avg_7)

section_title(
    "6. Forecast result",
    "Summary of the selected forecast horizon."
)

r1, r2, r3, r4 = st.columns(4)
r1.metric(f"Day +{selected_steps} forecast", f"{final_point:.2f} units")
r2.metric(f"{selected_steps}-period total", f"{horizon_total:.2f} units")
r3.metric("Horizon avg vs. recent avg", f"{avg_vs_recent:+.1f}%")
r4.metric("Model type", result.get("model_type", "direct_multi_horizon"))

if avg_vs_recent > 10:
    st.success("The projected average over the selected horizon is above the recent 7-period average.")
elif avg_vs_recent < -10:
    st.warning("The projected average over the selected horizon is below the recent 7-period average.")
else:
    st.info("The projected average over the selected horizon is close to the recent 7-period average.")

section_title(
    "7. Forecast visualization",
    "Here you can see the recent observed demand together with the forward-looking forecast path."
)

hist_plot_df = recent[["dt", "sale_amount"]].copy()
hist_plot_df["dt"] = pd.to_datetime(hist_plot_df["dt"])
hist_plot_df["series"] = "Historical observed sales"
hist_plot_df["value"] = hist_plot_df["sale_amount"].astype(float)

forecast_plot_df = pd.concat(
    [
        pd.DataFrame(
            {
                "dt": [last_observed_date],
                "series": ["Forecast path"],
                "value": [latest_sales],
                "horizon": [0],
            }
        ),
        path_df.rename(columns={"forecast_date": "dt", "predicted_demand": "value"})[
            ["dt", "value", "horizon"]
        ].assign(series="Forecast path"),
    ],
    ignore_index=True,
)

history_line = alt.Chart(hist_plot_df).mark_line(strokeWidth=3).encode(
    x=alt.X("dt:T", title="Date"),
    y=alt.Y("value:Q", title="Demand (sale_amount units)"),
    tooltip=[
        alt.Tooltip("dt:T", title="Date"),
        alt.Tooltip("value:Q", title="Observed demand", format=".2f"),
    ],
)

forecast_line = alt.Chart(forecast_plot_df).mark_line(strokeDash=[6, 4], strokeWidth=3).encode(
    x=alt.X("dt:T", title="Date"),
    y=alt.Y("value:Q", title="Demand (sale_amount units)"),
    tooltip=[
        alt.Tooltip("dt:T", title="Date"),
        alt.Tooltip("horizon:Q", title="Periods ahead"),
        alt.Tooltip("value:Q", title="Forecasted demand", format=".2f"),
    ],
)

forecast_points = alt.Chart(forecast_plot_df[forecast_plot_df["horizon"] > 0]).mark_point(filled=True, size=65).encode(
    x=alt.X("dt:T", title="Date"),
    y=alt.Y("value:Q", title="Demand (sale_amount units)"),
    tooltip=[
        alt.Tooltip("dt:T", title="Forecast date"),
        alt.Tooltip("horizon:Q", title="Periods ahead"),
        alt.Tooltip("value:Q", title="Forecasted demand", format=".2f"),
    ],
)

boundary_rule = alt.Chart(
    pd.DataFrame({"dt": [last_observed_date]})
).mark_rule(strokeDash=[2, 2]).encode(
    x="dt:T"
)

chart = (history_line + forecast_line + forecast_points + boundary_rule).properties(height=430)
st.altair_chart(chart, use_container_width=True)

st.caption(
    f"Showing {len(recent)} observed record(s) out of {available_history} available for this store-product pair, "
    f"followed by the next {selected_steps} forecasted period(s)."
)

callout(
    "Why the forecast may look smoother than history",
    "The forecast path represents expected demand under the latest known business context. "
    "Historical spikes can come from promotions, stock changes, events, weather shifts, or random demand noise. "
    "If future promotion, replenishment, event, or weather plans are available, they should be added as future scenario inputs."
)

section_title(
    "8. Forecast path table",
    f"Detailed values for the next {selected_steps} forecasted period(s)."
)

path_table = path_df[["horizon", "forecast_date", "predicted_demand"]].copy()
path_table.columns = ["Periods ahead", "Forecast date", "Forecasted demand"]
st.dataframe(path_table, use_container_width=True, hide_index=True)

section_title(
    "9. Business interpretation",
    "How to interpret the result for planning."
)

left, right = st.columns(2)

with left:
    callout(
        "Demand signal",
        f"Across the selected {selected_steps}-period horizon, the projected average is {horizon_avg:.2f}, "
        f"while the recent 7-period average is {recent_avg_7:.2f}."
    )

with right:
    if float(latest["in_stock_ratio"]) < 0.35 or float(latest["stockout_hours"]) >= 10:
        callout(
            "Operational note",
            "Availability looks constrained. Before reading low sales as weak demand, check replenishment and shelf availability."
        )
    elif avg_vs_recent > 10:
        callout(
            "Operational note",
            "Demand is trending above the recent average. This scenario may need closer replenishment planning."
        )
    elif avg_vs_recent < -10:
        callout(
            "Operational note",
            "Demand is trending below the recent average. Review whether this reflects lower demand, weaker activity, or reduced promotion support."
        )
    else:
        callout(
            "Operational note",
            "The projected demand is broadly in line with the recent average. No major shift is indicated from this scenario alone."
        )

st.sidebar.write("API Base URL:", API_BASE_URL)
