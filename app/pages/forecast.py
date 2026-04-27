import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

from ui import inject_css, sidebar_guide, hero, section_title, callout

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://fresh-retail-copilot-api-837696130499.us-central1.run.app",
)

HORIZON_ORDER = [
    ("1_day_ahead", "1 day ahead"),
    ("1_week_ahead", "1 week ahead"),
    ("1_month_ahead", "1 month ahead"),
]

HORIZON_LABELS = dict(HORIZON_ORDER)

INT_FEATURES = {
    "city_id",
    "store_id",
    "management_group_id",
    "first_category_id",
    "second_category_id",
    "third_category_id",
    "product_id",
    "holiday_flag",
    "activity_flag",
    "day_of_week",
    "month",
    "day_of_month",
    "week_of_year",
    "is_weekend",
}

st.set_page_config(page_title="Forecast | Fresh Retail Copilot", page_icon="🔮", layout="wide")
inject_css()
sidebar_guide("Forecast")

hero(
    "🔮 Forecast",
    (
        "This page generates three horizon-specific demand forecasts for the selected store-product pair: "
        "1 day ahead, 1 week ahead, and 1 month ahead."
    ),
    badges=["1 day ahead", "1 week ahead", "1 month ahead", "Horizon-specific models"],
    eyebrow="Demand forecast workspace",
)


@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_parquet("data/processed/app_forecast_scenarios_rich.parquet")
    df["dt"] = pd.to_datetime(df["dt"])
    return df.sort_values(["store_id", "product_id", "dt"]).reset_index(drop=True)


@st.cache_data(ttl=300, show_spinner=False)
def load_business_models():
    r = requests.get(f"{API_BASE_URL}/business-horizon-models", timeout=20)
    r.raise_for_status()
    return r.json()


def make_payload(row, features, horizon_key):
    payload = {}
    for feature in features:
        if feature in INT_FEATURES:
            payload[feature] = int(row[feature])
        else:
            payload[feature] = float(row[feature])
    payload["horizon"] = horizon_key
    return payload


def predict_business_horizon(payload):
    r = requests.post(
        f"{API_BASE_URL}/predict-business-horizon",
        json=payload,
        timeout=25,
    )
    r.raise_for_status()
    return r.json()


def pct_change(value, ref):
    if abs(ref) < 1e-9:
        return 0.0
    return 100.0 * (value - ref) / ref


df = load_data()

try:
    bundle = load_business_models()
except Exception as e:
    st.error(f"Could not load business-horizon models from API: {e}")
    st.info(f"Check API_BASE_URL: {API_BASE_URL}")
    st.stop()

registry = bundle.get("registry", {})
features = bundle.get("features", [])

if not registry or not features:
    st.error("Business-horizon registry or feature list is missing from the API response.")
    st.stop()

section_title(
    "1. Forecast setup",
    "Each point forecast comes from a model trained specifically for that horizon."
)

setup_cols = st.columns(3)
for col, (hkey, hname) in zip(setup_cols, HORIZON_ORDER):
    meta = registry.get(hkey, {})
    with col:
        st.markdown(f"### {hname}")
        st.metric("Model", meta.get("selected_model", "N/A"))
        st.caption(
            f"MAE: {float(meta.get('mae', 0.0)):.4f} | "
            f"RMSE: {float(meta.get('rmse', 0.0)):.4f} | "
            f"SMAPE: {float(meta.get('smape', 0.0)):.2f}"
        )

callout(
    "What this means",
    "These are direct point forecasts, not an interpolated forecast path. "
    "The 1-day model predicts demand exactly 1 period ahead, the 1-week model predicts exactly 7 periods ahead, "
    "and the 1-month model predicts exactly 30 periods ahead."
)

st.markdown("---")

section_title(
    "2. Choose the product scenario",
    "Select one of the available demo store-product scenarios. The page will generate all three planning-horizon forecasts together."
)

demo_store_count = df["store_id"].nunique()
demo_product_count = df["product_id"].nunique()
demo_pair_count = df[["store_id", "product_id"]].drop_duplicates().shape[0]

s1, s2, s3 = st.columns(3)
s1.metric("Demo stores", f"{demo_store_count:,}")
s2.metric("Demo products", f"{demo_product_count:,}")
s3.metric("Demo store-product pairs", f"{demo_pair_count:,}")

callout(
    "Demo scenario coverage",
    "This deployed app uses a curated scenario dataset so the live demo stays fast and lightweight. "
    "The forecasting API itself is designed to score any store-product row that has the required forecast-ready features."
)

c1, c2, c3 = st.columns(3)

with c1:
    stores = sorted(df["store_id"].unique().tolist())
    selected_store = st.selectbox("Store", stores)

with c2:
    product_candidates = sorted(
        df.loc[df["store_id"] == selected_store, "product_id"].unique().tolist()
    )
    selected_product = st.selectbox("Product", product_candidates)

with c3:
    history_window = st.selectbox(
        "Historical window shown",
        [14, 30, 60, 90],
        index=3,
        help="How many most recent historical records to show before the three forecast points.",
    )

subset = df[
    (df["store_id"] == selected_store) &
    (df["product_id"] == selected_product)
].sort_values("dt").reset_index(drop=True)

if subset.empty:
    st.error("No rows found for this store-product combination.")
    st.stop()

recent = subset.tail(history_window).copy()
latest = subset.iloc[-1].copy()
last_date = pd.to_datetime(latest["dt"])

recent_avg_7 = float(subset.tail(7)["sale_amount"].mean()) if len(subset) >= 7 else float(subset["sale_amount"].mean())
latest_sales = float(latest["sale_amount"])

section_title(
    "3. What these forecasts represent",
    "Each forecast point answers a different business question."
)

d1, d2, d3 = st.columns(3)
with d1:
    st.markdown("### 1 day ahead")
    st.write(
        "Estimated demand for the **next period after the latest observed row**. "
        "Useful for short-term replenishment and near-term operational planning."
    )

with d2:
    st.markdown("### 1 week ahead")
    st.write(
        "Estimated demand for the row **7 periods after the latest observed row**. "
        "Useful for short-horizon ordering and weekly inventory planning."
    )

with d3:
    st.markdown("### 1 month ahead")
    st.write(
        "Estimated demand for the row **30 periods after the latest observed row**. "
        "Useful for medium-horizon planning, allocation, and forward inventory positioning."
    )

callout(
    "How the point is produced",
    "For each horizon, the model uses the latest available feature row for this store-product pair — including demand history, "
    "rolling statistics, stock availability, calendar, weather, and product/store identifiers — and predicts demand at that exact future horizon."
)

section_title(
    "4. Current context",
    "Recent observed demand and availability conditions for the selected scenario."
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Latest observed sales", f"{latest_sales:.2f}")
m2.metric("Recent 7-period average", f"{recent_avg_7:.2f}")
m3.metric("In-stock ratio", f"{float(latest['in_stock_ratio']):.2f}")
m4.metric("Stockout hours", f"{float(latest['stockout_hours']):.0f}")

if float(latest["in_stock_ratio"]) < 0.35 or float(latest["stockout_hours"]) >= 10:
    st.warning(
        "Availability looks constrained in the latest row. "
        "Observed sales may understate true demand if the product was not fully available."
    )
else:
    st.success(
        "Availability looks more stable in the latest row, so observed sales are less likely to be heavily constrained by stockout."
    )

with st.expander("Features used by the deployed forecasting models"):
    feature_groups = pd.DataFrame(
        [
            {"Feature group": "Product and store identity", "Examples": "city_id, store_id, management_group_id, category ids, product_id"},
            {"Feature group": "Lag demand", "Examples": "lag_1, lag_2, lag_3, lag_7, lag_14, lag_21, lag_28"},
            {"Feature group": "Rolling demand statistics", "Examples": "rolling_mean_7/14/28, rolling_std_7/14/28, rolling_min/max"},
            {"Feature group": "Availability", "Examples": "stock_hour6_22_cnt, in_stock_ratio, stockout_hours, avg_sales_when_available"},
            {"Feature group": "Business context", "Examples": "discount, holiday_flag, activity_flag"},
            {"Feature group": "Calendar and weather", "Examples": "day_of_week, week_of_year, month, day_of_month, is_weekend, precipitation, temperature, humidity, wind"},
        ]
    )
    st.dataframe(feature_groups, use_container_width=True, hide_index=True)

section_title("5. Run forecasts")

if not st.button("Generate forecasts", use_container_width=True):
    st.info("Click the button to generate the 1-day, 1-week, and 1-month forecasts.")
    st.stop()

results = []
try:
    for hkey, hname in HORIZON_ORDER:
        payload = make_payload(latest, features, hkey)
        out = predict_business_horizon(payload)
        horizon_steps = int(out["horizon_steps"])
        forecast_date = last_date + pd.Timedelta(days=horizon_steps)
        pred = float(out["predicted_demand"])

        results.append(
            {
                "horizon_key": hkey,
                "Forecast horizon": hname,
                "Forecast date": forecast_date,
                "Predicted demand": pred,
                "Model": out["model_name"],
                "Model class": out["model_class"],
                "MAE": float(out["mae"]),
                "RMSE": float(out["rmse"]),
                "SMAPE": float(out["smape"]),
                "Horizon steps": horizon_steps,
                "Change vs recent avg (%)": pct_change(pred, recent_avg_7),
            }
        )
except Exception as e:
    st.error(f"API request failed: {e}")
    st.info(f"Check API_BASE_URL: {API_BASE_URL}")
    st.stop()

results_df = pd.DataFrame(results).sort_values("Horizon steps").reset_index(drop=True)

section_title(
    "6. Forecast summary",
    "These are the three horizon-specific point forecasts generated for the selected scenario."
)

sum_cols = st.columns(3)
for col, row in zip(sum_cols, results_df.to_dict("records")):
    with col:
        st.markdown(f"### {row['Forecast horizon']}")
        st.metric("Predicted demand", f"{row['Predicted demand']:.2f} units")
        st.caption(f"Forecast date: {pd.to_datetime(row['Forecast date']).date()}")
        st.caption(f"Model: {row['Model']}")
        st.caption(
            f"MAE: {row['MAE']:.4f} | RMSE: {row['RMSE']:.4f} | SMAPE: {row['SMAPE']:.2f}"
        )

section_title(
    "7. Forecast visualization",
    "The line is historical observed demand. The three markers are the actual horizon-specific forecasts."
)

history_plot = recent[["dt", "sale_amount"]].copy()
history_plot["dt"] = pd.to_datetime(history_plot["dt"])
history_plot = history_plot.rename(columns={"sale_amount": "value"})
history_plot["series"] = "Observed demand"

forecast_plot = results_df[["Forecast horizon", "Forecast date", "Predicted demand"]].copy()
forecast_plot = forecast_plot.rename(
    columns={"Forecast date": "dt", "Predicted demand": "value"}
)
forecast_plot["dt"] = pd.to_datetime(forecast_plot["dt"])

connector_rows = []
for row in results_df.to_dict("records"):
    connector_rows.append({"dt": last_date, "value": latest_sales, "Forecast horizon": row["Forecast horizon"]})
    connector_rows.append({"dt": row["Forecast date"], "value": row["Predicted demand"], "Forecast horizon": row["Forecast horizon"]})
connector_plot = pd.DataFrame(connector_rows)

chart_start = pd.to_datetime(history_plot["dt"].min())
chart_end = pd.to_datetime(max(history_plot["dt"].max(), forecast_plot["dt"].max()))

history_line = alt.Chart(history_plot).mark_line(strokeWidth=3).encode(
    x=alt.X(
        "dt:T",
        title="Date",
        scale=alt.Scale(domain=[chart_start, chart_end]),
    ),
    y=alt.Y("value:Q", title="Demand (sale_amount units)"),
    tooltip=[
        alt.Tooltip("dt:T", title="Date"),
        alt.Tooltip("value:Q", title="Observed demand", format=".2f"),
    ],
)

connector_line = alt.Chart(connector_plot).mark_line(strokeDash=[6, 4], strokeWidth=2).encode(
    x=alt.X("dt:T", scale=alt.Scale(domain=[chart_start, chart_end])),
    y=alt.Y("value:Q"),
    detail="Forecast horizon:N",
    color=alt.Color("Forecast horizon:N", title="Forecast horizon"),
    tooltip=[
        alt.Tooltip("Forecast horizon:N", title="Forecast horizon"),
        alt.Tooltip("dt:T", title="Date"),
        alt.Tooltip("value:Q", title="Demand", format=".2f"),
    ],
)

forecast_points = alt.Chart(forecast_plot).mark_point(filled=True, size=180).encode(
    x=alt.X("dt:T", scale=alt.Scale(domain=[chart_start, chart_end])),
    y=alt.Y("value:Q"),
    color=alt.Color("Forecast horizon:N", title="Forecast horizon"),
    shape=alt.Shape("Forecast horizon:N", title="Forecast horizon"),
    tooltip=[
        alt.Tooltip("Forecast horizon:N", title="Forecast horizon"),
        alt.Tooltip("dt:T", title="Forecast date"),
        alt.Tooltip("value:Q", title="Predicted demand", format=".2f"),
    ],
)

boundary_rule = alt.Chart(pd.DataFrame({"dt": [last_date]})).mark_rule(strokeDash=[2, 2]).encode(
    x=alt.X("dt:T", scale=alt.Scale(domain=[chart_start, chart_end]))
)

chart = (history_line + connector_line + forecast_points + boundary_rule).properties(height=430)
st.altair_chart(chart, use_container_width=True)

st.caption(
    f"Showing the latest {len(recent)} observed records and the three deployed point forecasts. "
    "There is no interpolation between forecast dates."
)

callout(
    "How to read this chart",
    "The historical line shows observed demand. "
    "The forecast markers are the actual predictions for D+1, D+7, and D+30. "
    "The dashed lines only connect the last observed point to each forecast point for readability. "
    "They do not imply predicted values for the dates in between."
)

section_title(
    "8. Forecast results table",
    "This table reports the forecasted value and the validation metrics of the model used for each horizon."
)

display_df = results_df.copy()
display_df["Forecast date"] = pd.to_datetime(display_df["Forecast date"]).dt.date
display_df["Predicted demand"] = display_df["Predicted demand"].map(lambda x: round(float(x), 3))
display_df["MAE"] = display_df["MAE"].map(lambda x: round(float(x), 4))
display_df["RMSE"] = display_df["RMSE"].map(lambda x: round(float(x), 4))
display_df["SMAPE"] = display_df["SMAPE"].map(lambda x: round(float(x), 2))
display_df["Change vs recent avg (%)"] = display_df["Change vs recent avg (%)"].map(lambda x: round(float(x), 1))

st.dataframe(
    display_df[
        [
            "Forecast horizon",
            "Forecast date",
            "Predicted demand",
            "Model",
            "Model class",
            "MAE",
            "RMSE",
            "SMAPE",
            "Change vs recent avg (%)",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)

section_title(
    "9. Interpretation",
    "Here is how to interpret the three forecasts from an operational point of view."
)

for row in results_df.to_dict("records"):
    pred = row["Predicted demand"]
    delta_pct = row["Change vs recent avg (%)"]

    if float(latest["in_stock_ratio"]) < 0.35 or float(latest["stockout_hours"]) >= 10:
        guidance = (
            "Recent availability appears constrained, so low recent sales may not fully reflect true demand. "
            "Use this forecast together with replenishment and stock review."
        )
    elif delta_pct > 10:
        guidance = (
            "Demand is above the recent short-term average. This horizon may call for closer replenishment planning."
        )
    elif delta_pct < -10:
        guidance = (
            "Demand is below the recent short-term average. This may indicate softer demand or weaker near-term pull."
        )
    else:
        guidance = (
            "Demand is close to the recent short-term average. This suggests a more stable near-term outlook."
        )

    st.markdown(f"### {row['Forecast horizon']}")
    st.write(
        f"Forecast date: **{pd.to_datetime(row['Forecast date']).date()}**  \n"
        f"Predicted demand: **{pred:.2f} units**  \n"
        f"Model used: **{row['Model']}**  \n"
        f"Change vs recent 7-period average: **{delta_pct:+.1f}%**"
    )
    st.write(guidance)

st.sidebar.write("API Base URL:", API_BASE_URL)
