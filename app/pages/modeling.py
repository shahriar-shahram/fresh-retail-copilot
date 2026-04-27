import json
from pathlib import Path

import pandas as pd
import streamlit as st

from ui import inject_css, sidebar_guide, hero, section_title, product_card, callout, flow_step

st.set_page_config(page_title="Modeling | Fresh Retail Copilot", page_icon="🧪", layout="wide")
inject_css()
sidebar_guide("Modeling")

REGISTRY_PATH = Path("models/business_horizon_model_registry.json")
METADATA_PATH = Path("models/business_horizon_metadata.json")
COMPARISON_PATH = Path("results/models/business_horizon_model_comparison.csv")

hero(
    "Modeling",
    (
        "The forecasting system is trained around three business planning horizons: "
        "1 day ahead, 1 week ahead, and 1 month ahead. Each horizon has its own model-selection study."
    ),
    badges=["CatBoost", "LightGBM", "XGBoost", "Horizon-specific models", "Rich feature set"],
    eyebrow="Forecast model study",
)

section_title("Modeling workflow")

steps = st.columns(5, gap="small")
with steps[0]:
    flow_step(1, "Raw data", "Store, product, sales, stock, promo, weather.")
with steps[1]:
    flow_step(2, "Features", "IDs, lags, rolling stats, calendar, context.")
with steps[2]:
    flow_step(3, "Targets", "1 day, 1 week, 1 month ahead.")
with steps[3]:
    flow_step(4, "Compare", "Baselines and boosted models.")
with steps[4]:
    flow_step(5, "Deploy", "Best model per horizon.")

section_title("Selected production models")

if REGISTRY_PATH.exists():
    with open(REGISTRY_PATH, "r") as f:
        registry = json.load(f)

    c1, c2, c3 = st.columns(3)

    horizon_order = [
        ("1_day_ahead", "1 day ahead"),
        ("1_week_ahead", "1 week ahead"),
        ("1_month_ahead", "1 month ahead"),
    ]

    for col, (key, label) in zip([c1, c2, c3], horizon_order):
        meta = registry.get(key, {})
        with col:
            product_card(
                label,
                (
                    f"Selected model: **{meta.get('selected_model', 'N/A')}**  \n"
                    f"MAE: `{meta.get('mae', 0):.4f}`  \n"
                    f"RMSE: `{meta.get('rmse', 0):.4f}`  \n"
                    f"SMAPE: `{meta.get('smape', 0):.2f}`"
                ),
            )

    callout(
        "Why separate models?",
        (
            "Short-term and longer-horizon forecasts behave differently. "
            "A model that works best for tomorrow is not always the best model for a month-ahead forecast, "
            "so the app selects the best model separately for each planning horizon."
        ),
    )
else:
    st.warning("Business horizon registry not found. Run `python scripts/train_business_horizon_models.py` first.")

section_title("Model comparison")

if COMPARISON_PATH.exists():
    comparison = pd.read_csv(COMPARISON_PATH)

    display_cols = [
        "horizon_label",
        "model",
        "model_class",
        "mae",
        "rmse",
        "smape",
        "rank_by_mae",
        "train_rows",
        "validation_rows",
    ]

    available_cols = [c for c in display_cols if c in comparison.columns]
    st.dataframe(comparison[available_cols], use_container_width=True, hide_index=True)

    st.caption(
        "Lower MAE, RMSE, and SMAPE are better. The deployed model for each horizon is the best trainable model by validation MAE."
    )
else:
    st.warning("Model comparison file not found.")

section_title("Feature groups")

f1, f2, f3 = st.columns(3, gap="medium")
with f1:
    product_card(
        "Identity features",
        "`city_id`, `store_id`, product/category IDs"
    )
with f2:
    product_card(
        "Demand history",
        "`lag_1`, `lag_2`, `lag_3`, `lag_7`, `lag_14`, `lag_21`, `lag_28`"
    )
with f3:
    product_card(
        "Rolling demand",
        "Rolling mean, std, min, and max over 7, 14, and 28 periods"
    )

f4, f5, f6 = st.columns(3, gap="medium")
with f4:
    product_card(
        "Availability",
        "`stock_hour6_22_cnt`, `in_stock_ratio`, `stockout_hours`, sales when available"
    )
with f5:
    product_card(
        "Business context",
        "`discount`, `holiday_flag`, `activity_flag`"
    )
with f6:
    product_card(
        "Weather and calendar",
        "Temperature, humidity, wind, precipitation, day/week/month features"
    )

section_title("Baselines included")

callout(
    "Baseline checks",
    (
        "The study compares ML models against simple forecasting rules such as last observed value, "
        "7-period average, seasonal lag-7, seasonal lag-28, and naive mean. "
        "This makes the model improvement easier to defend."
    ),
)

section_title("Deployment note")

st.success(
    "The API exposes `/business-horizon-models` and `/predict-business-horizon`. "
    "The Forecast page uses the selected duration to call the correct horizon-specific model."
)
