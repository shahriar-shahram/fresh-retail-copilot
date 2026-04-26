import streamlit as st
from ui import inject_css, sidebar_guide, hero, section_title, product_card, callout

st.set_page_config(page_title="Features | Fresh Retail Copilot", page_icon="🧱", layout="wide")
inject_css()
sidebar_guide("Features")

hero(
    "Features",
    "Here we build a cleaner demand signal by combining sales, availability, calendar, promotion, and weather context.",
    badges=["Availability", "Lost sales", "Forecasting", "Business readout"],
    eyebrow="Demand modeling",
)

section_title("Core capabilities")

row1 = st.columns(3, gap="medium")
with row1[0]:
    product_card(
        "🛒 Availability-aware demand",
        "We use stock signals so low sales are not automatically treated as weak demand."
    )
with row1[1]:
    product_card(
        "📉 Missed demand",
        "We estimate where demand may have existed but was not observed because the item was unavailable."
    )
with row1[2]:
    product_card(
        "🔮 Next-demand forecast",
        "We send engineered features to the model and return a forecast for the next step."
    )

row2 = st.columns(3, gap="medium")
with row2[0]:
    product_card(
        "📊 Store-product view",
        "We inspect one store and product at a time to keep the analysis explainable."
    )
with row2[1]:
    product_card(
        "🧠 Feature payload",
        "Lag, rolling, stockout, availability, promotion, calendar, and weather signals are used together."
    )
with row2[2]:
    product_card(
        "☁️ Served inference",
        "The forecast is called through a backend API instead of staying inside a local script."
    )

section_title("Problem → data science response")

left, right = st.columns(2, gap="large")
with left:
    callout(
        "Problem",
        "Observed sales are sometimes censored by inventory. The dataset may hide demand that could not be served."
    )
with right:
    callout(
        "Response",
        "We add availability context, estimate missed demand, and forecast using a cleaner target signal."
    )
