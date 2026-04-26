import streamlit as st
from ui import inject_css, sidebar_guide, hero, section_title, muted, product_card, callout

inject_css()
sidebar_guide("Overview")

hero(
    "Fresh Retail Copilot",
    (
        "A data-driven retail forecasting workflow for fresh retail demand planning. "
        "The main idea is simple: low sales do not always mean low demand."
    ),
    badges=["Demand forecasting", "Stockouts", "Lost sales", "API-backed app"],
    eyebrow="Retail forecasting workflow",
)

callout(
    "Project definition",
    "Fresh Retail Copilot builds a cleaner demand signal by accounting for stock availability before forecasting demand."
)

section_title("Why this exists")

left, right = st.columns(2, gap="large")
with left:
    product_card(
        "The retail problem",
        "Fresh products can be unavailable during the day. When that happens, sales data only shows what was sold, not what customers wanted."
    )
with right:
    product_card(
        "The data science goal",
        "Use availability, sales history, and context features to estimate demand more carefully."
    )

section_title("What the workflow answers")

a, b, c = st.columns(3, gap="medium")
with a:
    product_card("Was demand actually low?", "Check whether low sales came with low availability.")
with b:
    product_card("How much demand was missed?", "Estimate possible lost sales during constrained periods.")
with c:
    product_card("What should we expect next?", "Run a forecast from the latest engineered feature row.")

section_title("Explore")

n1, n2, n3 = st.columns(3, gap="medium")
with n1:
    product_card("🧱 Features", "Demand signals, stockout logic, and forecast inputs.")
with n2:
    product_card("🔁 Workflow", "Visual pipeline from data to prediction.")
with n3:
    product_card("📊 Forecast pages", "Explore data and run the API-backed forecast.")

muted(
    "The app connects demand modeling, availability analysis, API inference, and a simple decision-facing interface."
)
