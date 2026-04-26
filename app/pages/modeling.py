import streamlit as st
from ui import inject_css, sidebar_guide, hero, section_title, product_card, callout, flow_step

st.set_page_config(page_title="Modeling | Fresh Retail Copilot", page_icon="🧪", layout="wide")
inject_css()
sidebar_guide("Modeling")

hero(
    "Modeling",
    (
        "Here we explain how the forecast is built: first clean the demand signal, then train the model, "
        "then serve the trained model through the API."
    ),
    badges=["Training data", "Feature engineering", "Demand correction", "Forecast model", "API inference"],
    eyebrow="Forecasting method",
)

section_title("How the forecast is done")

steps = st.columns(5, gap="small")

with steps[0]:
    flow_step(1, "Raw records", "Start from sales, stock, promo, calendar, and weather data.")

with steps[1]:
    flow_step(2, "Availability check", "Identify rows where sales may be limited by stockouts.")

with steps[2]:
    flow_step(3, "Demand target", "Build a cleaner target using observed sales and recovered demand.")

with steps[3]:
    flow_step(4, "Train model", "Train the forecast model on engineered demand features.")

with steps[4]:
    flow_step(5, "Serve forecast", "Use the trained model through the backend API.")

section_title("What goes into the model")

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    product_card(
        "Sales history",
        "Lag features and rolling averages summarize recent demand behavior."
    )

with col2:
    product_card(
        "Availability context",
        "Stockout hours, in-stock ratio, and average sales when available help interpret low-sales periods."
    )

with col3:
    product_card(
        "External signals",
        "Discounts, holidays, activity flags, weather, weekday, and month add business context."
    )

section_title("Training idea")

left, right = st.columns(2, gap="large")

with left:
    callout(
        "Why not train only on raw sales?",
        "Raw sales may be biased when the product was unavailable. If the model learns directly from those rows, it may underforecast demand."
    )

with right:
    callout(
        "What we do instead",
        "We use availability-aware features and a corrected demand target so the model has better context before learning."
    )

section_title("Is the app using the trained model?")

st.markdown("""
Yes, the Streamlit app sends the selected scenario features to the backend prediction API.

The backend is responsible for loading the trained model artifact and returning the forecast.
The app then displays that returned value as the next-demand prediction.
""")

st.code(
    """
selected scenario
    ↓
engineered feature row
    ↓
POST /predict
    ↓
trained model inference
    ↓
predicted demand
    """,
    language="text",
)

section_title("Feature payload used for inference")

st.markdown("""
The forecast request is based on features such as:

- recent demand: `lag_1`, `lag_7`, `rolling_mean_7`
- availability: `stockout_hours`, `avg_sales_when_available`
- demand variability: `demand_std`
- business context: `discount`, `holiday_flag`, `activity_flag`
- weather: `precpt`, `avg_temperature`, `avg_humidity`, `avg_wind_level`
- calendar: `day_of_week`, `month`
""")
