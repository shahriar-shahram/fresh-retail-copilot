import streamlit as st
from pathlib import Path
from ui import inject_css, sidebar_guide, hero, section_title, product_card, callout, flow_step

st.set_page_config(page_title="Workflow | Fresh Retail Copilot", page_icon="🔁", layout="wide")
inject_css()
sidebar_guide("Workflow")

ASSET_DIR = Path("app/assets")

hero(
    "Workflow",
    "Here we move from raw retail records to a forecast that can be called from an app.",
    badges=["Data", "Features", "Recovery", "API", "Dashboard"],
    eyebrow="Forecasting pipeline",
)

section_title("Pipeline")

steps = st.columns(5, gap="small")
with steps[0]:
    flow_step(1, "Data", "Sales, stock, calendar, promo, weather.")
with steps[1]:
    flow_step(2, "Features", "Lag, rolling, stockout, availability.")
with steps[2]:
    flow_step(3, "Recovery", "Estimate demand beyond raw sales.")
with steps[3]:
    flow_step(4, "Forecast", "Predict next demand.")
with steps[4]:
    flow_step(5, "App", "Show result and readout.")

flow_path = ASSET_DIR / "retail_flow.png"
if flow_path.exists():
    st.image(str(flow_path), caption="Demand forecasting workflow", use_container_width=True)
else:
    st.warning("Missing app/assets/retail_flow.png")

section_title("Stockout effect")

left, right = st.columns([1.15, 0.85], gap="large", vertical_alignment="center")
with left:
    stockout_path = ASSET_DIR / "stockout_concept.png"
    if stockout_path.exists():
        st.image(str(stockout_path), caption="Observed sales can be below true demand.", use_container_width=True)
    else:
        st.warning("Missing app/assets/stockout_concept.png")

with right:
    product_card(
        "Why this matters",
        "A low-sales period is not always a low-demand period. Availability changes how we should read the target variable."
    )
    st.write("")
    callout(
        "Modeling idea",
        "Before forecasting, we check whether the sales signal is reliable or constrained by stockouts."
    )
