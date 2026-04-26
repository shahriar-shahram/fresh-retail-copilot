import streamlit as st
from ui import inject_css, sidebar_guide, hero, section_title, muted, product_card, callout
import os
from pathlib import Path

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://fresh-retail-copilot-api-837696130499.us-central1.run.app"
)

ASSET_DIR = Path("app/assets")

st.set_page_config(page_title="API and Future | Fresh Retail Copilot", page_icon="☁️", layout="wide")
inject_css()
sidebar_guide("API and Future")

hero(
    "☁️ API and Future",
    "How the forecasting workflow is served and extended.",
    badges=["FastAPI", "Docker", "Cloud API", "Streamlit", "User datasets"],
)

section_title("System architecture", "The frontend handles exploration. The backend handles prediction.")

cloud_path = ASSET_DIR / "cloud_architecture.png"
if cloud_path.exists():
    st.image(str(cloud_path), caption="Backend and frontend architecture", use_container_width=True)
else:
    st.warning("Missing app/assets/cloud_architecture.png")

section_title("Backend highlights")

col1, col2, col3 = st.columns(3, gap="medium")
with col1:
    product_card("FastAPI backend", "Serves demand forecasts through an API endpoint.")
with col2:
    product_card("Docker packaging", "Packages the backend so the inference service is easier to run and deploy.")
with col3:
    product_card("Streamlit frontend", "Gives users a simple way to explore scenarios and read the forecast.")

section_title("Live API")

st.code(API_BASE_URL)

st.caption(
    "Scenario features go to the backend. The backend returns the demand forecast."
)

section_title("Future direction")

st.markdown("""
The next version can let a user upload their own sales and inventory data.

The app could then inspect the columns and suggest what can be done, for example:

- demand forecasting
- stockout detection
- lost-sales estimation
- store-product diagnostics
- promotion analysis
- inventory recommendations

That would move this from a fixed demo to a more flexible retail analytics copilot.
""")
