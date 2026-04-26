import streamlit as st

st.set_page_config(page_title="Fresh Retail Copilot", page_icon="🥬", layout="wide")

pages = [
    st.Page("overview.py", title="Overview", icon="🥬"),
    st.Page("pages/features.py", title="Features", icon="🧱"),
    st.Page("pages/workflow.py", title="Workflow", icon="🔁"),
    st.Page("pages/modeling.py", title="Modeling", icon="🧪"),
    st.Page("pages/data_explorer.py", title="Data Explorer", icon="📊"),
    st.Page("pages/forecast.py", title="Forecast", icon="🔮"),
    st.Page("pages/api_future.py", title="API & Future", icon="☁️"),
]

pg = st.navigation(pages, position="sidebar", expanded=True)
pg.run()
