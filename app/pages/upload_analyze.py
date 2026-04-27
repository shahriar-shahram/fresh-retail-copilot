import streamlit as st
import pandas as pd
import numpy as np
from ui import inject_css, sidebar_guide, hero, section_title, product_card, callout

st.set_page_config(page_title="Upload and Analyze | Fresh Retail Copilot", page_icon="📤", layout="wide")
inject_css()
sidebar_guide("Upload and Analyze")


def read_uploaded_file(uploaded_file):
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)

    raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")


def infer_columns(df: pd.DataFrame):
    cols = list(df.columns)
    lower_map = {c: str(c).lower() for c in cols}

    def find_first(keywords):
        for col, low in lower_map.items():
            if any(k in low for k in keywords):
                return col
        return None

    return {
        "date": find_first(["date", "dt", "day", "time", "timestamp"]),
        "sales": find_first(["sale", "sales", "units", "qty", "quantity", "demand", "amount"]),
        "product": find_first(["product", "item", "sku", "article"]),
        "store": find_first(["store", "shop", "location", "branch"]),
        "stock": find_first(["stock", "inventory", "available", "availability", "in_stock"]),
        "discount": find_first(["discount", "promo", "promotion", "price_cut"]),
        "price": find_first(["price", "unit_price", "selling_price"]),
        "weather": find_first(["weather", "temperature", "temp", "humidity", "precip", "rain", "wind"]),
    }


def build_data_quality_table(df: pd.DataFrame):
    return pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "missing_values": [int(df[c].isna().sum()) for c in df.columns],
        "missing_pct": [float(df[c].isna().mean() * 100) for c in df.columns],
        "unique_values": [int(df[c].nunique(dropna=True)) for c in df.columns],
    }).sort_values("missing_pct", ascending=False)


hero(
    "Upload and Analyze",
    (
        "This page is for user-provided retail data. Before uploading, the user should know what columns are needed "
        "and what kind of analysis is possible from their file."
    ),
    badges=["CSV", "Excel", "Required schema", "Data quality", "Forecast readiness"],
    eyebrow="Bring your own data",
)


section_title("Forecast-ready data requirements")

st.markdown("""
To run the same type of forecast on a client dataset, the data should support three business horizons:

- **1 day ahead**
- **1 week ahead**
- **1 month ahead**

The strongest version of the model uses product/store identity, demand history, stock availability, promotion/activity signals, weather, and calendar features.
""")

required_cols = [
    {"Group": "Identity", "Examples": "city_id, store_id, product_id, category IDs"},
    {"Group": "Demand", "Examples": "date/time column, sale_amount or demand quantity"},
    {"Group": "Availability", "Examples": "stock count, in-stock ratio, stockout hours, inventory status"},
    {"Group": "Business context", "Examples": "discount, promotion flag, holiday flag, activity/event flag"},
    {"Group": "Weather", "Examples": "temperature, humidity, wind, precipitation"},
    {"Group": "Calendar", "Examples": "date column used to create weekday, month, week-of-year"},
]

st.dataframe(required_cols, use_container_width=True, hide_index=True)

callout(
    "How uploaded data would be handled",
    "The app would first map uploaded columns into a forecast-ready schema, generate lag and rolling features, "
    "then use the appropriate horizon model for 1-day, 1-week, or 1-month demand forecasting."
)

section_title("Required data format")

st.markdown("""
The uploaded file should be a **table**, where each row is one sales observation.

A row can represent one product on one day, one product in one store on one day, or a more detailed hourly record.
The exact level is flexible, but the file must have a clear time column and a sales or demand column.
""")

req1, req2, req3 = st.columns(3, gap="medium")

with req1:
    product_card(
        "Required: date/time",
        "A column such as `date`, `dt`, `day`, `timestamp`, or `week`. This tells the app the order of observations."
    )

with req2:
    product_card(
        "Required: sales/demand",
        "A column such as `sales`, `sale_amount`, `units_sold`, `quantity`, or `demand`. This is the target signal."
    )

with req3:
    product_card(
        "Recommended: product/store",
        "Columns such as `product_id`, `sku`, `item`, `store_id`, or `location` allow product-level and store-level analysis."
    )

section_title("Recommended columns")

st.markdown("""
The more context the file has, the better the analysis can be.
""")

schema_rows = [
    {
        "Column type": "Date or time",
        "Example names": "date, dt, day, timestamp, week",
        "Required?": "Required",
        "Why it matters": "Creates the time order for forecasting.",
    },
    {
        "Column type": "Sales or demand",
        "Example names": "sales, sale_amount, units_sold, quantity, demand",
        "Required?": "Required",
        "Why it matters": "This is the target the model tries to understand or forecast.",
    },
    {
        "Column type": "Product identifier",
        "Example names": "product_id, sku, item_id, product_name",
        "Required?": "Recommended",
        "Why it matters": "Allows product-level demand analysis.",
    },
    {
        "Column type": "Store or location",
        "Example names": "store_id, store, location, branch",
        "Required?": "Recommended",
        "Why it matters": "Allows store-level demand analysis.",
    },
    {
        "Column type": "Stock or availability",
        "Example names": "stock, inventory, in_stock_ratio, availability, stockout_hours",
        "Required?": "Needed for stockout-aware analysis",
        "Why it matters": "Helps separate weak demand from constrained sales.",
    },
    {
        "Column type": "Price, discount, or promotion",
        "Example names": "price, discount, promo_flag, promotion",
        "Required?": "Optional",
        "Why it matters": "Helps explain demand changes caused by business actions.",
    },
    {
        "Column type": "Calendar flags",
        "Example names": "holiday_flag, event_flag, activity_flag",
        "Required?": "Optional",
        "Why it matters": "Captures known demand shifts around holidays or events.",
    },
    {
        "Column type": "Weather",
        "Example names": "temperature, humidity, rain, wind, weather",
        "Required?": "Optional",
        "Why it matters": "Useful for fresh products affected by weather.",
    },
]

st.dataframe(pd.DataFrame(schema_rows), use_container_width=True, hide_index=True)

section_title("What the app can do based on available columns")

c1, c2, c3 = st.columns(3, gap="medium")

with c1:
    product_card(
        "Basic forecasting",
        "Needs at least a date/time column and a sales/demand column."
    )

with c2:
    product_card(
        "Product or store analysis",
        "Needs product and/or store identifiers."
    )

with c3:
    product_card(
        "Stockout-aware forecasting",
        "Needs stock, inventory, availability, or stockout-related columns."
    )

callout(
    "Important",
    "If the file does not include stock or availability information, the app can still inspect sales trends, but it cannot honestly estimate stockout-driven lost sales."
)

section_title("Example acceptable structure")

example_df = pd.DataFrame({
    "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
    "store_id": ["S001", "S001", "S001"],
    "product_id": ["P100", "P100", "P100"],
    "sales": [18, 5, 22],
    "stock_available": [1, 0, 1],
    "discount": [0.10, 0.00, 0.15],
    "holiday_flag": [1, 0, 0],
})

st.dataframe(example_df, use_container_width=True, hide_index=True)

st.caption(
    "The column names do not have to match exactly. The app tries to detect common names, and a later version can let the user map columns manually."
)

section_title("Upload file")

uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file",
    type=["csv", "xlsx", "xls"],
)

if uploaded_file is None:
    st.info("Upload a file when ready. The app will inspect the schema first before trying any forecasting.")
    st.stop()

try:
    df = read_uploaded_file(uploaded_file)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

if df.empty:
    st.error("The uploaded file is empty.")
    st.stop()

section_title("Dataset overview")

rows, cols = df.shape
inferred = infer_columns(df)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows", f"{rows:,}")
m2.metric("Columns", f"{cols:,}")
m3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
m4.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

st.dataframe(df.head(50), use_container_width=True)

section_title("Detected column roles")

role_df = pd.DataFrame(
    [{"role": role, "detected_column": col if col is not None else "Not detected"} for role, col in inferred.items()]
)
st.dataframe(role_df, use_container_width=True, hide_index=True)

section_title("Readiness check")

has_date = inferred["date"] is not None
has_sales = inferred["sales"] is not None
has_stock = inferred["stock"] is not None
has_product = inferred["product"] is not None
has_store = inferred["store"] is not None

r1, r2, r3 = st.columns(3, gap="medium")

with r1:
    if has_date and has_sales:
        product_card("✅ Basic forecast ready", "Date and sales/demand columns were detected.")
    else:
        product_card("⚠️ Basic forecast not ready", "The file needs both date/time and sales/demand columns.")

with r2:
    if has_product or has_store:
        product_card("✅ Segmented analysis ready", "Product or store identifiers were detected.")
    else:
        product_card("⚠️ Segmentation limited", "No clear product or store identifier was detected.")

with r3:
    if has_stock:
        product_card("✅ Stockout analysis ready", "Stock or availability information was detected.")
    else:
        product_card("⚠️ Stockout analysis limited", "No stock or availability column was detected.")

section_title("Data quality")

quality = build_data_quality_table(df)
st.dataframe(quality, use_container_width=True, hide_index=True)

section_title("Quick demand trend")

if has_date and has_sales:
    date_col = inferred["date"]
    sales_col = inferred["sales"]

    plot_df = df[[date_col, sales_col]].copy()
    plot_df[date_col] = pd.to_datetime(plot_df[date_col], errors="coerce")
    plot_df[sales_col] = pd.to_numeric(plot_df[sales_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[date_col, sales_col]).sort_values(date_col)

    if not plot_df.empty:
        trend = plot_df.groupby(date_col, as_index=False)[sales_col].sum()
        st.line_chart(trend.set_index(date_col)[[sales_col]])
    else:
        st.warning("The detected date and sales columns could not be converted cleanly for plotting.")
else:
    st.info("A demand trend needs both date/time and sales/demand columns.")

section_title("What can be solved from this file?")

solutions = []

if has_date and has_sales:
    solutions.append("Demand forecasting over time")
    solutions.append("Trend and seasonality inspection")

if has_product:
    solutions.append("Product-level demand analysis")

if has_store:
    solutions.append("Store-level demand analysis")

if has_stock:
    solutions.append("Stockout-aware demand correction")
    solutions.append("Lost-sales estimation")

if inferred["discount"] is not None:
    solutions.append("Discount or promotion impact analysis")

if inferred["weather"] is not None:
    solutions.append("Weather-aware demand analysis")

if not solutions:
    st.warning("The dataset needs clearer date and sales fields before meaningful forecasting can be done.")
else:
    for item in solutions:
        st.write(f"✅ {item}")

section_title("Next step")

st.markdown("""
After the schema check, the next step is column mapping.

The user should confirm which columns mean date, sales, product, store, stock, and promotion.
Then the app can engineer features and run a forecast in a controlled way.
""")
