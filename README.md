# 🛒 Fresh Retail Copilot

> 🚀 A stockout-aware demand forecasting and retail intelligence system  
> built to recover true demand, estimate lost sales, and support smarter inventory decisions.

---

## 🌟 Overview

Fresh Retail Copilot is an end-to-end machine learning system built around a real and difficult retail problem:

> **Observed sales are not always the same as true demand.**

In many retail settings, especially fresh retail, products may be unavailable for part of the day or fully out of stock. When that happens, low sales do **not** necessarily mean low customer demand. They may simply mean customers wanted the product but could not buy it.

This project is designed to handle exactly that situation.

Instead of treating raw sales as the ground truth, Fresh Retail Copilot:

- 🧠 learns from stock-aware signals
- 📉 estimates **latent (true) demand**
- 💸 quantifies **lost sales**
- 📈 forecasts future demand using a **corrected demand target**
- ⚡ serves predictions through a **FastAPI backend**
- 🖥️ exposes results through an **interactive Streamlit application**
- 🐳 packages everything in Docker for deployment

This makes the project much more than a standard forecasting notebook. It is a full machine learning system designed around a real operational constraint.

---

## ❓ Why this project matters

Most retail forecasting projects assume that historical sales are a clean reflection of customer demand.

That assumption is often wrong.

If a product is stocked all day, observed sales may be a good signal of demand.

But if a product is out of stock for part of the day, then observed sales are **censored** by inventory availability.

### ✅ Example

Imagine a product with true customer demand of **20 units**, but only **5 units** were available.

What do we observe?

- Sales = 5
- True demand = 20
- Lost demand = 15

A naive model would learn:
> “Demand was 5.”

A stockout-aware model should learn:
> “Observed sales were 5, but true demand was likely much higher.”

That distinction matters because it affects:

- 📦 replenishment planning
- 🏬 inventory allocation
- 💰 revenue estimation
- 📉 stockout monitoring
- 🎯 promotion strategy
- 📊 operational decision support

---

## 🎯 Project goals

Fresh Retail Copilot is built to answer four key questions:

### 1. 📍 What is happening right now?
For a selected store-product scenario:
- what are recent sales?
- what is the current stock availability?
- what does the discount look like?
- what do weather and calendar conditions look like?

### 2. 🧠 What was the likely true demand?
If stock availability was limited, can we estimate what customers would have bought under better inventory conditions?

### 3. 💸 How much demand was lost?
Can we quantify lost sales caused by stockouts?

### 4. 📈 What is likely to happen next?
Can we forecast future demand using corrected demand rather than raw biased sales?

---

## 🧩 Problem formulation

The core idea behind this project is:

```text
Observed Sales = min(True Demand, Available Inventory)
```

This means:

- if inventory is sufficient → observed sales may reflect demand
- if inventory is constrained → observed sales underestimate demand

So the project does **not** directly trust raw sales as the final target.

Instead, it separates the problem into two main stages:

### Stage 1 — 🔍 Latent Demand Recovery
Estimate what demand likely would have been if inventory had not constrained sales.

### Stage 2 — 📈 Forecasting on Corrected Demand
Train the forecast model using a corrected demand target instead of raw observed sales.

This is what makes the project more realistic and more valuable than a typical forecasting demo.

---

## 🗂️ Dataset explanation

This project uses a fresh-retail dataset containing store, product, sales, availability, business, and weather signals.

Each row corresponds to a **store-product-time observation**.

The dataset includes several groups of features.

---

## 🏷️ Identifier and hierarchy features

These columns define the retail structure:

- `city_id`
- `store_id`
- `management_group_id`
- `first_category_id`
- `second_category_id`
- `third_category_id`
- `product_id`

These features allow the data to be analyzed at multiple levels:

- city
- store
- category
- product

This is important because retail demand is hierarchical. A demand pattern may depend on:
- geography
- store behavior
- product family
- individual SKU characteristics

---

## 🕒 Time feature

- `dt`

This is the time column used to sort and model the sequence of demand over time.

Time structure is essential for:
- lag features
- rolling averages
- trend analysis
- forecasting

---

## 💰 Demand / sales signal

- `sale_amount`

This is the observed sales amount.

At first glance, this looks like the obvious forecasting target.

But in this project, `sale_amount` is treated carefully, because it may be biased by stock constraints.

---

## 📦 Stock availability features

These are among the most important columns in the entire dataset.

### `stock_hour6_22_cnt`

This represents the number of hours the product was in stock between **6 AM and 10 PM**.

That is a 17-hour operating window.

So:

```text
in_stock_ratio = stock_hour6_22_cnt / 17
```

Examples:
- `17` → fully available all day
- `8` → available for about half the day
- `0` → unavailable all day

This is one of the key signals used to determine how trustworthy the observed sales are.

---

### `hours_stock_status`

This contains a 24-hour stock availability pattern, typically as an array of 0s and 1s.

Interpretation:
- `1` → in stock during that hour
- `0` → out of stock during that hour

This gives a much richer view of stock behavior than a single scalar availability count.

---

## 🧾 Hourly sales feature

### `hours_sale`

This contains the hourly sales pattern for the same time period.

It allows us to derive useful behavioral features such as:
- average sales when the product is available
- hourly demand variability
- peak demand hour

This is a strong feature source because it captures **within-day demand behavior**, not just total sales.

---

## 🛍️ Business and promotion features

These help explain why demand changes beyond pure history.

- `discount`
- `holiday_flag`
- `activity_flag`

These capture:
- pricing effects
- holiday-related changes
- business activities / promotions

These are useful exogenous variables for both recovery and forecasting.

---

## 🌦️ Weather features

Fresh retail demand is often affected by weather. The dataset includes:

- `precpt`
- `avg_temperature`
- `avg_humidity`
- `avg_wind_level`

These help the system account for environmental variation that may influence customer behavior.

---

## 😵 Why this dataset is difficult

This is not a toy forecasting dataset.

It is difficult for several reasons:

### 1. 🚫 Censored observations
Observed sales may be lower than true demand because the product was unavailable.

### 2. 🧱 Mixed data structure
The data includes:
- scalar features
- time-series lags
- hourly arrays
- hierarchical identifiers

### 3. 🏪 Hierarchical demand
Demand varies by:
- city
- store
- category
- product

### 4. 🌧️ External drivers
Demand depends not only on history, but also on:
- stock availability
- discounts
- promotions
- weather
- holidays

### 5. ⚙️ Operational realism
This is the kind of dataset where correct formulation matters more than blindly applying a model.

---

## 🏗️ System design

Fresh Retail Copilot is intentionally built as a full ML system, not just a notebook.

### High-level pipeline

```text
Raw Data
   ↓
Inspection / Subsetting
   ↓
Feature Engineering
   ↓
Latent Demand Recovery
   ↓
Lost Sales Estimation
   ↓
Corrected Demand Construction
   ↓
Forecast Model
   ↓
FastAPI Inference Service
   ↓
Streamlit Dashboard
```

---

## 🧠 Feature engineering

Feature engineering is one of the strongest parts of the project.

The goal is to transform raw transactional and stock information into a model-ready feature space that respects how the data was generated.

### ✅ Stock-aware features

Examples include:

- `in_stock_ratio`
- `is_full_stock`
- `is_stockout`
- `stockout_hours`

These features tell the model whether low sales are reliable or likely constrained.

---

### ✅ Calendar features

Examples include:

- `day_of_week`
- `month`

These help capture recurring demand patterns.

---

### ✅ Time-series features

Examples include:

- `lag_1`
- `lag_7`
- `rolling_mean_7`

These allow the system to capture short-term memory and local trend.

---

### ✅ Hourly behavior summaries

Instead of expanding 24-hour arrays into 24 raw columns, the project extracts meaningful summaries such as:

- `avg_sales_when_available`
- `peak_hour`
- `demand_std`

This keeps the model expressive without becoming noisy or fragile.

---

### ✅ Exogenous contextual features

The final feature space also includes:
- discount
- holiday
- activity
- precipitation
- temperature
- humidity
- wind level

These help the model move beyond purely autoregressive logic.

---

## 🔍 Latent demand recovery

This is the first major modeling stage.

### Why recovery is needed

If a product is not available, raw sales may underestimate true customer demand.

So instead of learning from all rows equally, the project distinguishes between:

- ✅ relatively reliable rows with high stock availability
- ⚠️ less reliable rows with lower stock availability

### Recovery approach

A model is trained on the cleaner subset and then applied to constrained rows.

Conceptually:

```text
Recovered Demand = model(features)
Lost Sales = max(Recovered Demand - Observed Sales, 0)
```

### Why this is valuable

This makes it possible to estimate:
- how much demand was suppressed by stockouts
- which store-product combinations are operationally risky
- where inventory issues are likely creating revenue leakage

---

## 💸 Lost sales estimation

Once latent demand is estimated, the system computes:

```text
Lost Sales = max(Recovered Demand - Observed Sales, 0)
```

This turns the project from a pure prediction system into a **decision-support system**.

### Example interpretation

If:
- observed sales = `17.6`
- recovered demand = `32.8`

Then:
- estimated lost sales = `15.2`

That suggests a major stockout-related revenue gap.

This kind of output is directly useful for:
- replenishment prioritization
- product-level diagnostics
- operational reporting

---

## 📈 Forecasting on corrected demand

The second modeling stage uses a **corrected demand target**.

The idea is:

- if stock availability is high → use observed sales
- if stock availability is low → use recovered demand

This produces a cleaner target for forecasting.

### Why this matters

A forecast trained on raw sales may inherit inventory bias.

A forecast trained on corrected demand is better aligned with what customers actually wanted.

That is the forecasting target that matters for inventory planning.

---

## 🤖 Models used

This project currently uses lightweight, interpretable, deployable models.

That is intentional.

The goal is to build something that is:
- understandable
- stable
- fast to serve
- easy to deploy
- strong enough to be useful

### Recovery model
Used to estimate latent demand.

### Forecast model
Used to predict next-demand based on corrected targets and engineered features.

This is a strong baseline architecture and a solid production-style starting point.

---

## 📏 Evaluation

The project evaluates model performance with standard regression metrics such as:

- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error

These metrics are applied in the context of:
- latent demand recovery
- corrected-demand forecasting

The focus is not only on prediction quality, but on whether the learned target is **meaningful and operationally valid**.

---

## 🖥️ Frontend (Streamlit)

The Streamlit application is designed to make the system understandable and interactive.

For a selected store-product scenario, it shows:

- 🎛️ store and product selection
- 📍 current context
- 📉 recent demand trend
- 💸 lost sales insight
- 🔮 next-demand prediction
- 🧠 short business explanation
- 📈 prediction vs recent trend

This makes the system feel like a real product instead of a notebook demo.

---

## ⚡ Backend (FastAPI)

The FastAPI service provides real-time prediction through an API endpoint.

### Example endpoint

```http
POST /predict
```

### Input
A structured feature payload derived from the latest store-product context.

### Output
A numerical prediction for next demand.

This makes the ML system portable and deployable in a production-style architecture.

---

## 🐳 Docker and deployment

The API is containerized using Docker.

That means the system is packaged with:
- code
- dependencies
- model artifacts
- serving logic

This eliminates “works on my machine” problems and makes the backend ready for deployment to platforms like:

- Google Cloud Run
- other container-based services

---

## 📁 Repository structure

```text
fresh-retail-copilot/
├── api/
│   └── main.py
├── app/
│   └── main.py
├── notebooks/
│   ├── inspect_data.py
│   ├── create_subset.py
│   ├── inspect_subset.py
│   ├── feature_engineering.py
│   ├── train_recovery_model.py
│   ├── lost_sales_analysis.py
│   ├── build_training_target.py
│   └── train_forecast_model.py
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── models/
│   ├── forecast_model.pkl
│   └── forecast_features.pkl
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ▶️ Running the project locally

### 1. Create environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the backend

```bash
uvicorn api.main:app --reload
```

API docs:

```text
http://127.0.0.1:8000/docs
```

### 4. Run the frontend

```bash
streamlit run app/main.py
```

### 5. Docker version of the API

```bash
docker build -t fresh-retail-api .
docker run -p 8080:8080 fresh-retail-api
```

Then point the app to the Docker API URL.

---

## ☁️ Deployment plan

The current deployment strategy is:

- deploy the **API** first
- keep the **frontend** local or deploy it separately afterward

This is a practical and scalable architecture:
- backend = stateless service
- frontend = presentation layer

---

## 💼 Business value

Fresh Retail Copilot can support:

- 📦 inventory optimization
- 🚨 stockout impact analysis
- 📈 demand planning
- 💸 lost revenue estimation
- 🎯 promotion analysis
- 🧠 retail decision support

This is especially relevant in fresh retail, where product availability and demand patterns can change rapidly.

---

## ⚠️ Current limitations

Like any real ML system, this project has constraints.

Current limitations include:

- single-step demand forecasting rather than long-horizon forecasting
- simplified recovery logic based on availability thresholds
- no probabilistic uncertainty intervals yet
- frontend is a decision-support prototype, not a full enterprise product

These are not weaknesses in the core system. They are natural opportunities for future work.

---

## 🔮 Future work

Potential next upgrades include:

- multi-step forecasting
- hierarchical forecasting across city/store/category/product
- stronger recovery models
- gradient boosting or sequence models
- uncertainty estimation
- public deployment of the frontend
- LLM-powered retail assistant
- reinforcement learning for inventory decisions

---

## 🏆 Why this project stands out

Most forecasting projects answer:

> “What will sales be?”

Fresh Retail Copilot also asks:

> “What would demand have been if stock had been available?”  
> “How much demand did the business fail to capture?”  
> “How can we forecast demand more accurately under operational constraints?”

That shift — from raw prediction to **constraint-aware demand intelligence** — is what makes this project meaningful.

---

## 👨‍💻 Author

**Shahriar Shahram**  
PhD Candidate in Electrical & Computer Engineering  
University of Central Florida

---

## ✅ Final takeaway

Fresh Retail Copilot is built on one core principle:

> Sales data is only meaningful if you understand the inventory constraints that generated it.

By combining stock-aware feature engineering, latent demand recovery, lost-sales estimation, corrected-demand forecasting, API serving, and interactive visualization, this project turns raw retail observations into a practical retail intelligence system.

## Live API

- Base URL: `https://fresh-retail-api-837696130499.us-central1.run.app`
- API Docs: `https://fresh-retail-api-837696130499.us-central1.run.app/docs`
