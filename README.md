# 🛒 Fresh Retail Copilot

> A stockout-aware demand forecasting and retail intelligence system  
> that reconstructs true demand, quantifies lost sales, and enables data-driven inventory decisions.

---

## 📌 Executive Summary

In real-world retail systems, observed sales are often **censored by inventory constraints**. When a product is out of stock, sales drop — but demand may still exist.

This project builds a **production-ready machine learning pipeline** that:

- Recovers **latent (true) demand** under stockout conditions  
- Estimates **lost sales at store-product-day level**  
- Trains forecasting models on **de-biased demand signals**  
- Serves predictions via a **FastAPI backend**  
- Exposes insights through a **Streamlit dashboard**  
- Is fully **containerized and deployable to cloud environments**

---

## 🧠 Problem Formulation

### Observational Bias in Retail Data

Retail datasets typically contain:

```text
Observed Sales = min(True Demand, Available Inventory)
