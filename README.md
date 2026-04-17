# 🛒 Fresh Retail Copilot

> Stockout-aware demand forecasting and retail intelligence system  
> Built to recover true demand, quantify lost sales, and enable better inventory decisions.

---

## 🚀 Overview

Fresh Retail Copilot is an end-to-end machine learning system that addresses a critical retail problem:

> **Observed sales ≠ true demand**

Products often sell less not because demand is low, but because they are unavailable.

This system reconstructs **latent (true) demand**, estimates **lost sales**, and produces **corrected forecasts** for better decision-making.

---

## 🧠 Key Features

- 📦 **Stockout-aware feature engineering**
- 🔍 **Latent demand recovery model**
- 📉 **Lost sales estimation**
- 📈 **Forecasting on corrected demand**
- ⚡ **FastAPI real-time inference API**
- 🖥 **Interactive Streamlit dashboard**
- 🐳 **Dockerized for deployment**
- ☁️ **Cloud-ready (GCP Cloud Run)**

---

## 🧩 Problem Statement

Retail data is inherently biased:

- If a product is out of stock → sales drop to zero
- But demand may still exist

Traditional forecasting models:

```text
Learn from incorrect signals → underestimate demand
