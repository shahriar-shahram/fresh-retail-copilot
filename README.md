# Fresh Retail Copilot

A stockout-aware demand forecasting and retail intelligence system designed to recover true demand, estimate lost sales, and support better inventory decisions.

---

## 1. Introduction

Retail forecasting sounds simple on the surface: look at past sales and predict future sales. In practice, however, retail demand forecasting is much harder than it first appears. One of the biggest reasons is that **observed sales are not always the same as true customer demand**.

If a product is fully available, observed sales may be a good approximation of demand. But if a product is unavailable for part of the day, or out of stock completely, then low sales do **not** necessarily mean low demand. They may simply mean customers wanted the product but could not buy it.

This project is built around that exact problem.

Fresh Retail Copilot is an end-to-end machine learning system for **stockout-aware demand modeling**. It takes fresh-retail sales data, constructs stock-aware features, estimates latent demand under stock constraints, quantifies lost sales, and then trains a forecasting model on corrected demand rather than raw observed sales. The system exposes its predictions and insights through both a **FastAPI backend** and a **Streamlit frontend**, and is designed to be **Dockerized and cloud-deployable**.

This makes the project more than a standard forecasting notebook. It is a full machine learning system built around a real operational challenge in retail.

---

## 2. Why this problem matters

In many real businesses, especially grocery, fresh retail, and fast-moving inventory environments, stock availability changes constantly throughout the day. This creates a form of bias in the historical data.

A simple forecasting model might learn a pattern like this:

- product was unavailable
- sales were low
- therefore demand must have been low

That conclusion is often wrong.

The correct interpretation is closer to:

- product was unavailable
- sales were low because customers could not buy it
- true demand may have been significantly higher than what was observed

This distinction matters because it affects:

- replenishment planning
- inventory control
- lost revenue estimation
- promotion strategy
- product availability decisions
- operational risk assessment

If a business underestimates true demand, it may continue under-stocking key products, which creates a negative loop of recurring stockouts and missed sales. A stockout-aware system attempts to break that loop.

---

## 3. Core idea of the project

The central idea behind this project is:

> **Observed sales are a censored version of true demand.**

A simplified formulation is:

```text
Observed Sales = min(True Demand, Available Inventory)
