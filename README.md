# Fresh Retail Copilot

Stockout-aware demand forecasting and retail intelligence system built from real fresh-retail transactional data.

## Overview

Fresh Retail Copilot is an end-to-end machine learning system designed to handle one of the hardest retail forecasting problems: observed sales are often censored by stockouts, so raw sales do not always reflect true customer demand.

This project addresses that by:
- engineering stock-aware features from store-product-day retail data
- estimating latent demand under stockout conditions
- quantifying lost sales
- forecasting future demand using corrected demand signals
- serving predictions through a FastAPI backend
- exposing insights through a Streamlit frontend

## Problem Statement

In retail, low observed sales do not always mean low demand. A product may sell poorly because it was unavailable for part or all of the day.

This system distinguishes between:
- **observed sales**
- **recovered latent demand**
- **lost sales due to stockouts**

That makes the resulting forecasts more useful for inventory and demand planning.

## Dataset

The project uses a fresh-retail dataset with fields such as:
- `city_id`, `store_id`, `product_id`
- `dt`
- `sale_amount`
- `hours_sale`
- `stock_hour6_22_cnt`
- `hours_stock_status`
- `discount`
- `holiday_flag`
- `activity_flag`
- weather variables such as precipitation, temperature, humidity, and wind level

## Pipeline

### 1. Data inspection and subsetting
A manageable MVP subset is created from the raw parquet files for rapid experimentation.

### 2. Feature engineering
The feature pipeline builds:
- stock availability ratio
- stockout indicators
- calendar features
- hourly-demand summary features
- lag features
- rolling demand features

### 3. Latent demand recovery
A demand recovery model is trained on higher-availability rows and used to estimate true demand under stockout conditions.

### 4. Lost sales estimation
Recovered demand is compared against observed sales to estimate demand lost due to stockouts.

### 5. Forecasting
A forecasting model is trained on corrected demand rather than raw censored sales.

### 6. Serving and UI
- **FastAPI** serves real-time demand predictions
- **Streamlit** provides an interactive business-facing interface

## Architecture

```text
Raw Retail Data
    ↓
Inspection / Subsetting
    ↓
Feature Engineering
    ↓
Latent Demand Recovery
    ↓
Lost Sales Estimation
    ↓
Forecast Model
    ↓
FastAPI Backend
    ↓
Streamlit Frontend