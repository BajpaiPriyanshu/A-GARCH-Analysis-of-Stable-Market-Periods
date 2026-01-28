# Impact of Distributional Assumptions on Volatility Forecasting

## 📌 Overview
This project investigates the impact of error distribution assumptions (Normal vs. Student-t) on volatility forecasting and risk estimation. Focusing on a stable market period for **Reliance Industries (RELIANCE.NS)**, it tests the hypothesis that complex, heavy-tailed distributions offer negligible advantages over Gaussian assumptions during calm market regimes.

## 🚀 Key Features
* **Volatility Modeling:** Implementation of **GARCH(1,1)**, **EGARCH(1,1)**, and **GJR-GARCH(1,1)** models.
* **Distributional Comparison:** Side-by-side analysis of **Normal (Gaussian)** vs. **Student-t** error distributions.
* **Risk Metrics:** Calculation of 1-day ahead **Value at Risk (VaR 99%)** and **Expected Shortfall (ES 97.5%)**.
* **Hypothesis Testing:** Automated evaluation of risk estimate differences against a defined significance threshold (0.2%).

## 📊 Data & Methodology
* **Asset:** Reliance Industries (RELIANCE.NS)
* **Data Source:** Yahoo Finance API (`yfinance`)
* **Period:** Jan 2018 – Jan 2019 (Stable Regime)
* **Training Window:** 250 Trading Days

## 🛠️ Installation & Usage

### Prerequisites
Ensure you have Python 3.8+ and the following libraries installed:

```bash
pip install numpy pandas yfinance arch scipy
