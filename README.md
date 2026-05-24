# Airbnb Price Prediction

> **Predict the nightly price of any Airbnb listing in New York City** using property features, location, and host characteristics — with an interactive web app.

🔗 **Live demo:** _coming soon_

---

## Project Overview

This project builds an end-to-end machine learning pipeline to estimate Airbnb nightly prices. It is designed as a portfolio piece demonstrating the full data science workflow: data collection, exploratory analysis, feature engineering, model training, interpretability, and deployment.

**Cities covered**
| Phase | City | Status |
|-------|------|--------|
| 1 | New York City | 🔄 In progress |
| 2 | Los Angeles | ⏳ Planned |

---

## Business Problem

A host setting a new listing doesn't know what price to charge. A traveler wants to know if a listing is fairly priced. This model gives both parties an objective, data-driven price estimate in seconds.

**Key question:** Given the location, type, and features of a listing, what is the expected nightly price?

---

## Data

Source: [Inside Airbnb](https://insideairbnb.com/) — public snapshots, no scraping required.

| File | Description | Rows (NYC) |
|------|-------------|------------|
| `listings.csv` | One row per listing, all features + price | ~40 000 |
| `calendar.csv` | Daily availability & price (future use) | ~14 M |

> **Snapshot date:** _to be documented after download_

Data is **not versioned in this repo** (see `.gitignore`). Download instructions are in [`data/raw/README.md`](data/raw/README.md).

---

## Methodology

The project is split into 7 sequential notebooks:

```
01_EDA          → understand the data, map prices geographically
02_Cleaning     → handle missing values, outliers, corrupt entries
03_Features     → engineer geo / property / host / review features
04_Preprocessing→ encode, scale, log-transform target, train/test split
05_Modeling_Linear → Ridge, Lasso, ElasticNet baselines
06_Modeling_Trees  → Random Forest, XGBoost, LightGBM
07_Final_Evaluation → model selection, SHAP explainability, export
```

Each notebook produces an artifact consumed by the next, ensuring full reproducibility.

---

## Results

> _Section to be completed after model training._

| Metric | Value |
|--------|-------|
| RMSE (log price) | — |
| MAE (dollars) | — |
| Predictions within 10% | — |
| Predictions within 20% | — |

<!-- Add price map and SHAP plot here -->

---

## Key Insights

> _Section to be completed after EDA and SHAP analysis._

---

## How to Run

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/airbnb-price-prediction.git
cd airbnb-price-prediction
```

**2. Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Download the data**

Follow the instructions in [`data/raw/README.md`](data/raw/README.md).

**4. Run the notebooks in order**
```bash
jupyter notebook
```
Open and run `01_EDA.ipynb` through `07_Final_Evaluation.ipynb` sequentially.

**5. Launch the app locally**
```bash
streamlit run app/app.py
```

---

## Limitations and Future Work

**Current limitations (v1)**
- NYC only; no seasonality modeling
- Price is the listed price, not the booked price
- No NLP on listing descriptions or reviews

**Planned extensions**
- Los Angeles (Phase 2)
- Dynamic pricing from `calendar.csv`
- NLP features from listing descriptions
- Sentiment analysis on guest reviews
- Occupancy rate prediction
- REST API with FastAPI + Docker deployment

---

## License

[MIT](LICENSE)
