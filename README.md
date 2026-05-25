# Airbnb Price Prediction — Austin, TX

## Project Overview
End-to-end machine learning pipeline to predict Airbnb nightly prices in Austin, Texas.
Built as a portfolio project targeting the US freelance data science market.

## Business Problem
Given a listing's characteristics (location, property type, amenities, host profile),
predict the nightly price a host should charge to remain competitive in the Austin market.

## Data
| File | Source | Snapshot | Records |
|------|--------|----------|---------|
| `listings.csv.gz` | [Inside Airbnb](https://insideairbnb.com/get-the-data/) | September 16, 2025 | ~14,000 listings |
| `calendar.csv.gz` | [Inside Airbnb](https://insideairbnb.com/get-the-data/) | September 16, 2025 | ~5M rows |

> Data is not included in this repository. Download instructions in `data/raw/Austin/SOURCES.md`.

## Pipeline
| Notebook | Role |
|----------|------|
| `01_EDA.ipynb` | Exploratory data analysis |
| `02_Cleaning.ipynb` | Data cleaning |
| `03_Features.ipynb` | Feature engineering |
| `04_Preprocessing.ipynb` | Encoding, scaling, train/test split |
| `05_Modeling_Linear.ipynb` | Ridge, Lasso, ElasticNet |
| `06_Modeling_Trees.ipynb` | Random Forest, XGBoost, LightGBM |
| `07_Final_Evaluation.ipynb` | Model selection, SHAP, final metrics |

## Results
*To be completed after modeling phase.*

## How to Run
```bash
git clone https://github.com/capristunt/airbnb-price-prediction
cd airbnb-price-prediction
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Download data — see data/raw/Austin/SOURCES.md
jupyter notebook
```

## Limitations and Future Work
- Phase 2 : apply pipeline to a second US city (Dallas or Nashville)
- NLP on listing descriptions
- Dynamic pricing from `calendar.csv`
- Deployment via FastAPI + Docker

## Stack
Python · pandas · scikit-learn · XGBoost · LightGBM · SHAP · Streamlit · folium