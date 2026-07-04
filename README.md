# Customer Feedback Analytics System (CFAS)

An end-to-end analytics system that cleans, analyzes, and visualizes customer
feedback data to measure satisfaction, uncover service gaps, and support
data-driven retention strategy — built as an MCA Major Project
(Course Code: 23ONMCR-753, Chandigarh University, CDOE).

## 📌 Project Overview

This system processes customer feedback data to:
- Calculate a **Customer Satisfaction Index (CSI)** using a Bayesian-weighted
  rating methodology
- Perform **sentiment analysis** and **keyword extraction** on open-ended
  reviews using VADER and TF-IDF
- Identify **key drivers** of satisfaction via correlation and regression
  analysis
- **Segment customers/products** into actionable groups using K-Means
  clustering
- Present all findings in an **interactive dashboard**

## 📊 Data Source

**Amazon India Sales & Customer Reviews Dataset** — Kaggle
([karkavelrajaj/amazon-sales-dataset](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset))

Used as a secondary dataset (1,465 products, cleaned to 1,350) given the
project's compressed timeline. Product `rating` is used as the satisfaction
proxy in place of a live primary survey.

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data Cleaning & Analysis | Python, Pandas, NumPy |
| Sentiment Analysis | VADER (vaderSentiment) |
| Keyword Extraction | scikit-learn (TF-IDF) |
| Statistical Analysis | scikit-learn (Linear Regression), SciPy |
| Segmentation | scikit-learn (K-Means Clustering) |
| Visualization | Matplotlib, Plotly |
| Dashboard | Self-contained interactive HTML (Plotly.js) |
| Development Environment | Jupyter Notebook, VS Code |

## 📁 Project Structure

```
CFAS_Project/
├── 01_cleaning_and_csi.ipynb          # Data cleaning + CSI calculation
├── 02_sentiment_and_keywords.ipynb    # VADER sentiment + TF-IDF keywords
├── 03_drivers_and_segmentation.ipynb  # Driver analysis + K-Means segmentation
├── generate_dashboard.py              # Builds the interactive HTML dashboard
├── dashboard.html                     # Generated dashboard (open in browser)
├── data/
│   ├── raw/
│   │   └── amazon_raw.csv             # Original dataset
│   ├── cleaned_feedback_data.csv
│   ├── csi_by_product.csv
│   ├── csi_by_category.csv
│   ├── csi_by_price_tier.csv
│   ├── feedback_with_sentiment.csv
│   ├── feedback_final_with_clusters.csv
│   ├── top_positive_keywords.csv
│   ├── top_negative_keywords.csv
│   ├── sentiment_by_category.csv
│   └── cluster_profile.csv
└── README.md
```

## 🚀 How to Run

1. Clone the repository:
   ```
   git clone https://github.com/singhkhushi2610/Customer-Satisfaction-and-Feedback-Analytics-Dashboard-System.git
   cd Customer-Satisfaction-and-Feedback-Analytics-Dashboard-System
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install pandas numpy scikit-learn vaderSentiment matplotlib plotly jupyter
   ```

3. Run the notebooks in order (`01` → `02` → `03`) to reproduce the full
   analysis pipeline.

4. Generate the dashboard:
   ```
   python generate_dashboard.py
   ```

5. Open `dashboard.html` directly in your browser — no server required.

## 📈 Key Findings

- **Overall CSI:** 81.8 / 100 (simple average), 83.2 / 100 (review-volume weighted)
- **Sentiment-rating correlation:** r = 0.218 (weak-positive — written sentiment
  and star ratings capture overlapping but distinct signal)
- **Discount % negatively correlates with rating** (r = -0.162) — steeper
  discounts are associated with slightly lower satisfaction
- **Four customer segments identified:**
  - Core Performers (540 products, budget, high satisfaction)
  - Premium Flagship (66 products, highest sentiment)
  - Mainstream Budget (690 products, largest segment)
  - **At-Risk / Discount-Driven** (54 products, negative sentiment, heaviest
    discounting — priority for quality review)

## 🎓 Academic Context

This project was developed individually as part of the MCA Fourth Semester
Major Project (23ONMCR-753) at Chandigarh University's Centre for Distance &
Online Education, following the official project guidelines for report
structure, evaluation, and submission.

## 👤 Author

Khushi Singh — MCA, Chandigarh University