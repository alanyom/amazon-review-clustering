# Magazine Subscription Renewal Prediction

Unsupervised ML pipeline predicting magazine subscription renewal likelihood 
from 71K+ Amazon reviews using sentiment analysis, K-Means clustering, and 
SUPG approximate proxy labeling — with logistic regression validation and 
proof of convergence.

---

## Overview

This project analyzes raw Amazon reviews of magazine subscriptions to predict 
customer renewal likelihood using unsupervised learning techniques. Since no 
ground truth labels exist, K-Means clustering is used to group customers based 
on review features, sentiment scores, and rating data.

Key highlights:
- **71,497** Amazon magazine subscription reviews processed
- NLP sentiment analysis via pretrained DistilBERT model
- SQL database integration for structured data management
- SUPG approximate proxy labeling to expand a small hand-labeled dataset
- Logistic regression validation achieving **94% accuracy**
- Mathematical proof of label propagation convergence

---

## Dataset

- **Source:** Amazon Magazine Subscriptions Reviews (`Magazine_Subscriptions.jsonl`)
- **Size:** 71,497 reviews
- **Features used:**
  - `rating` — 1 to 5 star rating
  - `title` — Review title
  - `text` — Review body
  - `helpful_vote` — Number of helpful votes
  - `verified_purchase` — Whether purchase was verified
  - `timestamp` — Date of review

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| ML / Clustering | scikit-learn (K-Means, Logistic Regression) |
| NLP | HuggingFace Transformers (DistilBERT) |
| Database | MySQL + SQLAlchemy |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |

---

## Pipeline
```
Raw JSONL Data
      │
      ▼
Data Preprocessing & Cleaning
      │
      ▼
SQL Database Storage
      │
      ▼
Sentiment Analysis (DistilBERT)
   - Review text → sentiment score [-1, 1]
   - Review title → sentiment score [-1, 1]
      │
      ▼
K-Means Clustering (k=2)
   - Features: rating, verified_purchase,
     text_sentiment, title_sentiment
   - Cluster 0: High likelihood of renewal
   - Cluster 1: Low likelihood of renewal
      │
      ▼
Bootstrap Confidence Interval
   - 95% CI: (0.258, 0.264)
      │
      ▼
SUPG Proxy Labeling
   - 50 hand-labeled data points
   - Euclidean distance-based label propagation
   - Expands labeled dataset iteratively
      │
      ▼
Logistic Regression Validation
   - Trained on proxy-labeled data
   - 94% accuracy on labeled test set
      │
      ▼
Proof of Convergence
   - Switch count tracked across iterations
   - Labels stabilize as iterations increase
```

---

## Results

| Metric | Value |
|---|---|
| Total Reviews Processed | 71,497 |
| K-Means Clusters | 2 |
| High Renewal Likelihood | ~74% of reviews |
| Low Renewal Likelihood | ~26% of reviews |
| Bootstrap 95% CI | (0.258, 0.264) |
| Logistic Regression Accuracy | 94% |

**Key Findings:**
- The dataset has a significant positive skew in ratings, reflecting that 
  customers are more likely to leave reviews for positive experiences
- Sentiment scores from both review titles and bodies aligned well with 
  star ratings, validating the clustering approach
- Label propagation converged consistently, with switch counts trending 
  toward 0 as iterations increased
