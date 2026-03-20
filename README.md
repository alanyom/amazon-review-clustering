# Magazine Subscription Renewal Prediction

Unsupervised ML pipeline predicting magazine subscription renewal likelihood 
from 71K+ Amazon reviews using sentiment analysis, K-Means clustering, and 
SUPG approximate proxy labeling — with logistic regression validation and 
proof of convergence.

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Pipeline](#pipeline)
- [Results](#results)
- [Installation](#installation)

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
