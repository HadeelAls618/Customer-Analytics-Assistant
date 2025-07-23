
# Custalyze: Customer Analytics Assistant

## Project Overview

**Custalyze** is a data-driven system that helps businesses unlock the power of their customer data and offers a powerful foundation for **customer intelligence** through:

* **Customer Lifetime Value (CLV) prediction**
* **RFM-based segmentation**
* **Targeted customer strategy recommendations**

Built on a real-world **UK retail dataset**, this solution uses probabilistic modeling and clustering techniques to generate actionable insights—enabling smarter marketing decisions and stronger customer retention.



## Business Problem

Retailers often focus on short-term sales metrics, neglecting long-term customer value. This leads to:

* Inefficient marketing spend on low-value customers
* Missed opportunities for loyalty-building
* Poor segmentation for personalized outreach

**Custalyze** solves this by providing a data-driven system to:

* Identify high-value customers early
* Predict future purchasing behavior
* Guide marketing teams on where to invest for higher ROI


## Machine Learning Problem

To address the business needs, we reframed them into core ML tasks:

* **CLV Prediction:** A regression problem using probabilistic models (BG/NBD & Gamma-Gamma) to estimate a customer’s lifetime value.
* **Customer Segmentation:** An unsupervised clustering task using K-Means on RFM and CLV features to define strategic customer groups.
* **Purchase Behavior Modeling:** A time-series probabilistic approach to forecast frequency and value of future transactions.

These models transform historical sales data into predictive insights that drive personalized, profitable actions.


## Project Methodology

The project follows a modular pipeline from data processing to app deployment:

### 1. Data Preprocessing & Exploration

* Cleaned and explored the **UK retail dataset**
* Analyzed customer behavior metrics
* Identified missing data, anomalies, and trends

### 2. RFM Analysis

* Calculated **Recency**, **Frequency**, and **Monetary Value** for each customer
* Used RFM scoring to assess engagement levels
* Formed the foundation for segmentation and CLV modeling

### 3. CLV Prediction (Probabilistic Models)

* Applied **BG/NBD model** to estimate future purchase frequency
* Used **Gamma-Gamma model** to predict transaction values
* Combined models to compute **Customer Lifetime Value (CLV)**

### 4. Customer Segmentation

* Applied **K-Means Clustering** on CLV + RFM scores
* Defined 3 key customer segments:

| Segment    | Description                 | Strategy                  |
| ---------- | --------------------------- | ------------------------- |
| Low-Value  | Infrequent spenders         | Re-engagement & offers    |
| High-Value | Loyal, high spenders        | Retention & rewards       |
| At-Risk    | Moderate value, recent drop | Upsell & churn prevention |



## Contact

Want to collaborate or learn more?

* [LinkedIn](https://www.linkedin.com/in/hadeel-als)
* [Email](mailto:alsaadonhadeel@gmail.com)


