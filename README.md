# **Custalyze: Your Customer Analytics Assistant**

Welcome to **Custalyze**, your go-to analytics assistant for predicting **Customer Lifetime Value (CLV)**, providing personalized product recommendations, and offering deep insights into customer behavior. Simply enter a customer ID, and Custalyze will help you uncover key insights that enhance customer engagement and optimize marketing strategies.

Explore the live app: [Custalyze Web App](https://customer-analytics-assistant-urzknsdg8udnsjm9x4efhd.streamlit.app/)

---

## **Project Overview**

**Custalyze** helps retail businesses leverage customer data to predict **Customer Lifetime Value (CLV)**, understand customer segments, and provide personalized product recommendations. Built using a UK retail dataset, this project enables businesses to optimize their marketing efforts by providing valuable insights into each customer’s potential value and behavior, all through a user-friendly web interface.

---

### **App Features:**
- Predict **Customer Lifetime Value (CLV)** for individual customers.
- Perform **RFM Analysis** to evaluate customer engagement and value.
- **Customer Segmentation** based on predicted CLV and RFM scores for tailored marketing strategies.
- Provide **Personalized Product Recommendations** using advanced recommendation systems based on customer segments.
- Simple web interface that allows users to input a customer ID and instantly retrieve detailed insights.

---

## **Problem Statement & Solution**

In today’s highly competitive retail market, understanding customer behavior and predicting future customer value are essential for optimizing marketing strategies and maximizing business growth. Many businesses invest heavily in customer acquisition without fully recognizing the long-term potential of individual customers, leading to inefficient marketing spend and missed opportunities for customer retention.

**Custalyze** addresses these challenges by providing a comprehensive solution that quantifies a customer’s potential and delivers actionable insights. Built using a UK retail dataset, **Custalyze** empowers businesses to make informed decisions, optimize marketing resources, and retain high-value customers. By offering predictive insights into CLV and personalized strategies for each customer segment, businesses can tailor their marketing efforts, improve customer engagement, and drive sustainable growth.

---

## **Project Workflow**

### **Step 1: Data Preprocessing and Exploration**
- **Data cleaning** and **Exploratory Data Analysis (EDA)** performed on the UK retail dataset.
- Identify key trends in customer behavior: purchase frequency, spending patterns, and recency.
- Prepare the dataset for RFM analysis and modeling.

### **Step 2: RFM Analysis**
- Conduct **RFM Analysis** to evaluate customers based on:
  - **Recency**: How recently a customer made a purchase.
  - **Frequency**: How often a customer makes purchases.
  - **Monetary Value**: How much a customer spends.
- Group customers based on RFM scores to categorize engagement and value levels.
- Output provides insight into customer activity, forming the foundation for segmentation and CLV predictions.

### **Step 3: Predict CLV Using Statistical Models**
- **BG/NBD Model**: Predict future purchase probabilities based on past activity.
- **Gamma-Gamma Model**: Estimate future transaction value based on previous spending patterns.
- Combine both models to calculate **Customer Lifetime Value (CLV)** for each customer.
- **CLV outputs** guide targeted marketing and personalized strategies.

### **Step 4: Customer Segmentation and Strategy**
- Use **K-Means Clustering** to segment customers based on **CLV** and **RFM scores**.
- Develop tailored strategies for each segment:
  - **Low-Engagement Customers (Segment 0)**:
    - **Characteristics**: Low recency, frequency, and monetary value.
    - **Strategy**: Focus on onboarding and re-engagement.
    - **Action**: Use personalized offers and promote best-sellers to increase activity.
  - **High-Value Loyal Customers (Segment 1)**:
    - **Characteristics**: High recency, frequency, and monetary value.
    - **Strategy**: Retain and enhance loyalty.
    - **Action**: Provide exclusive offers, early access, and personalized recommendations.
  - **Moderate CLV Customers (Segment 2)**:
    - **Characteristics**: Moderate spending, high recency (potential churn risk).
    - **Strategy**: Focus on retention and increasing engagement.
    - **Action**: Use upselling, cross-selling, and bundle offers to drive higher value and prevent churn.

### **Step 5: Build Personalized Product Recommendation System**
- Implement recommendation systems tailored to each customer segment:
  - **Segment 0 (Low-Engagement Customers)**:
    - **Approach**: **Content-Based Filtering** for personalized offers and best-sellers.
    - **Goal**: Onboard and re-engage customers to increase activity.
  - **Segment 1 (High-Value Loyal Customers)**:
    - **Approach**: **Hybrid System** combining **Content-Based Filtering** and **Collaborative Filtering** for highly personalized and exclusive offers.
    - **Goal**: Retain top-tier customers and reward loyalty with personalized, premium offers.
  - **Segment 2 (Moderate CLV Customers)**:
    - **Approach**: **Hybrid System** combining **Association Rule-Based Systems** (upselling/cross-selling) and **Collaborative Filtering** for relevant product bundles and complementary offers.
    - **Goal**: Increase order value and prevent churn by offering strategic product suggestions.

### **Step 6: Model Deployment Using Streamlit**
- **Streamlit App**: Deploy the CLV prediction model and customer insights in an interactive web application.
- Features of the app:
  - Input customer IDs to view analytics such as CLTV predictions, segmentation, and product recommendations.
  - Navigate between different tasks (CLTV analysis, segmentation, personalized recommendations).
  - Provide real-time insights for business decision-makers.
  
  Explore the live app: [Custalyze Web App](https://customer-analytics-assistant-urzknsdg8udnsjm9x4efhd.streamlit.app/)

---

## **Future Enhancements**

- **Real-Time CLV Prediction**: Implement real-time data streaming to update CLV predictions dynamically as customer behavior changes.
- **A/B Testing**: Integrate A/B testing to measure the effectiveness of personalized product recommendations and marketing strategies.
- **Advanced Dashboard**: Add more visualizations and dynamic dashboards to track customer engagement, CLV trends, and marketing performance.

---

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
