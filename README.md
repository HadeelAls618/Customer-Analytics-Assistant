# **Custalyze: Your Customer Analytics Assistant**

Welcome to **Custalyze**, your go-to analytics assistant for predicting **Customer Lifetime Value (CLV)**, providing personalized product recommendations, and offering deep insights into customer behavior. Simply enter a customer ID, and Custalyze will help you uncover key insights that will enhance your customer engagement and optimize your marketing strategies.

Explore the live app: [Custalyze Web App](https://customer-analytics-assistant-urzknsdg8udnsjm9x4efhd.streamlit.app/)

---

## **Project Overview**

**Custalyze** helps retail businesses leverage customer data to predict **Customer Lifetime Value (CLV)**, understand customer segments, and provide personalized product recommendations. Built using a UK retail dataset, this project enables businesses to optimize their marketing efforts by providing valuable insights into each customer’s potential value and behavior, all through a user-friendly web interface.

### **Key Features:**
- Predict **Customer Lifetime Value (CLV)** for individual customers.
- Perform **RFM Analysis** to evaluate customer engagement and value.
- **Customer Segmentation** based on predicted CLV and RFM scores for tailored marketing strategies.
- Offer **Personalized Product Recommendations** using advanced recommendation systems based on customer segments.
- Simple web interface that allows users to input a customer ID and retrieve detailed insights instantly.

---

## **Introduction**

In the highly competitive retail market, understanding customer behavior and predicting future customer value are crucial to optimizing marketing strategies. Many businesses spend excessively on customer acquisition without realizing the long-term potential of each customer. **Custalyze** offers businesses a solution by quantifying a customer's potential and providing actionable insights to maximize customer retention and drive business growth.

Built using a **UK retail dataset**, Custalyze enables businesses to predict CLV, segment customers, and deliver personalized product recommendations tailored to each customer’s value and engagement level.

---

## **Problem Statement**

Retail businesses often struggle to understand the long-term value of individual customers. As a result, they may waste resources on inefficient marketing strategies or miss out on retaining high-value customers. **Custalyze** addresses these challenges by offering:

- Insights into **Customer Lifetime Value (CLV)** to forecast future worth.
- **Customer Segmentation** to personalize marketing efforts.
- **Personalized Product Recommendations** tailored to each customer segment to boost engagement and sales.
  
---

## **Solution Overview**

**Custalyze** provides a streamlined solution for businesses looking to:

- **Predict Customer Lifetime Value (CLV)**: Accurately forecast the future potential of customers.
- **Segment Customers**: Group customers into meaningful segments based on their behavior and potential value.
- **Personalize Engagement**: Offer product recommendations based on customer preferences, history, and segmentation.
- **Optimize Marketing Spend**: Focus marketing efforts on high-value customer segments to maximize returns.
---

## **Project Workflow**

### **Step 1: Data Preprocessing and Exploration**
- Perform data cleaning and **Exploratory Data Analysis (EDA)** to identify customer behaviors and trends in the UK retail dataset.

### **Step 2: RFM Analysis**
- Conduct **RFM Analysis** to evaluate each customer's **Recency, Frequency,** and **Monetary** value.
- Use RFM scores to group customers into different segments based on engagement and value.

### **Step 3: Predict CLV Using Statistical Models**
- **BG/NBD Model**: Predict future purchase probabilities based on past customer activity.
- **Gamma-Gamma Model**: Estimate the monetary value of future transactions.
- Combine these models to accurately predict the **Customer Lifetime Value (CLV)** for each customer.

### **Step 4: Customer Segmentation**
- Apply **K-Means Clustering** to segment customers based on their predicted CLV and RFM scores.
- Segment customers into actionable groups (e.g., high-value loyal customers, at-risk customers) to inform marketing strategies.

### **Step 5: Build Personalized Product Recommendation System**
- Use different recommendation algorithms tailored to customer segments for personalized product suggestions. See the detailed breakdown in the **Recommendation System** section.

### **Step 6: Model Deployment Using Streamlit**
- **Streamlit App**: Deploy the CLV prediction model and customer insights via an interactive **Streamlit** web application.
  - Users can explore customer analytics by entering a customer ID and selecting from various tasks (CLTV analysis, segmentation, recommendations).
  - Check out the deployed app here: [Custalyze Web App](https://customer-analytics-assistant-urzknsdg8udnsjm9x4efhd.streamlit.app/)

---

## **Technologies and Tools**

- **Data Processing**: Pandas, NumPy
- **Modeling**: Lifetimes (BG/NBD, Gamma-Gamma models), Scikit-learn
- **Customer Segmentation**: K-Means Clustering
- **Recommendation System**: 
  - Content-Based Filtering
  - Collaborative Filtering
  - Association Rule-Based Systems
- **Visualization and Deployment**: matplotlib, seaborn, Streamlit
---

## **Future Enhancements**

- **Real-Time CLV Prediction**: Implement real-time data streaming to update CLV predictions dynamically as customer behavior changes.
- **A/B Testing**: Integrate A/B testing to measure the effectiveness of personalized product recommendations and marketing strategies.
- **Advanced Dashboard**: Add more visualizations and dynamic dashboards to track customer engagement, CLV trends, and marketing performance.

---

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
