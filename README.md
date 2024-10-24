# **Custalyze: Your Customer Analytics Assistant**

Welcome to **Custalyze**, your go-to analytics assistant for predicting **Customer Lifetime Value (CLV)**, providing personalized product recommendations, and offering deep insights into customer behavior. Simply enter a customer ID, and Custalyze will help you uncover key insights that will enhance your customer engagement and optimize your marketing strategies.

Explore the live app: [Custalyze Web App](https://customer-analytics-assistant-urzknsdg8udnsjm9x4efhd.streamlit.app/)

---

## **Project Overview**

**Custalyze** helps retail businesses leverage customer data to predict **Customer Lifetime Value (CLV)**, understand customer segments, and provide personalized product recommendations. Built using a UK retail dataset, this project enables businesses to optimize their marketing efforts by providing valuable insights into each customer’s potential value and behavior, all through a user-friendly web interface.
---
### **App Features:**
- Predict **Customer Lifetime Value (CLV)** for individual customers.
- Perform **RFM Analysis** to evaluate customer engagement and value.
- **Customer Segmentation** based on predicted CLV and RFM scores for tailored marketing strategies.
- Offer **Personalized Product Recommendations** using advanced recommendation systems based on customer segments.
- Simple web interface that allows users to input a customer ID and retrieve detailed insights instantly.

---

## **Problem Statment**

In today’s highly competitive retail market, understanding customer behavior and predicting future customer value are essential for optimizing marketing strategies and maximizing business growth. Many businesses invest heavily in customer acquisition without fully recognizing the long-term potential of individual customers, leading to inefficient marketing spend and missed opportunities for customer retention.
**Custalyze** addresses these challenges by providing a comprehensive solution that quantifies a customer’s potential and delivers actionable insights. Built using a UK retail dataset, Custalyze empowers businesses to make informed decisions, optimize marketing resources, and retain high-value customers. By offering predictive insights into CLV and personalized strategies for each customer segment, businesses can tailor their marketing efforts, improve customer engagement, and drive sustainable growth.

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

## **Future Enhancements**

- **Real-Time CLV Prediction**: Implement real-time data streaming to update CLV predictions dynamically as customer behavior changes.
- **A/B Testing**: Integrate A/B testing to measure the effectiveness of personalized product recommendations and marketing strategies.
- **Advanced Dashboard**: Add more visualizations and dynamic dashboards to track customer engagement, CLV trends, and marketing performance.

---

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
