# app/app.py
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# === Import Custom Modules ===
from src.processing import preprocess_data
from src.modelling import compute_rfm, clv_and_custom_forecast
from src.segmentation import segment_customer
from src.recommendation import (
    combined_recommendation_segment0,
    combined_recommendation_segment1,
    combined_recommendation_segment2,
    get_customer_purchase_history
)
from src.visuals import apply_custom_styles, setup_sidebar_image

# === Apply Styles and Load Image ===
apply_custom_styles()
setup_sidebar_image("app/images/CLV_image2.png")

# === Load and Preprocess Data ===
data = pd.read_csv("data/transformed/data_full2.csv")
preprocessed_data = preprocess_data(data)

# === Sidebar Widgets ===
st.sidebar.header("What do you want to know about your customer?")
customer_id = st.sidebar.selectbox("Select Customer ID", preprocessed_data['Customer ID'].unique())
task = st.sidebar.selectbox("Select a Task", ['Introduction', 'CLTV Analysis', 'CLV Prediction', 'Customer Segmentation', 'Product Recommendation'])

# === Main Title ===
st.markdown("""
    <style>
    .main-title {
        font-size: 32px;
        font-weight: bold;
        text-align: center;
        color: #333333;
    }
    </style>
    <h1 class="main-title">Custalyze (your Analytics Assistant!)</h1>
    """, unsafe_allow_html=True)

# === Task: Introduction ===
if task == "Introduction":
    st.markdown("""
    ### Welcome to Custalyze👋
    Custalyze helps you quantify your customer's potential, predict their future value, and optimize your strategies.
    """)
    st.image("app/images/CLV_image.png", use_container_width=True)

# === Task: CLTV Analysis ===
elif task == 'CLTV Analysis':
    st.header(f"CLTV Analysis for Customer {customer_id} 🔍")
    rfm_data = compute_rfm(preprocessed_data, customer_id)

    # Calculate Metrics
    from src.modelling import cltv_c_analysis
    (total_transactions, total_products, total_amount, average_order_value,
     purchase_frequency, customer_value, cltv_c_value) = cltv_c_analysis(preprocessed_data, customer_id)

    recency = rfm_data['recency'].values[0] if not rfm_data.empty else None
    monetary_value = rfm_data['monetary_value'].values[0] if not rfm_data.empty else None
    predicted_alive_prob = None
    if not rfm_data.empty:
        from src.modelling import bgf
        predicted_alive_prob = bgf.conditional_probability_alive(
            rfm_data['frequency'].values[0],
            rfm_data['recency'].values[0],
            rfm_data['T'].values[0]
        )

    churn_risk = "Low"
    if recency > 600 and purchase_frequency < 10:
        churn_risk = "High"
    elif recency > 500 and purchase_frequency < 15:
        churn_risk = "Moderate-High"
    elif recency > 300 and purchase_frequency < 20:
        churn_risk = "Moderate"

    total_revenue = preprocessed_data['TotalPrice'].sum()
    customer_revenue_percent = (total_amount / total_revenue) * 100 if total_revenue > 0 else 0

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Total Transactions**: {total_transactions} 🧾")
        st.markdown(f"**Total Products**: {total_products} 📦")
        st.markdown(f"**Total Amount**: ${total_amount:.2f} 💰")
        st.markdown(f"**Average Order Value**: ${average_order_value:.2f} 🔢")
        st.markdown(f"**Purchase Frequency**: {purchase_frequency:.2f} 🔄")
    with col2:
        st.markdown(f"**Recency**: {recency} days 🕰")
        st.markdown(f"**Monetary Value**: ${monetary_value:.2f} 💵")
        st.markdown(f"**% Revenue Contribution**: {customer_revenue_percent:.2f}% 💼")
        st.markdown(f"**Alive Probability**: {predicted_alive_prob:.2f} 🔮")
        st.markdown(f"**Churn Risk**: {churn_risk} 🚨")

# === Task: CLV Prediction ===
elif task == 'CLV Prediction':
    st.header(f"CLV prediction for Customer {customer_id}")
    forecast_period = st.number_input("Enter prediction Period (in days) 📅", min_value=1, value=365, step=1)
    with st.spinner('Calculating CLV...'):
        clv, predicted_purchases, predicted_monetary_value, forecast_data = clv_and_custom_forecast(preprocessed_data, customer_id, forecast_period)

    if clv is not None and forecast_data is not None and not forecast_data.empty:
        customer_forecast_data = forecast_data[forecast_data['Customer ID'] == customer_id]
        if not customer_forecast_data.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Alive Probability**: {customer_forecast_data['Predicted Probability Of Customer Alive'].values[0]:.2f} 🧭")
                st.markdown(f"**Predicted Purchases**: {customer_forecast_data['Predicted Purchases'].values[0]:.2f} 🛒")
            with col2:
                st.markdown(f"**Monetary Per Transaction**: ${customer_forecast_data['Predicted Monetary Value Per Transaction'].values[0]:.2f} 💵")
                st.markdown(f"**Predicted CLV**: ${customer_forecast_data['Discounted CLV'].values[0]:,.2f} 💰")
        else:
            st.warning("No data available for this period.")
    else:
        st.warning("Insufficient data to predict CLV.")

# === Task: Customer Segmentation ===
elif task == 'Customer Segmentation':
    st.markdown(f"### Customer Segmentation & Strategy for Customer {customer_id} 🔍")
    rfm_data = compute_rfm(preprocessed_data, customer_id)
    segment = segment_customer(rfm_data)
    if segment is not None:
        segments = {
            0: ("Low CLV, Low Engagement", "Focus on onboarding and re-engagement."),
            1: ("High CLV, High Engagement", "Retain top customers with exclusives."),
            2: ("Moderate CLV, Moderate Engagement", "Upsell and cross-sell to increase value.")
        }
        title, strategy = segments.get(segment, ("Unknown", "N/A"))
        st.markdown(f"**Segment {segment}: {title}**")
        st.markdown(f"**Strategy**: {strategy}")
    else:
        st.warning("No RFM data found.")

# === Task: Product Recommendation ===
elif task == 'Product Recommendation':
    st.header(f"Personalized Recommendations for Customer {customer_id} 🛍")
    rfm_data = compute_rfm(preprocessed_data, customer_id)
    segment = segment_customer(rfm_data)
    if segment is not None:
        customer_history = get_customer_purchase_history(preprocessed_data, customer_id)
        last_purchased_item = customer_history[0] if len(customer_history) > 0 else None
        st.markdown(f"**Last Purchased Item**: {last_purchased_item}")
        with st.spinner('Generating recommendations...'):
            if segment == 0:
                recs = combined_recommendation_segment0(preprocessed_data, customer_id)
                st.markdown(f"**Best-Sellers**: {', '.join(recs['Best-Sellers'])} ")
                st.markdown(f"**Personalized**: {', '.join(recs['Personalized Recommendations'])} ")
            elif segment == 1:
                recs = combined_recommendation_segment1(preprocessed_data, customer_id, last_purchased_item)
                st.markdown(f"**Exclusive Offers**: {', '.join(recs)} ")
            elif segment == 2:
                recs = combined_recommendation_segment2(preprocessed_data, preprocessed_data, customer_id)
                st.markdown(f"**Cross-Sell**: {', '.join(recs['Cross-Sell Recommendations'])} ")
                st.markdown(f"**Upsell**: {', '.join(recs['Upsell Recommendations'])} ⬆")
