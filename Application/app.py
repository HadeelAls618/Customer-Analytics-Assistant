
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes import BetaGeoFitter, GammaGammaFitter
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from collections import defaultdict
from efficient_apriori import apriori as efficient_apriori


# Load pre-trained models
bgf = joblib.load('Application/models/bgf_model.joblib')
ggf = joblib.load('models/ggf_model.joblib')
scaler = joblib.load('models/scaled_features.joblib')
kmeans = joblib.load('models/kmeans_model.joblib')

# Load customer transaction data
data = pd.read_csv('Dataset/data_full2.csv')

# Data preprocessing function
def preprocess_data(data):
    data = data.dropna(subset=['Customer ID', 'Description'])
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
    data['Customer ID'] = data['Customer ID'].astype(int)  # Ensure Customer ID is an integer
    data['TotalPrice'] = data['Quantity'] * data['Price']  # Ensure TotalPrice is computed
    return data

# Preprocess association rules into a dictionary for faster lookup
def preprocess_rules(rules):
    rule_dict = defaultdict(list)
    for rule in rules:
        for lhs_item in rule.lhs:
            rule_dict[lhs_item].append((list(rule.rhs), rule.confidence, rule.lift))

    # Sort the recommendations in each list by confidence and then by lift
    for lhs_item in rule_dict:
        rule_dict[lhs_item] = sorted(rule_dict[lhs_item], key=lambda x: (x[1], x[2]), reverse=True)

    return rule_dict

# Function for RFM Analysis
def compute_rfm(customer_data, customer_id, max_date=None):
    if not max_date:
        max_date = customer_data['InvoiceDate'].max()
    customer_data = customer_data[customer_data['Customer ID'] == customer_id]
    rfm = summary_data_from_transaction_data(
        customer_data,
        customer_id_col='Customer ID',
        datetime_col='InvoiceDate',
        monetary_value_col='TotalPrice',
        freq='D',
        include_first_transaction=False
    )
    return rfm


def clv_and_custom_forecast(data, customer_id, forecast_period=365):
    fixed_discount_rate = 0.01  # Fixed discount rate

    # 1. Calculate RFM data for the specific customer using the compute_rfm function
    rfm_data = compute_rfm(data, customer_id)

    if not rfm_data.empty:
        # Convert scalar values into pandas Series with a single row, indexed by customer_id
        frequency = pd.Series(rfm_data['frequency'].iloc[0], index=[customer_id])
        recency = pd.Series(rfm_data['recency'].iloc[0], index=[customer_id])
        monetary_value = pd.Series(rfm_data['monetary_value'].iloc[0], index=[customer_id])
        T = pd.Series(rfm_data['T'].iloc[0], index=[customer_id])

        # 2. Predict CLV for the next year using the BG/NBD and Gamma-Gamma models
        predicted_purchases = bgf.predict(30, frequency, recency, T)
        predicted_monetary_value = ggf.conditional_expected_average_profit(frequency, monetary_value)

        # 3. Compute CLV based on the forecast period (in months)
        clv = ggf.customer_lifetime_value(
            bgf,
            frequency,
            recency,
            T,
            monetary_value,
            time=forecast_period / 30,  # Converting forecast period to months
            discount_rate=fixed_discount_rate,
            freq="D"  # Frequency in days
        )

        # 4. Create a custom forecast output for the customer using transaction data
        full_data = summary_data_from_transaction_data(
            data,
            customer_id_col='Customer ID',
            datetime_col='InvoiceDate',
            monetary_value_col='TotalPrice',
            freq='D',
            include_first_transaction=False
        ).reset_index()

        # 5. Predict the probability of customer being alive based on transaction data
        full_data["Predicted Probability Of Customer Alive"] = bgf.conditional_probability_alive(
            frequency=full_data['frequency'],
            recency=full_data['recency'],
            T=full_data['T']
        ).round(3)

        # 6. Predict purchases for the specified forecast period
        full_data["Predicted Purchases"] = bgf.predict(
            t=forecast_period, 
            frequency=full_data['frequency'], 
            recency=full_data['recency'], 
            T=full_data['T']
        ).round(2)

        # 7. Predict monetary value per transaction
        full_data["Predicted Monetary Value Per Transaction"] = ggf.conditional_expected_average_profit(
            full_data['frequency'], full_data['monetary_value']
        ).round(2)

        # 8. Compute the discounted CLV for the forecast period
        full_data["Discounted CLV"] = ggf.customer_lifetime_value(
            bgf,
            full_data['frequency'],
            full_data['recency'],
            full_data['T'],
            full_data['monetary_value'],
            time=forecast_period / 30,  # Converting forecast period to months
            discount_rate=fixed_discount_rate,
            freq="D"
        ).round(2)

        return clv, predicted_purchases, predicted_monetary_value, full_data

    # 9. Return None if no data is available for the customer
    return None, None, None, None

# Function to calculate CLTV-C Analysis
def cltv_c_analysis(data, customer_id, profit_margin=0.10):
    # Filter data for the specific customer
    customer_data = data[data['Customer ID'] == customer_id]

    # Check if the customer_data is empty (no transactions for the customer)
    if customer_data.empty:
        return 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0

    # Ensure InvoiceDate is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(customer_data['InvoiceDate']):
        customer_data['InvoiceDate'] = pd.to_datetime(customer_data['InvoiceDate'], errors='coerce')

    # Drop any rows where InvoiceDate is NaT (invalid date)
    customer_data = customer_data.dropna(subset=['InvoiceDate'])

    # Check again if there are valid transactions after date cleaning
    if customer_data.empty:
        return 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0

    # Total Number of Transactions
    total_transactions = customer_data['Invoice'].nunique()

    # Total Number of Products
    total_products = customer_data['Quantity'].sum()

    # Total Amount (sum of total price for all products purchased by this customer)
    total_amount = customer_data['TotalPrice'].sum()

    # Average Order Value (Total Amount divided by number of transactions)
    average_order_value = total_amount / total_transactions if total_transactions > 0 else 0

    # Calculate time period for this customer's transactions (in days)
    # Find the min and max dates for this specific customer
    min_date = customer_data['InvoiceDate'].min()
    max_date = customer_data['InvoiceDate'].max()

    # Ensure we have a valid time period (minimum 1 day)
    time_period_days = (max_date - min_date).days if min_date != max_date else 1  # Avoid division by zero

    # Purchase Frequency (calculated for the specific customer over their transaction period)
    purchase_frequency = total_transactions / time_period_days if time_period_days > 0 else 0

    # Customer Value (Average order value * purchase frequency)
    customer_value = average_order_value * purchase_frequency

    # CLTV-C Value (Customer Value multiplied by Profit Margin)
    cltv_c_value = customer_value * profit_margin

    return total_transactions, total_products, total_amount, average_order_value, purchase_frequency, customer_value, cltv_c_value



# Function to segment customers
def segment_customer(rfm_data, forecast_period=12):  # Added forecast_period as parameter
    if rfm_data.empty:
        return None

    # Predict Discounted CLV using the forecast period
    rfm_data["Discounted CLV"] = ggf.customer_lifetime_value(
        bgf,
        rfm_data['frequency'],
        rfm_data['recency'],
        rfm_data['T'],
        rfm_data['monetary_value'],
        time=forecast_period / 30,  # Convert forecast period (days) to months
        discount_rate=0.01  # Fixed discount rate
    )

    scaled_features = scaler.transform(rfm_data[['recency', 'frequency', 'monetary_value', 'Discounted CLV']])

    segment = kmeans.predict(scaled_features)

    return segment[0]


# Function to provide strategies based on segment
def recommend_strategy(segment):
    strategies = {
        0: ("Low CLV, Low Engagement", "Focus on onboarding and re-engagement with personalized offers."),
        1: ("High CLV, High Engagement", "Retain high-value customers with exclusive offers."),
        2: ("Moderate CLV, Moderate Engagement", "Increase engagement to prevent churn.")
    }
    return strategies.get(segment, ("Unknown Segment", "No specific strategy available."))

# Function to recommend best-sellers excluding customer history
def recommend_best_sellers(segment_data, customer_history, top_n=5):
    best_sellers = segment_data.groupby('Description')['Quantity'].sum().sort_values(ascending=False)
    best_sellers = best_sellers[~best_sellers.index.isin(customer_history)]  # Exclude items in customer history
    return best_sellers.head(top_n).index.tolist()

# Function to recommend similar items using TF-IDF for personalized offers
def recommend_similar_items(segment_data, item_description, top_n=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    unique_items = segment_data['Description'].unique()
    item_matrix = vectorizer.fit_transform(unique_items)
    similarity = cosine_similarity(item_matrix)

    item_to_index = {item: idx for idx, item in enumerate(unique_items)}
    index_to_item = {idx: item for item, idx in item_to_index.items()}

    idx = item_to_index.get(item_description)
    if idx is None:
        return []
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]  # Exclude the item itself
    return [index_to_item[i[0]] for i in sim_scores]

# Combined recommendation function for Segment 0
def combined_recommendation_segment0(segment_data, user_id, top_n_best_sellers=5, top_n_similar=5):
    """
    Generate combined best-seller and personalized recommendations for a user based on training data.

    Parameters:
    - segment_data: DataFrame for the specific segment.
    - user_id: The Customer ID for whom to generate recommendations.
    - top_n_best_sellers: Number of best-sellers to recommend.
    - top_n_similar: Number of similar items to recommend.

    Returns:
    - Dictionary with 'Best-Sellers' and 'Personalized Recommendations'.
    """
    # Get user purchase history data
    user_data = segment_data[segment_data['Customer ID'] == user_id]
    if user_data.empty:
        return {"Best-Sellers": [], "Personalized Recommendations": []}

    # Get customer purchase history
    customer_history = user_data['Description'].unique()

    # Recommend best-sellers excluding customer history
    best_sellers = recommend_best_sellers(segment_data, customer_history, top_n=top_n_best_sellers)

    # Step 2: Recommend similar items based on a best-seller (e.g., first best-seller)
    if best_sellers:
        sample_item = best_sellers[0]  # Pick the first best-seller for personalized recommendations
        similar_items = recommend_similar_items(segment_data, sample_item, top_n=top_n_similar)
    else:
        similar_items = []

    # Combine both recommendations
    recommendations = {
        'Best-Sellers': best_sellers,
        'Personalized Recommendations': similar_items
    }

    return recommendations

# Collaborative Filtering for Segment 1
def create_user_item_matrix(segment_data):
    return segment_data.pivot_table(index='Customer ID', columns='Description', values='Quantity', fill_value=0)

def apply_svd(user_item_matrix, n_components=100):
    svd = TruncatedSVD(n_components=n_components)
    return svd.fit_transform(user_item_matrix)

def compute_user_similarity(latent_matrix):
    return cosine_similarity(latent_matrix)

def recommend_collaborative(user_id, user_item_matrix, user_similarity, segment_data, customer_history, top_n=5):
    if user_id not in user_item_matrix.index:
        return []

    # Get the user's index
    user_idx = user_item_matrix.index.get_loc(user_id)

    # Compute similarity scores for all users
    sim_scores = list(enumerate(user_similarity[user_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get similar users based on similarity scores
    similar_users = [user_item_matrix.index[i[0]] for i in sim_scores]

    # Recommend items bought by similar users but exclude already purchased items
    recommended_items = segment_data[
        (segment_data['Customer ID'].isin(similar_users)) &
        (~segment_data['Description'].isin(customer_history))
    ]['Description'].value_counts().head(top_n).index.tolist()

    return recommended_items

# Combined recommendation function for Segment 1
def combined_recommendation_segment1(segment_data, user_id, last_purchased_item, top_n=5):
    # Create user-item matrix
    user_item_matrix = create_user_item_matrix(segment_data)

    # Apply SVD to reduce dimensionality
    latent_matrix = apply_svd(user_item_matrix)

    # Compute similarity between users
    user_similarity = compute_user_similarity(latent_matrix)

    # Get customer purchase history
    customer_history = segment_data[segment_data['Customer ID'] == user_id]['Description'].unique()

    # Generate collaborative recommendations for the user, excluding already purchased items
    collab_recommendations = recommend_collaborative(user_id, user_item_matrix, user_similarity, segment_data, customer_history, top_n)

    return collab_recommendations

# -------------------------------
# Segment 2: Upselling, Cross-selling using Association Rule-Based Systems and Collaborative Filtering
# -------------------------------

# Cross-Selling Functions

def get_customer_purchase_history(segment_data, customer_id):
    """
    Returns the entire purchase history for the given customer.
    """
    customer_data = segment_data[segment_data['Customer ID'] == customer_id].sort_values(by='InvoiceDate', ascending=False)
    return customer_data['Description'].unique()

def recommend_cross_sell(purchased_item, rule_dict, customer_history, top_n=5):
    """
    Recommend cross-sell items based on preprocessed association rules, excluding customer history.
    """
    recommendations = []
    if purchased_item in rule_dict:
        for rhs, conf, lift in rule_dict[purchased_item]:
            for item in rhs:
                if item not in customer_history and item not in recommendations:
                    recommendations.append(item)
                if len(recommendations) >= top_n:
                    break
            if len(recommendations) >= top_n:
                break
    return recommendations[:top_n]

# Upselling Functions

def recommend_upsell(purchased_item, transactions_df, top_n=3, sort_by='Quantity', ascending=False):
    """
    Recommends higher-priced and popular items for upselling.

    Parameters:
    - sort_by: Column to sort the items for upselling. Default is 'Quantity'.
    - ascending: Sort order. For popularity, typically descending order.
    """
    # Get unique prices for the purchased item
    item_prices = transactions_df[transactions_df['Description'] == purchased_item]['Price'].unique()
    if len(item_prices) == 0:
        return []
    item_price = item_prices.max()  # Assuming max price if multiple exist

    # Find higher-priced items
    higher_priced_items_df = transactions_df[transactions_df['Price'] > item_price]

    if higher_priced_items_df.empty:
        return []

    # Aggregate popularity (e.g., total Quantity sold)
    higher_priced_popularity = higher_priced_items_df.groupby('Description')['Quantity'].sum().reset_index()

    # Sort by specified criteria
    if sort_by in higher_priced_popularity.columns:
        higher_priced_popularity = higher_priced_popularity.sort_values(by=sort_by, ascending=ascending)

    # Recommend top N higher-priced and popular items
    recommended_items = higher_priced_popularity.head(top_n)['Description'].tolist()

    return recommended_items

# Combined Recommendation Function for Segment 2
def combined_recommendation_segment2(segment_data, transactions_df, customer_id, top_n_cross_sell=5, top_n_upsell=3, min_support=0.02, min_confidence=0.4):
    """
    Generate combined cross-selling and upselling recommendations for Segment 2 based on the customer's purchase history.
    """
    # Step 1: Get customer purchase history (all items purchased by this customer)
    customer_history = get_customer_purchase_history(segment_data, customer_id)

    # If customer has no purchase history, return an empty list
    if len(customer_history) == 0:
        return {
            'Cross-Sell Recommendations': [],
            'Upsell Recommendations': []
        }

    # Step 2: Generate association rules using efficient_apriori
    transactions = segment_data.groupby('Invoice')['Description'].apply(list).tolist()

    # Dynamically adjust support and confidence based on number of transactions
    num_transactions = len(transactions)
    adjusted_support = min_support
    adjusted_confidence = min_confidence
    if num_transactions < 1000:
        adjusted_support = 0.01
        adjusted_confidence = 0.2
    elif num_transactions < 5000:
        adjusted_support = 0.015
        adjusted_confidence = 0.25

    # Apply efficient Apriori with dynamic thresholds
    itemsets, rules_efficient = efficient_apriori(transactions, min_support=adjusted_support, min_confidence=adjusted_confidence)

    if len(rules_efficient) == 0:
        return {
            'Cross-Sell Recommendations': [],
            'Upsell Recommendations': []
        }

    # Step 3: Preprocess rules for efficient lookup
    rule_dict = preprocess_rules(rules_efficient)

    # Step 4: Collect cross-sell recommendations based on customer history
    cross_sell_recs = []
    for item in customer_history:
        recs = recommend_cross_sell(item, rule_dict, customer_history, top_n_cross_sell)
        cross_sell_recs.extend(recs)

    # Deduplicate cross-sell recommendations
    cross_sell_recs = list(dict.fromkeys(cross_sell_recs))[:top_n_cross_sell]

    # Step 5: Collect upsell recommendations based on customer history
    upsell_recs = []
    for item in customer_history:
        recs = recommend_upsell(item, transactions_df, top_n=top_n_upsell, sort_by='Quantity', ascending=False)
        upsell_recs.extend(recs)

    # Deduplicate upsell recommendations
    upsell_recs = list(dict.fromkeys(upsell_recs))[:top_n_upsell]

    return {
        'Cross-Sell Recommendations': cross_sell_recs,
        'Upsell Recommendations': upsell_recs
    }
st.markdown("""
    <style>
    /* Body and Font Styling */
    body {
        background: linear-gradient(135deg, #f8f9fa, #d1e8ff);
        font-family: 'Roboto', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        text-align: center;
        color: #2c3e50;
        font-weight: bold;
        margin-bottom: 10px;
    }
    h1 {
        font-size: 3em;
    }
    p, li {
        font-size: 16px;
        line-height: 1.8;
        color: #555;
        margin-bottom: 15px;
    }

    /* Button Styling */
    .stButton > button {
        background-color: #007BFF;
        color: white;
        font-size: 18px;
        padding: 12px 24px;
        border-radius: 8px;
        border: none;
        transition: background-color 0.3s ease, transform 0.2s ease;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stButton > button:hover {
        background-color: #0056b3;
        transform: scale(1.05);
    }

    /* Sidebar Styling */
    .stSidebar {
        background-color: #eef2f7;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }

    /* Expander Styling */
    .st-expander {
        background-color: #f0f4f7 !important;
        border-radius: 10px !important;
        padding: 20px !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }

    /* Enhanced Table */
    .stTable {
        border-radius: 10px;
        overflow: hidden;
    }
    table td, table th {
        padding: 12px;
    }

    /* Aligning Graphics */
    .graphics-container {
        display: flex;
        justify-content: space-around;
        margin: 40px 0;
    }
    .graphics-container img {
        width: 150px;
        height: auto;
    }

    </style>
""", unsafe_allow_html=True)

from PIL import Image

# Load the images
clv_image = Image.open("images/CLV_image2.png")

# Sidebar Image with Custom Width and Centering
st.sidebar.markdown(
    """
    <style>
    [data-testid="stSidebar"] .stImage {
        display: block;
        margin-left: auto;
        margin-right: auto;
        margin-top: -20px;  /* Adjust this value to move the image upwards if needed */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.image(clv_image, width=200)  # Adjust the width to make it smaller



# Sidebar Content
st.sidebar.header("What do you want to know about you customer?")
customer_id = st.sidebar.selectbox("Select Customer ID", data['Customer ID'].unique())
task = st.sidebar.selectbox("Select a Task", ['Introduction','CLTV Analysis', 'CLV Prediction', 'Customer Segmentation', 'Product Recommendation'])

# Main Title
st.title("Custalyze, your Analytics Assistant! ")
# Preprocess data
preprocessed_data = preprocess_data(data)

# Main Content Image with Custom Size
if task == "Introduction":
    # Custom CSS for Title and Text
    st.markdown("""
        <style>
        h1 {
            font-size: 2.5rem;  /* Slightly smaller title font */
            color: #2c3e50;
            text-align: center;
            font-weight: bold;
            margin-bottom: 20px;
        }
        p {
            font-size: 1.2rem;
            color: #555;
            text-align: center;
            line-height: 1.6;
        }
        </style>
    """, unsafe_allow_html=True)


    # Introduction Text
    st.markdown("""
    ### Welcome to Custalyze👋

Many businesses invest in customer acquisition without fully understanding how much each customer will bring in the future, leading to wasted budgets and missed opportunities. That’s where **Custalyze** (Customer + Analyze) steps in.

Custalyze helps you quantify your customer's potential, predict their future value, and optimize your strategies. Proactively engage your audience, prevent churn, and maximize marketing spend for the greatest impact.
    """)

    # Main Image with Controlled Width
    #st.image("images/CLV_image.png", width=500)  # Adjust the width to make the image smaller
    # Main Image with Controlled Width and Centering
    col1, col2, col3 = st.columns([1, 6, 1])

    with col1:
     st.write("")

    with col2:
     st.image("CLV_image.png", width=500)  # Adjust the width to make the image smaller

    with col3:
     st.write("")


# CLTV Analysis Section
elif task == 'CLTV Analysis':
    st.header(f"CLTV Analysis for Customer {customer_id} 🔍")

    # Calculate RFM data for the customer
    rfm_data = compute_rfm(preprocessed_data, customer_id)

    # Calculate CLTV-C Analysis
    total_transactions, total_products, total_amount, average_order_value, purchase_frequency, customer_value, cltv_c_value = cltv_c_analysis(preprocessed_data, customer_id)

    # Extract Recency and Monetary Value from RFM data
    recency = rfm_data['recency'].values[0] if not rfm_data.empty else None
    monetary_value = rfm_data['monetary_value'].values[0] if not rfm_data.empty else None
    purchase_frequency = rfm_data['frequency'].values[0] if not rfm_data.empty else 0

    # Predicted probability of customer being alive
    predicted_alive_prob = bgf.conditional_probability_alive(rfm_data['frequency'].values[0], rfm_data['recency'].values[0], rfm_data['T'].values[0]) if not rfm_data.empty else None

    # Churn risk indicator based on recency and frequency
    if recency > 600 and purchase_frequency < 10:
     churn_risk = "High"  # Very high recency and very low frequency
    elif recency > 500 and purchase_frequency < 15:
     churn_risk = "Moderate-High"  # Above-average recency and below-average frequency
    elif recency > 300 and purchase_frequency < 20:
     churn_risk = "Moderate"  # Moderate recency and frequency
    else:
     churn_risk = "Low"  # Low recency or high frequency


    # Calculate total revenue across all customers
    total_revenue = preprocessed_data['TotalPrice'].sum()  # Assuming 'TotalPrice' is the correct column
    customer_revenue_percent = (total_amount / total_revenue) * 100 if total_revenue > 0 else 0

    # Use two columns to organize the metrics
    col1, col2 = st.columns(2)

    # Display CLTV-C Analysis
    with col1:
        st.markdown(f"**Total Number of Transactions**: {total_transactions} 🧾")
        st.markdown(f"**Total Number of Products**: {total_products} 📦")
        st.markdown(f"**Total Amount**: ${total_amount:.2f} 💰")
        st.markdown(f"**Average Order Value**: ${average_order_value:.2f} 🔢")
        st.markdown(f"**Purchase Frequency**: {purchase_frequency:.2f} 🔄")

    with col2:
        st.markdown(f"**Recency (Days Since Last Purchase)**: {recency} days 🕰")
        st.markdown(f"**Monetary Value per transaction**: ${monetary_value:.2f} 💵")
        st.markdown(f"**Contribution to Total Revenue**: {customer_revenue_percent:.2f}% 💼")
        predicted_alive_prob = predicted_alive_prob[0] if isinstance(predicted_alive_prob, np.ndarray) else predicted_alive_prob
        st.markdown(f"**Predicted Probability of Customer Alive**: {predicted_alive_prob:.2f} 🔮")
        st.markdown(f"**Churn Risk Indicator**: {churn_risk} 🚨")

    # Divider for better sectioning
    st.markdown("---")

elif task == 'CLV Prediction':
    st.header(f"CLV prediction for Customer {customer_id} ")

    # Get user input for the forecast period
    forecast_period = st.number_input("Enter prediction Period (in days) 📅", min_value=1, value=365, step=1)

    # Recalculate the custom forecast based on customer_id and forecast_period
    with st.spinner('Calculating CLV...'):
        clv, predicted_purchases, predicted_monetary_value, forecast_data = clv_and_custom_forecast(preprocessed_data, customer_id, forecast_period)

    # Check for forecast data and display results
    if clv is not None and forecast_data is not None and not forecast_data.empty:
        st.subheader(f"Results for {forecast_period} days:")

        # Filter the forecast data for the specific customer_id
        customer_forecast_data = forecast_data[forecast_data['Customer ID'] == customer_id]

        if not customer_forecast_data.empty:
            # Display the forecast data using columns
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**Predicted Probability Of Customer Alive**: {customer_forecast_data['Predicted Probability Of Customer Alive'].values[0]:.2f} 🧭")
                st.markdown(f"**Predicted Purchases**: {customer_forecast_data['Predicted Purchases'].values[0]:.2f} 🛒")

            with col2:
                st.markdown(f"**Predicted Monetary Per Transaction**: ${customer_forecast_data['Predicted Monetary Value Per Transaction'].values[0]:.2f} 💵")
                st.markdown(f"**Predicted CLV**: ${customer_forecast_data['Discounted CLV'].values[0]:,.2f} 💰")
        else:
            st.warning(f"No data available for Customer {customer_id} for the specified period.")

    else:
        if clv is None:
            st.warning(f"Insufficient data to predict CLV for Customer {customer_id}.")
        else:
            st.warning(f"No data available for Customer {customer_id} for the specified period.")

    # Horizontal line divider for better section separation
    st.markdown("---")



       

# Customer Segmentation & Strategy Section
elif task == 'Customer Segmentation':
    st.markdown(f"### Customer Segmentation & Strategy for Customer {customer_id}🔍")

    rfm_data = compute_rfm(preprocessed_data, customer_id)
    segment = segment_customer(rfm_data)

    if segment is not None:
        # Display segment number and description
        if segment == 0:
            st.markdown(f"**Segment:  {segment} (Low CLV, Low Engagement and Value)** 📉")
            st.markdown("**Characteristics**: Low recency, low frequency, and monetary value indicate new or inactive customers.📉")
            st.markdown("**Action**: Focus on onboarding and re-engagement. Encourage more activity and increase spending. 🎯")
            st.markdown("**Strategies**: Personalized offers and best-sellers can reignite interest or nurture new customers. 💡")

        elif segment == 1:
            st.markdown(f"**Segment:  {segment} (High CLV, High Engagement and Value)** 📈")
            st.markdown("**Characteristics**: High frequency, high monetary value, and high recency indicate top-tier customers. 📈")
            st.markdown("**Action**: Retain these customers by maximizing their value and enhancing loyalty. 🎯")
            st.markdown("**Strategies**: Exclusive offers, early access, and personalized recommendations can keep them loyal. 💡")

        elif segment == 2:
            st.markdown(f"**Segment: {segment} (Moderate CLV, Moderate Engagement and Value)** 📉")
            st.markdown("**Characteristics**: Moderate frequency and monetary value, but high recency indicates churn risk.📉")
            st.markdown("**Action**: Focus on retention and increasing engagement to prevent churn. 🎯")
            st.markdown("**Strategies**: Upselling, cross-selling can help retain them and transition to higher value. 💡")

    else:
        st.warning("No valid RFM data available for this customer.")

    # Optional: Add a divider for better separation
    st.markdown("---")

# Product Recommendation Task
elif task == 'Product Recommendation':
    st.header(f"Personalized Recommendations for Customer {customer_id} 🛍")

    # Compute RFM and determine customer segment
    rfm_data = compute_rfm(preprocessed_data, customer_id)
    segment = segment_customer(rfm_data)

    if segment is not None:
        # Fetch customer purchase history
        customer_history = get_customer_purchase_history(preprocessed_data, customer_id)
        last_purchased_item = customer_history[0] if len(customer_history) > 0 else None

        # Display last purchased item
        st.markdown(f"**Last Purchased Item**: {last_purchased_item} 🛍")

        # Initialize recommended items
        recommended_items = []
    with st.spinner('Generating recommendations...'):
        # Recommend products based on the segment
        if segment == 0:
            segment_recommendations = combined_recommendation_segment0(preprocessed_data, customer_id)
            if segment_recommendations:
                best_sellers = segment_recommendations.get('Best-Sellers', [])
                personalized_recommendations = segment_recommendations.get('Personalized Recommendations', [])
                st.markdown(f"**Best-Sellers**: {', '.join(best_sellers)} 🔥")
                st.markdown(f"**Personalized Recommendations**: {', '.join(personalized_recommendations)} ✨")

        elif segment == 1:
            segment_recommendations = combined_recommendation_segment1(preprocessed_data, customer_id, last_purchased_item)
            if segment_recommendations:
                st.markdown(f"**Exclusive offers Recommendations**: {', '.join(segment_recommendations)} 🤝")
            else:
                st.markdown("No recommendations available for Segment 1.")

        elif segment == 2:
            segment_recommendations = combined_recommendation_segment2(preprocessed_data, preprocessed_data, customer_id)
            cross_sell = segment_recommendations.get('Cross-Sell Recommendations', [])
            upsell = segment_recommendations.get('Upsell Recommendations', [])
            if cross_sell:
                st.markdown(f"**Cross-Sell Recommendations**: {', '.join(cross_sell)} 🔄")
            if upsell:
                st.markdown(f"**Upsell Recommendations**: {', '.join(upsell)} ⬆️")
            if not cross_sell and not upsell:
                st.markdown("No recommendations available for Segment 2.")

