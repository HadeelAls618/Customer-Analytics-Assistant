
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from collections import defaultdict
from efficient_apriori import apriori as efficient_apriori

# Segment 0 - Best Sellers and Similar Items
def recommend_best_sellers(segment_data, customer_history, top_n=5):
    best_sellers = segment_data.groupby('Description')['Quantity'].sum().sort_values(ascending=False)
    best_sellers = best_sellers[~best_sellers.index.isin(customer_history)]
    return best_sellers.head(top_n).index.tolist()

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
    sim_scores = sorted(list(enumerate(similarity[idx])), key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    return [index_to_item[i[0]] for i in sim_scores]

def combined_recommendation_segment0(segment_data, user_id, top_n_best_sellers=5, top_n_similar=5):
    user_data = segment_data[segment_data['Customer ID'] == user_id]
    if user_data.empty:
        return {"Best-Sellers": [], "Personalized Recommendations": []}

    customer_history = user_data['Description'].unique()
    best_sellers = recommend_best_sellers(segment_data, customer_history, top_n=top_n_best_sellers)
    if best_sellers:
        sample_item = best_sellers[0]
        similar_items = recommend_similar_items(segment_data, sample_item, top_n=top_n_similar)
    else:
        similar_items = []

    return {
        'Best-Sellers': best_sellers,
        'Personalized Recommendations': similar_items
    }

# Segment 1 - Collaborative Filtering
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

    user_idx = user_item_matrix.index.get_loc(user_id)
    sim_scores = sorted(list(enumerate(user_similarity[user_idx])), key=lambda x: x[1], reverse=True)
    similar_users = [user_item_matrix.index[i[0]] for i in sim_scores]
    recommended_items = segment_data[
        (segment_data['Customer ID'].isin(similar_users)) & 
        (~segment_data['Description'].isin(customer_history))
    ]['Description'].value_counts().head(top_n).index.tolist()

    return recommended_items

def combined_recommendation_segment1(segment_data, user_id, last_purchased_item, top_n=5):
    user_item_matrix = create_user_item_matrix(segment_data)
    latent_matrix = apply_svd(user_item_matrix)
    user_similarity = compute_user_similarity(latent_matrix)
    customer_history = segment_data[segment_data['Customer ID'] == user_id]['Description'].unique()
    return recommend_collaborative(user_id, user_item_matrix, user_similarity, segment_data, customer_history, top_n)

# Segment 2 - Cross-Sell and Upsell
def get_customer_purchase_history(segment_data, customer_id):
    customer_data = segment_data[segment_data['Customer ID'] == customer_id].sort_values(by='InvoiceDate', ascending=False)
    return customer_data['Description'].unique()

def preprocess_rules(rules):
    rule_dict = defaultdict(list)
    for rule in rules:
        for lhs_item in rule.lhs:
            rule_dict[lhs_item].append((list(rule.rhs), rule.confidence, rule.lift))
    for lhs_item in rule_dict:
        rule_dict[lhs_item] = sorted(rule_dict[lhs_item], key=lambda x: (x[1], x[2]), reverse=True)
    return rule_dict

def recommend_cross_sell(purchased_item, rule_dict, customer_history, top_n=5):
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

def recommend_upsell(purchased_item, transactions_df, top_n=3, sort_by='Quantity', ascending=False):
    item_prices = transactions_df[transactions_df['Description'] == purchased_item]['Price'].unique()
    if len(item_prices) == 0:
        return []
    item_price = item_prices.max()
    higher_priced_items_df = transactions_df[transactions_df['Price'] > item_price]
    if higher_priced_items_df.empty:
        return []
    higher_priced_popularity = higher_priced_items_df.groupby('Description')['Quantity'].sum().reset_index()
    if sort_by in higher_priced_popularity.columns:
        higher_priced_popularity = higher_priced_popularity.sort_values(by=sort_by, ascending=ascending)
    return higher_priced_popularity.head(top_n)['Description'].tolist()

def combined_recommendation_segment2(segment_data, transactions_df, customer_id, top_n_cross_sell=5, top_n_upsell=3, min_support=0.02, min_confidence=0.4):
    customer_history = get_customer_purchase_history(segment_data, customer_id)
    if len(customer_history) == 0:
        return {'Cross-Sell Recommendations': [], 'Upsell Recommendations': []}

    transactions = segment_data.groupby('Invoice')['Description'].apply(list).tolist()
    num_transactions = len(transactions)
    if num_transactions < 1000:
        min_support, min_confidence = 0.01, 0.2
    elif num_transactions < 5000:
        min_support, min_confidence = 0.015, 0.25

    itemsets, rules_efficient = efficient_apriori(transactions, min_support=min_support, min_confidence=min_confidence)
    if len(rules_efficient) == 0:
        return {'Cross-Sell Recommendations': [], 'Upsell Recommendations': []}

    rule_dict = preprocess_rules(rules_efficient)
    cross_sell_recs = []
    for item in customer_history:
        cross_sell_recs.extend(recommend_cross_sell(item, rule_dict, customer_history, top_n_cross_sell))
    cross_sell_recs = list(dict.fromkeys(cross_sell_recs))[:top_n_cross_sell]

    upsell_recs = []
    for item in customer_history:
        upsell_recs.extend(recommend_upsell(item, transactions_df, top_n=top_n_upsell, sort_by='Quantity', ascending=False))
    upsell_recs = list(dict.fromkeys(upsell_recs))[:top_n_upsell]

    return {
        'Cross-Sell Recommendations': cross_sell_recs,
        'Upsell Recommendations': upsell_recs
    }
