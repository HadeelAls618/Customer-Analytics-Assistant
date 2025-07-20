
import joblib

# Load models for segmentation
scaler = joblib.load('models/scaled_features.joblib')
kmeans = joblib.load('models/kmeans_model.joblib')

def segment_customer(rfm_data, forecast_period=12):
    if rfm_data.empty:
        return None

    from modelling import bgf, ggf  # Lazy import to avoid circular dependencies

    rfm_data["Discounted CLV"] = ggf.customer_lifetime_value(
        bgf,
        rfm_data['frequency'],
        rfm_data['recency'],
        rfm_data['T'],
        rfm_data['monetary_value'],
        time=forecast_period / 30,
        discount_rate=0.01,
        freq="D"
    )

    scaled_features = scaler.transform(rfm_data[['recency', 'frequency', 'monetary_value', 'Discounted CLV']])
    segment = kmeans.predict(scaled_features)
    return segment[0]
