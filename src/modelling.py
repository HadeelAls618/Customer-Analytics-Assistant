
import joblib
import pandas as pd
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes import BetaGeoFitter, GammaGammaFitter

# Load pre-trained models
bgf = joblib.load('models/bgf_model.joblib')
ggf = joblib.load('models/ggf_model.joblib')

# Compute RFM
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

# CLV and Forecast
def clv_and_custom_forecast(data, customer_id, forecast_period=365):
    fixed_discount_rate = 0.01
    rfm_data = compute_rfm(data, customer_id)

    if not rfm_data.empty:
        frequency = pd.Series(rfm_data['frequency'].iloc[0], index=[customer_id])
        recency = pd.Series(rfm_data['recency'].iloc[0], index=[customer_id])
        monetary_value = pd.Series(rfm_data['monetary_value'].iloc[0], index=[customer_id])
        T = pd.Series(rfm_data['T'].iloc[0], index=[customer_id])

        predicted_purchases = bgf.predict(30, frequency, recency, T)
        predicted_monetary_value = ggf.conditional_expected_average_profit(frequency, monetary_value)

        clv = ggf.customer_lifetime_value(
            bgf,
            frequency,
            recency,
            T,
            monetary_value,
            time=forecast_period / 30,
            discount_rate=fixed_discount_rate,
            freq="D"
        )

        full_data = summary_data_from_transaction_data(
            data,
            customer_id_col='Customer ID',
            datetime_col='InvoiceDate',
            monetary_value_col='TotalPrice',
            freq='D',
            include_first_transaction=False
        ).reset_index()

        full_data["Predicted Probability Of Customer Alive"] = bgf.conditional_probability_alive(
            frequency=full_data['frequency'],
            recency=full_data['recency'],
            T=full_data['T']
        ).round(3)

        full_data["Predicted Purchases"] = bgf.predict(
            t=forecast_period,
            frequency=full_data['frequency'],
            recency=full_data['recency'],
            T=full_data['T']
        ).round(2)

        full_data["Predicted Monetary Value Per Transaction"] = ggf.conditional_expected_average_profit(
            full_data['frequency'], full_data['monetary_value']
        ).round(2)

        full_data["Discounted CLV"] = ggf.customer_lifetime_value(
            bgf,
            full_data['frequency'],
            full_data['recency'],
            full_data['T'],
            full_data['monetary_value'],
            time=forecast_period / 30,
            discount_rate=fixed_discount_rate,
            freq="D"
        ).round(2)

        return clv, predicted_purchases, predicted_monetary_value, full_data

    return None, None, None, None
