
import pandas as pd

def preprocess_data(data):
    data = data.dropna(subset=['Customer ID', 'Description'])
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
    data['Customer ID'] = data['Customer ID'].astype(int)
    data['TotalPrice'] = data['Quantity'] * data['Price']
    return data
