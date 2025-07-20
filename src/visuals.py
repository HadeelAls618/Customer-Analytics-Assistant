
import streamlit as st
from PIL import Image

def apply_custom_styles():
    st.markdown("""
        <style>
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
        h1 { font-size: 3em; }
        p, li {
            font-size: 16px;
            line-height: 1.8;
            color: #555;
            margin-bottom: 15px;
        }
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
        .stSidebar {
            background-color: #eef2f7;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }
        .st-expander {
            background-color: #f0f4f7 !important;
            border-radius: 10px !important;
            padding: 20px !important;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .stTable {
            border-radius: 10px;
            overflow: hidden;
        }
        table td, table th {
            padding: 12px;
        }
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

def setup_sidebar_image(image_path="Application/images/CLV_image2.png"):
    st.markdown("""
        <style>
        [data-testid="stSidebar"] { width: auto; }
        @media (max-width: 768px) { [data-testid="stSidebar"] { width: 200px; } }
        @media (max-width: 480px) { [data-testid="stSidebar"] { width: 150px; } }
        @media (max-width: 320px) { [data-testid="stSidebar"] { width: 120px; } }
        [data-testid="stSidebar"] .stImage {
            display: block;
            margin-left: auto;
            margin-right: auto;
            margin-top: -20px;
        }
        [data-testid="stSidebar"] img {
            width: 200px;
        }
        @media (max-width: 768px) { [data-testid="stSidebar"] img { width: 150px; } }
        @media (max-width: 480px) { [data-testid="stSidebar"] img { width: 100px; } }
        </style>
    """, unsafe_allow_html=True)
    clv_image = Image.open(image_path)
    st.sidebar.image(clv_image, width=200)
