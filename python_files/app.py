import streamlit as st
import joblib
import pandas as pd
from config import api_key, time_from, output_size, brave_binary, driver_path, glassnode_api_key, rapid_api
from predictions import preprocess_text, compute_tfidf, compute_word2vec, join_tfidf_w2v, mean_absolute_percentage_error, evaluate_naive_approach, perform_grid_search

# Load the saved model
model = joblib.load('xgb_btc_model1.pkl')

from data_ingestion import get_news_data, get_glassnode_data, fetch_all_data, scrape_text
real_time_data = fetch_all_data(api_key, time_from, output_size, glassnode_api_key) 
news_data, glassnode_data, fear_greed_index = real_time_data

from data_processing import parse_date, join_data_one_day_per_row, combined_function, full_pipeline
chunk_size = 200
result = full_pipeline(chunk_size, news_data, glassnode_data)

engineered_data = feature_engineering_function(processed_data)

# Use the model to make predictions
prediction = model.predict(engineered_data)

# Display the prediction
st.write(prediction)
