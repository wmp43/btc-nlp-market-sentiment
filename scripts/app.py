import streamlit as st
import pandas as pd
import sqlite3
from config import db_path, rapid_api, table2, clf_path, reg_path
from data_ingestion import btc_fear_greed_idx
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from joblib import load
from datetime import datetime, timedelta
from model_dev import add_classifier_predictions


# Import other required libraries

def predictions_master(clf_model_path, reg_model_path, database_name, table):
    dataf = db_query(database_name, table)
    clf_feature_space = add_classifier_predictions(dataf, clf_model_path)
    reg_model = load(reg_model_path)
    fs = clf_feature_space[['PCA1', 'PCA2', 'price', 'day_of_week', 'lagged_price', 'returns',
                                     'mean_neg_sentiment', 'max_neg_sentiment', 'min_neg_sentiment',
                                     'std_neg_sentiment', '3d_rolling_volatility', '7d_rolling_volatility',
                                     'mean_pos_sentiment', 'max_pos_sentiment', 'min_pos_sentiment',
                                     'std_pos_sentiment', 'clf_predictions']]
    reg_predictions = reg_model.predict(fs)
    return reg_predictions[0]


def db_query(database_path, table_name):
    """
    Get Data From DB to process for Model
    """
    conn = sqlite3.connect(database_path)
    print('opened sqlite connection')
    query = f'SELECT * FROM {table_name}'
    df = pd.read_sql_query(query, conn)
    conn.close()
    print('closed sqlite connection')
    return df


st.set_page_config(layout="wide")  # This should be the first Streamlit command

st.title('BTC Market Sentiment Application')
st.sidebar.header('Navigation')
selection = st.sidebar.radio("Go to",
                             ['About', 'Data Ingestion & Engineering', 'Feature Engineering', 'Data Visualization',
                              'Model Development', 'Predictions'])

# Data Ingestion & Engineering, Feature Engineering, Model Development
data = db_query(db_path, table2)

if selection == 'Data Visualization':
    st.header("Data Visualization")
    st.markdown("""
        - 10 day rolling mean of returns with average horizontal line
    """)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(data['time_published'], data['future_returns'].rolling(10).mean())
    avg_rolling_mean = data['future_returns'].rolling(10).mean().mean()
    ax.axhline(y=avg_rolling_mean, color='r', linestyle='--', label=f'Average Returns {round(avg_rolling_mean, 4)}')
    plt.xlabel('Date')
    plt.ylabel('Returns')
    plt.legend()
    plt.title('10 Rolling Mean of Daily Returns')
    st.pyplot(fig)

    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plotting price
    data['time_published'] = pd.to_datetime(data['time_published'])
    ax1.plot(data['time_published'], data['price'], color='b', label='Price')
    ax1.set_xlabel('Year-Month')
    ax1.set_ylabel('Price', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Creating a second axis to plot sentiment
    ax2 = ax1.twinx()
    ax2.plot(data['time_published'], (data['mean_pos_sentiment'] - data['mean_neg_sentiment']).rolling(30).mean(),
             color='r', label='Sentiment Change')
    ax2.set_ylabel('Sentiment Change', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    locator = mdates.AutoDateLocator()
    ax1.xaxis.set_major_locator(locator)

    # Set the formatter to show the dates as YYYY-MM
    formatter = mdates.DateFormatter('%Y-%m')
    ax1.xaxis.set_major_formatter(formatter)

    # Rotating the date labels
    plt.xticks(rotation=45)

    plt.title('Price vs Sentiment Change Over Time')

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Get the Fear and Greed Index value and text
    fgi_val, fgi_txt = btc_fear_greed_idx(rapid_api)

    # Determine the color based on the value
    if fgi_val >= 70:
        color = 'red'
    elif 40 < fgi_val < 70:
        color = 'yellow'
    else:
        color = 'green'

    # Display the value in a large colored font
    st.markdown(f'<h1 style="color:{color};">{fgi_val}</h1>', unsafe_allow_html=True)
    # Display the text description below it
    st.markdown(f'<h3 style="color:{color};">{fgi_txt}</h3>', unsafe_allow_html=True)



elif selection == 'Predictions':
    st.header("Predictions")
    pred = predictions_master(clf_path, reg_path, db_path, table2)
    tomorrow = datetime.now() + timedelta(days=1)
    date_str = tomorrow.strftime("%Y-%m-%d")
    st.header(f'BTC returns for {date_str} will be {round(pred*100,3)}%')



elif selection == 'Data Ingestion & Engineering':
    st.header("Data Retrieval, Processing, and Integration")
    st.markdown("""
This section includes the functions responsible for retrieving data from various sources, extracting relevant text content, and merging the information into a unified structure. It covers the following key steps:

- API Calls: Functions to retrieve historical Bitcoin data and news content from specific URLs.
- Text Extraction: Methods to parse the HTML content of news articles, extracting the main text for further analysis.
- Data Merging and Transformation: Functions to combine the retrieved Bitcoin price data and news content, followed by necessary transformations and storage in an SQLite database.
These collective functions lay the foundation for the entire data processing pipeline, enabling further analysis and visualization in subsequent steps.
    """)
    code_example = """
    import requests
    import json
    import pandas as pd
    import numpy as np
    
    def get_news_data(api_key, time_from):
        '''
        Retrieves news articles related to Bitcoin and other cryptocurrencies, based on a specified time frame.

        This function iteratively makes calls to the Alpha Vantage API to gather news articles related to
        Bitcoin and cryptocurrencies. The retrieved data is appended to a global DataFrame, 'news_data',
        which contains the titles, URLs, and time of publication for each article. 

        The function continues to make API calls until there are fewer than 10 articles in the response,
        ensuring a comprehensive collection of recent news.

        Parameters:
        - api_key (str): The API key for accessing the Alpha Vantage service.
        - time_from (str): The start time for fetching news articles in the format 'YYYY-MM-DDTHH:MM:SS'. 
                        Articles published after this time will be included in the results.

        Returns:
        - news_data (DataFrame): A DataFrame containing the following columns:
            - 'titles': The title of the news article.
            - 'url': The URL link to the full news article.
            - 'time_published': The time the article was published in the format 'YYYY-MM-DDTHH:MM:SS'.

        Note:
        - The global DataFrame 'news_data' will be updated with the retrieved information.
        - Due to API limitations, there is a 17-second wait between consecutive calls to avoid rate limits.
        - As of June 2nd, 2023, the function is expected to return a DataFrame with approximately 3600 rows.
        '''

    def get_historical_btc_data(api_key):
        '''
        Retrieves the historical daily price data for Bitcoin (BTC) in USD from the Alpha Vantage API.

        This function makes a request to the Alpha Vantage API to fetch the full daily historical price data
        for Bitcoin in USD. The retrieved data is structured into a DataFrame, which includes the date and
        closing price for each trading day.

        Parameters:
        - api_key (str): The API key for accessing the Alpha Vantage service.

        Returns:
        - btc_price_data (DataFrame): A DataFrame containing the following columns:
            - 'date': The date of the price data in the format 'YYYY-MM-DD'.
            - 'price': The closing price of Bitcoin in USD on the specified date.

        Exceptions:
        - RequestsException: If the API request encounters any issues, a RequestsException will be printed
                            with details about the error.

        Note:
        - As of the last update, the corresponding API endpoint might be down or changed, and this function 
        may require modification to accommodate any new changes.
        '''

    def merge_data(btc_data, news_data):
        '''
        Merges Bitcoin price data and news data on the basis of date.

        This function takes two DataFrames containing Bitcoin price data and news data,
        and merges them based on the date. The dates in both DataFrames are parsed to 
        the appropriate format before merging. Duplicates are removed based on the 'titles' 
        column, and the 'date' column is dropped from the final result.

        Parameters:
        - btc_data (DataFrame): A DataFrame containing Bitcoin price data. It should have
                                the following columns:
                                - 'date': The date of the price data in the format 'YYYY-MM-DD'.
                                - 'price': The closing price of Bitcoin on the specified date.
        - news_data (DataFrame): A DataFrame containing news information. It should have
                                the following columns:
                                - 'titles': The title of the news article.
                                - 'url': The URL of the news article.
                                - 'time_published': The publication date and time of the news article.

        Returns:
        - merged_data (DataFrame): A DataFrame containing the merged Bitcoin price data 
                                and news information. The merged DataFrame will include 
                                columns from both input DataFrames, excluding the 'date' 
                                column from btc_data.

        Note:
        - The functions 'parse_date' and 'parse_float_date' are expected to be defined 
        elsewhere in the code and are used to parse the dates in the input DataFrames.
        - Find parse_date and parse_float_date at the end of the page

        '''
    """

    code_example1 = """
    from newspaper import Article
    from concurrent.futures import ThreadPoolExecutor
    import sqlite3
    import pandas as pd
    import numpy as np   

    def extract_text(url):
        '''
        Extracts the text content from a given URL.

        This function takes a URL, downloads the corresponding web page using the 'Article' class, 
        and parses the text content. If the extraction fails, an error message is printed.

        Parameters:
        - url (str): The URL of the web page to extract text from.

        Returns:
        - str: The text content of the web page, or None if extraction fails.

        Note:
        - A counter variable is used, but not returned, to count the number of failed extractions.
        Consider revising the logic if the counter is intended to be used outside of this function.
        '''

    def extract_content(df):
        '''
        Extracts the text content for a DataFrame containing URLs.

        This function takes a DataFrame containing URLs and utilizes the 'extract_text'
        function to extract the text content for each URL in parallel. Rows with missing
        content are dropped.

        Parameters:
        - df (DataFrame): A DataFrame containing URLs in the 'url' column.

        Returns:
        - DataFrame: A DataFrame with an additional 'content' column containing the
                    extracted text for each URL.

        Note:
        - The 'extract_text' function is expected to be defined elsewhere in the code.
        - This function uses 'ThreadPoolExecutor' to execute the text extraction in parallel.
        '''
    def df_to_db(merged_data, db_name, raw_table_name):
        '''
        Stores the given DataFrame into an SQLite database.

        This function takes a DataFrame, along with database name and table name,
        and stores the data into the specified SQLite database. The 'price' column 
        is rounded to two decimal places.

        Parameters:
        - merged_data (DataFrame): A DataFrame containing the data to be stored.
        - db_name (str): The name of the SQLite database.
        - raw_table_name (str): The name of the table within the database where the data will be stored.

        Note:
        - The DataFrame is expected to contain a column 'Unnamed: 0', which is renamed to 'Index'.
        - The function uses 'if_exists='append'' to append the data to the existing table.
        '''
    """

    st.markdown("Initial API call for Articles, URLs, and BTC Price. Further merges the data sources")
    st.code(code_example, language='python')
    st.markdown("Extracting Text from URLS and sending extracted text to SQLite DB")
    st.code(code_example1, language='python')

    st.header("Data Retrieval and Transformation for Time Series Analysis")
    st.markdown("""
    The functions in this section are designed to retrieve the processed data from the SQLite database and transform it into a time series format, allowing for more efficient analysis and modeling.
    """)

    code_example2 = """
    import pandas as pd
    import numpy as np
    import sqlite3


    def db_ingestion(db_name, raw_table_name):
        '''
        Retrieves data from a specified SQLite database and table for further processing.
        You'll see this function alot.

        Parameters:
            db_name (str): The name of the SQLite database.
            raw_table_name (str): The name of the table containing the raw data.

        Returns:
            DataFrame: A pandas DataFrame containing the data retrieved from the database.
        '''
    def join_data_one_day_per_row(df):
        '''
        Transforms the DataFrame by concatenating titles and text content to represent
        a single day per row, facilitating time series analysis.

        Parameters:
            df (DataFrame): The original DataFrame containing multiple entries per day.

        Returns:
            DataFrame: A transformed DataFrame with one row per day, with titles and 
            content concatenated and the price represented as the mean for each day.
        '''
    def df_to_db(df, db_name, table_name):
        '''
        Stores the given DataFrame into an SQLite database.

        This function takes a DataFrame, along with database name and table name,
        and stores the data into the specified SQLite database.

        Parameters:
        - df (DataFrame): A DataFrame containing the data to be stored.
        - db_name (str): The name of the SQLite database.
        - table_name (str): The name of the table within the database where the data will be stored.

        Note:
        - The function uses 'if_exists='append'' to append the data to the existing table.
        '''
    """

    st.markdown("Extracting Text from URLS and sending extracted text to SQLite DB")
    st.code(code_example2, language='python')


elif selection == 'Feature Engineering':
    st.header("Text Preprocessing, Lagging/Rolling Features, TF-IDF & Dimensionality Reduction")
    st.markdown("""
    These functions focus on preprocessing the raw textual data and transforming it 
    into meaningful numerical features for further analysis and modeling. 
    This includes tokenization, lemmatization, and the application of TF-IDF with dimensionality reduction. 
    This also includes creating lagging/rolling features for price include volatility, lagged price,
    and a target future returns feature.
    """)
    code_chunk0 = """
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

def preprocess_text(df):
    '''
    Preprocesses the text data by tokenizing, lemmatizing, and combining the title and content.
    
    Parameters:
        df (DataFrame): Original DataFrame containing 'titles' and 'content' columns.

    Returns:
        DataFrame: Modified DataFrame with tokenized, lemmatized text and a new column 'processed_text' containing the final processed version.
    
    Note:
        This function adds the following columns to the DataFrame:
        - 'combined_text': Combined titles and content.
        - 'tokenized_text': Tokenized version of the combined text.
        - 'lemmatized_tokens': Lemmatized version of the tokens.
        - 'processed_text': Final processed text.
    '''
def compute_tfidf(df):
    '''
    Computes the TF-IDF matrix for the preprocessed text and performs dimensionality reduction 
    using PCA.
    
    Parameters:
        df (DataFrame): DataFrame containing the 'processed_text' column, preprocessed by the 
        'preprocess_text' function.

    Returns:
        DataFrame: A DataFrame containing the two main components ('PCA1', 'PCA2') 
        obtained from PCA on the TF-IDF matrix, along with the corresponding 'time_published' column.
    
    Note:
        This function leverages TF-IDF (Term Frequency-Inverse Document Frequency) to create a 
        sparse representation of the text data and then applies PCA to reduce the dimensionality.

    Example:
        compute_tfidf(df)
    '''

def lagging_features(df):
    '''
    Enhances the DataFrame by creating lagging price features, lagging rolling average,
    lagging volatility, and rolling average for volume.

    Parameters:
        df (DataFrame): The original DataFrame containing time series data, such as price, 
        returns, etc.

    Returns:
        DataFrame: The modified DataFrame with new lagging features added.

    Note:
        The function adds the following columns to the DataFrame:
        - 'day_of_week': Day of the week, extracted from the 'time_published' column.
        - 'lagged_price': Price of the previous day.
        - 'returns': Percentage change in price.
        - '3d_rolling_volatility': 3-day rolling standard deviation of returns.
        - '7d_rolling_volatility': 7-day rolling standard deviation of returns.
        - 'future_returns': Shifted future returns to become target.
    '''
"""
    st.code(code_chunk0, language='python')

    code_chunk1 = """
    import pandas as pd
    import numpy as np
    import json
    import requests
    from transformers import BertTokenizer

    def divide_into_chunks(text, tokenizer, max_length=500):
        '''
        Divides a given text into chunks of a specified maximum length.
        
        Parameters:
            text (str): The input text to be chunked.
            tokenizer (Tokenizer): The tokenizer used to split the text into tokens.
            max_length (int, optional): The maximum length of each chunk. Defaults to 500.
        
        Returns:
            list: A list of strings, each representing a chunk of the original text.
        '''
    def get_sentiment(text, cryptobert_url, hugging_face_token):
        '''
        Retrieves the sentiment scores using CryptoBERT API for a given text.

        Parameters:
            text (str): The input text for sentiment analysis.
            cryptobert_url (str): URL to the CryptoBERT API.
            hugging_face_token (str): Authorization token for the Hugging Face platform.

        Returns:
            dict: Dictionary containing the sentiment scores for LABEL_0 and LABEL_1.
        '''
    def update_dataframe_with_sentiment(df, cryptobert_url, hugging_face_token):
        '''
        Processes the content of a DataFrame by dividing it into chunks, obtaining sentiment scores, 
        and updating the DataFrame with the statistics of the sentiment scores.

        Parameters:
            df (DataFrame): The original DataFrame with 'content' column.
            cryptobert_url (str): URL to the CryptoBERT API.
            hugging_face_token (str): Authorization token for the Hugging Face platform.

        Returns:
            DataFrame: Updated DataFrame with new columns representing the mean, max, min, and 
            standard deviation of the sentiment scores for both LABEL_0 and LABEL_1.

        Note:
            The function adds 8 new columns to the DataFrame representing the 
            mean, max, min, and standard deviation of the sentiment scores for both 
            LABEL_0 (negative sentiment) and LABEL_1 (positive sentiment).
        '''
    """
    st.header("Sentiment Analysis (HF) and Preprocessing")
    st.markdown("""
    The following functions are concerned with handling text data to extract sentiment scores using a specific pre-trained model (e.g., CryptoBERT). This involves tokenizing and dividing the text into chunks, getting sentiment scores from an API, and updating a DataFrame with the computed sentiment statistics.
    """)
    st.code(code_chunk1, language='python')

    st.header("Feature Space Visualizations")
    st.markdown("""
    Below are some visualizations of the feature space.
    - Correlation heat map
    - 
    """)
    df = data

    selected_columns = ['PCA1', 'PCA2', 'price', 'day_of_week', 'returns', '3d_rolling_volatility',
                        '7d_rolling_volatility', 'future_returns',
                        'mean_neg_sentiment', 'mean_pos_sentiment']

    corr_matrix = data[selected_columns].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Feature Space Correlation Heatmap')
    st.pyplot(plt)

elif selection == 'Model Development':
    st.header("Classifier Model Development")
    st.markdown("""
These functions cover the classification modeling part of the pipeline, focusing on predicting future returns as discrete classes.
- Labeling returns based on the calculated future returns, and encoding them into numerical classes.
- Splitting the data into training and validation sets, and handling imbalanced classes with SMOTE.
- Performing a grid search on an XGBoost classifier to find the optimal hyperparameters.
- Training the best model and evaluating its performance on the validation set, as well as saving the trained model.
- Adding classifier predictions to the DataFrame using the saved model.
""")
    code_chunk1 = '''
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import joblib

def label_returns(df):
    """
    Classify future returns into three categories: positive, negative, and moderate.
    The returns are then encoded into numerical classes for modeling.
    
    Parameters:
    df: DataFrame containing the future_returns column
    
    Returns:
    DataFrame with additional columns for return_class and return_class_encoded
    """

def perform_clf_grid_search(df):
    """
    Perform grid search to train and tune an XGBoost classifier.
    The model is trained on resampled data (using SMOTE) and evaluated on a validation set.
    
    Parameters:
    df: DataFrame containing the features and target variable
    
    Returns:
    Prints the best hyperparameters, validation accuracy, and saves the best model
    """

def add_classifier_predictions(df, classifier_model_path):
    """
    Load a pre-trained classifier model and use it to make predictions on given features.
    Add these predictions to the DataFrame.

    Parameters:
    df: DataFrame containing the features
    classifier_model_path: Path to the saved classifier model file

    Returns:
    DataFrame with the added classifier predictions column
    """
'''
    st.code(code_chunk1, language='python')

    st.header("Regressor Model Development")
    st.markdown("""
These functions provide the necessary steps for regression model development. 
- Adding classifier predictions to the data as features.
- Dropping unnecessary columns and splitting the data into training and validation sets.
- Performing a grid search on an XGBoost regressor to find the optimal hyperparameters.
- Training the best model with custom scoring metrics including RMSE, MAPE, MAE, and Directional Accuracy.
- Evaluating the best model against the validation set and saving the trained model.
- Implementing a naïve approach for comparison and evaluating it using RMSE and MAPE.
- Evaluating the model's performance with RMSE, MAPE, MAE, and Directional Accuracy, and comparing it with the naïve approach.""")

    code_chunk0 = '''
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
from sklearn.externals import joblib
import numpy as np
import pandas as pd

def perform_reg_grid_search(df, classifier_model_path):
    """
    Conducts a grid search using XGBoost regressor on the dataset after adding classifier predictions.
    Evaluates the best hyperparameters using custom scoring metrics and saves the best model.

    Parameters:
    df: DataFrame containing the features
    classifier_model_path: Path to the saved classifier model file

    Returns:
    Best parameters, Best MAPE score, Best mean_test_MAPE score
    """

def mean_absolute_percentage_error(y_true, y_pred): 
    """
    Calculates the Mean Absolute Percentage Error (MAPE) between true and predicted values.

    Parameters:
    y_true: Array-like, true target values
    y_pred: Array-like, predicted target values

    Returns:
    MAPE as a float value
    """

def directional_accuracy(y_true, y_pred):
    """
    Computes the Directional Accuracy by comparing the sign of the differences between true and predicted values.

    Parameters:
    y_true: Array-like, true target values
    y_pred: Array-like, predicted target values

    Returns:
    Directional Accuracy as a float value
    """

def evaluate_naive_approach(test_df):
    """
    Implements and evaluates a naïve approach that simply predicts the return as the previous day's return.

    Parameters:
    test_df: DataFrame containing the test data

    Returns:
    RMSE and MAPE for the naive approach
    """

def evaluate_model(model, X_val, y_val):
    """
    Evaluates the trained model's performance on validation data using RMSE, MAPE, MAE, and Directional Accuracy.

    Parameters:
    model: Trained model object
    X_val: DataFrame or Array-like, validation features
    y_val: DataFrame or Array-like, validation target values

    Returns:
    RMSE, MAPE, MAE, and Directional Accuracy of the model
    """
    '''
    st.code(code_chunk0, language='python')



elif selection == 'About':
    st.header("About The Project")
    st.markdown("Goal: Create a market sentiment application using text data from publications to "
                "understand the markets sentiment toward BTC and how this may relate with returns"
                " - Update: According to the Terms and Services of a vast majority of the publications that text was "
                "being scraped, it was against terms of service to continue scraping text data from these publications. "
                "Thus, I have decided to stop the system from continuing to scrape text data.")
    st.header("Data Ingestion and Engineering")
    st.markdown("I Sourced article URLs from an API and web scraped these URLs for their main content and titles."
                "After some data manipulation, I was able to store the content of these articles in an SQLite db."
                "I decided on SQLite because it is light weight and realistically there isn't enough data to warrant a "
                "heavier system. Additionally, it would also make a ton of sense to use a vector database here."
                " In order to keep the database up to date and predictions available for each subsequent day, "
                "I decided "
                "to use Airflow for orchestration. "
                "You can see more of the details on the Data Ingestion and Engineering page which shows specific "
                "functions and the documentation for each\n"
                "\nFuture Plans: A wider scope of text data.\n"
                "\nTechnologies: Pandas, SQLite3, newspaper3k, Numpy, Apache Airflow")

    st.header("Feature Engineering")
    st.markdown("Using the web scraped text, I was able to develop some main features for the model. The first method"
                "was to use a TF-IDF Matrix. Given the huge dimensionality that "
                "comes with TF-IDF (Term Frequency-Inverse Document Frequency), "
                "I decided to reduce"
                "the dimensionality using PCA (Principal Component Analysis). The second method was to use "
                "a pre-trained Hugging Face Model for sentiment analysis. This model was specifically tuned for crypto"
                " news (https://huggingface.co/kk08/CryptoBERT). This model outputted sentiment scores "
                "(positive and negative) from"
                " each 500 token chunk. I then developed summary statistics of the sentiment scores (mean, min, max, "
                "std)"
                "of the scores for both positive and negative labels. This created 8 new features for the model."
                " I also created some lagging features like volatility measures, lagging price, and set the target"
                "feature for both the classifier and regressor.\n"
                "\nFuture Plans: Another hugging face model for a wider feature space\n"
                "\nTechnologies: Pandas, Numpy, NLTK, Sklearn, Transformers"
                "")
    st.header("Model Development")
    st.markdown("Using the feature space of lagging features and NLP features (sentiment scores, tf-idf), I decided to"
                " use a stack ensemble approach to predicting next day returns. The first model was a XGB classifier"
                "with the targets being neutral, negative, and positive. The predictions from this model were then used"
                "as a feature for the regressor model. Some techniques used were a time-series grid search for features"
                " The model was tested on a validation set that was not used at all during the train/testing of the "
                "model. This was done to accurately assess generalization power on unseen data. A naive approach"
                " was also implemented to ensure the efficacy of developing a model in the first place.\n"
                "\nFuture Plans: Create a custom objective function.\n"
                "\nTechnologies: Pandas, Numpy, Sklearn, Imbalanced-Learn")

    st.header("About The Author")
    st.markdown("""
        I'm Myles, a recent Cornell Information Science Grad. 
        Interested in end to end data-driven systems. 
        Looking for full time roles related to Data Engineering/Science/Analytics/Product Management.
        Please Reach out to wmp43@cornell.edu for questions/suggestions/opportunities
    """)
