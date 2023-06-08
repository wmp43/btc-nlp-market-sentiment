'''
- BTC Price Predictions using the URL and Title
- Scikit-Learn Time Series methods
- Maybe a 1d covnet??
'''
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import pandas as pd
import nltk
#nltk.download('punkt')
#nltk.download('wordnet')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import xgboost as xgb
from sklearn.metrics import mean_squared_error

def compute_tfidf_matrix(text_column, input_path):
    dataframe = pd.read_csv(input_path)
    dataframe['titles'] = dataframe['titles'].astype(str)

    dataframe['tokenized_text'] = dataframe['titles'].apply(word_tokenize)

    lemmatizer = WordNetLemmatizer()
    dataframe['lemmatized_text'] = dataframe['tokenized_text'].apply(lambda tokens: [lemmatizer.lemmatize(token) for token in tokens])
    dataframe['processed_text'] = dataframe['lemmatized_text'].apply(' '.join)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(dataframe['processed_text'])

    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    tfidf_df.to_csv('title_tfidf.csv', index=False)
    return print('f yea')


input_path = 'final_data/joined_day_per_observation.csv'
text_column = 'titles'
#compute_tfidf_matrix(text_column, input_path)

def time_series_train_test_split(data, test_size, validate_size):
    num_rows = len(data)
    num_test = int(num_rows * test_size)
    num_validate = int(num_rows * validate_size)
    num_train = num_rows - num_test - num_validate
    
    train_set = data[:num_train]
    test_set = data[num_train:num_train + num_test]
    validate_set = data[num_train + num_test:]
    
    return train_set, test_set, validate_set


#data = pd.read_csv
#train_set, test_set, validate_set = time_series_train_test_split(data, 0.2, 0.1)

def train_and_evaluate_model(train_set, test_set, validate_set):
    # Get the predictor columns from the train_set
    predictors = train_set[['active_count']].copy()

    tfidf_matrix = pd.read_csv('title_tfidf.csv')
    
    # Merge the TF-IDF matrix with the predictor columns
    predictors = predictors.merge(tfidf_matrix, left_index=True, right_index=True)
    
    # Get the target variable
    target = train_set['price_usd_close']
    
    # Convert the predictor and target data to DMatrix format
    train_data = xgb.DMatrix(predictors, label=target)
    test_data = xgb.DMatrix(test_set[predictors.columns])
    validate_data = xgb.DMatrix(validate_set[predictors.columns])
    
    # Set the parameters for XGBoost
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        # Add other parameters as needed
    }
    
    # Train the model
    model = xgb.train(params, train_data)
    
    # Make predictions on the test and validate sets
    test_predictions = model.predict(test_data)
    validate_predictions = model.predict(validate_data)
    
    # Calculate and print the RMSE scores
    test_rmse = mean_squared_error(test_set['price_usd_close'], test_predictions, squared=False)
    validate_rmse = mean_squared_error(validate_set['price_usd_close'], validate_predictions, squared=False)
    
    print("Test RMSE:", test_rmse)
    print("Validate RMSE:", validate_rmse)


df = pd.read_csv('final_data/joined_day_per_observation.csv')
train_set, test_set, validate_set = time_series_train_test_split(df, test_size=0.2, validate_size=0.1)

train_and_evaluate_model(train_set, test_set, validate_set)
