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
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import xgboost as xgb
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import ast





def preprocess_text(texts):
    tokenized_texts = [word_tokenize(text) for text in texts]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [[lemmatizer.lemmatize(token) for token in tokens] for tokens in tokenized_texts]
    processed_texts = [' '.join(tokens) for tokens in lemmatized_tokens]
    model = Word2Vec(lemmatized_tokens, min_count=3)
    word2vec = np.array([model.wv[tokens] for tokens in lemmatized_tokens], dtype=np.float32)
    return processed_texts, word2vec

def compute_tfidf_matrix(processed_texts, word2vec_data):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_texts).toarray()

    word2vec_df = pd.DataFrame({'word2vec': word2vec_data})

    tfidf_w2v_df = pd.DataFrame(tfidf_matrix)
    tfidf_w2v_df = pd.concat([tfidf_w2v_df, word2vec_df], axis=1)
    tfidf_w2v_df.to_csv('tfidf_w2v_data.csv')

    return tfidf_w2v_df


def time_series_train_test_split(data, tfidf_w2v_data, test_size, validate_size):
    num_rows = len(data)
    num_test = int(num_rows * test_size)
    num_validate = int(num_rows * validate_size)

    train_data, temp_data = train_test_split(data, test_size=num_test+num_validate, shuffle=False)
    test_data, validate_data = train_test_split(temp_data, test_size=num_validate, shuffle=False)

    train_set = train_data.copy()
    test_set = test_data.copy()
    validate_set = validate_data.copy()

    # Merge the TF-IDF matrix and Word2Vec data with the corresponding sets
    train_set = pd.merge(train_set[['active_count', 'price_usd_close']], tfidf_w2v_data, left_index=True, right_index=True)
    test_set = pd.merge(test_set[['active_count', 'price_usd_close']], tfidf_w2v_data, left_index=True, right_index=True)
    validate_set = pd.merge(validate_set[['active_count', 'price_usd_close']], tfidf_w2v_data, left_index=True, right_index=True)

    train_set.to_csv('final_data/final_train.csv')
    test_set.to_csv('final_data/final_test.csv')
    validate_set.to_csv('final_data/final_validate.csv')

    return train_set, test_set, validate_set



def perform_grid_search(train_set, test_set):
    # Extract features and target variable from train_set, test_set, and validate_set
    train_features = train_set[['active_count', 'tfidf_matrix', 'word2vec']]
    train_target = train_set['price_usd_close']

    test_features = test_set[['active_count', 'tfidf_matrix', 'word2vec']]
    test_target = test_set['price_usd_close']


    # Define the parameter grid for grid search
    param_grid = {
        'eta': [0.1, 0.2, 0.3],
        'max_depth': [4, 6, 8],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    # Create the XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse')

    tscv = TimeSeriesSplit(n_splits=3)

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=tscv)

    # Fit the grid search on the training data
    grid_search.fit(train_features, train_target)

    # Print the best parameters and best score
    print("Best parameters: ", grid_search.best_params_)
    print("Best score on validation set: ", -grid_search.best_score_)

    # Evaluate the model on the test set using the best parameters
    best_model = grid_search.best_estimator_
    test_score = best_model.score(test_features, test_target)
    print("Score on test set: ", test_score)

    return grid_search.best_params_, -grid_search.best_score_, test_score


df = pd.read_csv('final_data/joined_day_per_observation.csv')
df['processed_text'], df['word2vec'] = zip(*df['titles'].apply(preprocess_text))

tfidf_w2v_data = compute_tfidf_matrix(df['processed_text'], df['word2vec'])

train_set, test_set, validate_set = time_series_train_test_split(df, tfidf_w2v_data, test_size=0.2, validate_size=0.1)

best_params, best_score, test_score = perform_grid_search(train_set, test_set)











# def train_and_evaluate_model(train_set, test_set, validate_set, tfidf_data):
#     # Get the predictor columns from the train_set
#     predictors = train_set[['active_count']].copy()

#     # Merge the Word2Vec data and TF-IDF data with the train, test, and validate sets
#     predictors = predictors.merge(train_set['word2vec_text'], left_index=True, right_index=True)
#     predictors = predictors.merge(tfidf_data, left_index=True, right_index=True)
#     test_set = test_set.merge(test_set['word2vec_text'], left_index=True, right_index=True)
#     test_set = test_set.merge(tfidf_data, left_index=True, right_index=True)
#     validate_set = validate_set.merge(validate_set['word2vec_text'], left_index=True, right_index=True)
#     validate_set = validate_set.merge(tfidf_data, left_index=True, right_index=True)

#     target = train_set['price_usd_close']
    
#     # Convert the predictor and target data to DMatrix format
#     train_data = xgb.DMatrix(train_set[['active_count', 'word2vec_text', '']])
#     test_data = xgb.DMatrix(test_set[predictors.columns], enable_categorical=True)
#     validate_data = xgb.DMatrix(validate_set[predictors.columns], enable_categorical=True)
    
#     # Set the parameters for XGBoost
#     params = {
#         'objective': 'reg:squarederror',
#         'eval_metric': 'rmse',
#         # Add other parameters as needed
#     }
    
#     # Train the model
#     model = xgb.train(params, train_data)
    
#     # Make predictions on the test and validate sets
#     test_predictions = model.predict(test_data)
#     validate_predictions = model.predict(validate_data)
    
#     # Calculate and print the RMSE scores
#     test_rmse = mean_squared_error(test_set['price_usd_close'], test_predictions, squared=False)
#     validate_rmse = mean_squared_error(validate_set['price_usd_close'], validate_predictions, squared=False)
    
#     print("Test RMSE:", test_rmse)
#     print("Validate RMSE:", validate_rmse)



# df = pd.read_csv('final_data/joined_day_per_observation.csv')
# df['processed_text'], df['word2vec_text'] = zip(*df['titles'].apply(preprocess_text))

# tfidf_data = compute_tfidf_matrix(df)

# train_set, test_set, validate_set = time_series_train_test_split(df, tfidf_data, df['word2vec_text'], test_size=0.2, validate_size=0.1)

# train_and_evaluate_model(df, train_set, test_set, validate_set, tfidf_data)

