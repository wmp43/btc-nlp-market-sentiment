import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import xgboost as xgb

def preprocess_text(data_path):
    df = pd.read_csv(data_path)  # Assuming the data is stored in a CSV file
    lemmatizer = WordNetLemmatizer()
    
    df['tokenized_text'] = df['titles'].apply(lambda text: word_tokenize(text.lower()))
    df['lemmatized_tokens'] = df['tokenized_text'].apply(lambda tokens: [lemmatizer.lemmatize(token) for token in tokens])
    df['processed_text'] = df['lemmatized_tokens'].apply(lambda tokens: ' '.join(tokens))
    df.to_csv('df_text.csv')    
    return df

def compute_tfidf(df):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['processed_text']).toarray()
    tfidf_df = pd.DataFrame(tfidf_matrix)
    tfidf_df['date'] = df.index
    tfidf_df.to_csv('tf_idf_df.csv')
    print(len(tfidf_df))
    return tfidf_df

def compute_word2vec(df):
    preprocessed_sentences = [text.split() for text in df['processed_text']]
    model = Word2Vec(preprocessed_sentences, window=5, min_count=3, workers=4)
    word2vec_vectors = []
    for index, row in df.iterrows():
        processed_text = row['processed_text']
        words = processed_text.split()

        word_vectors = [model.wv[word] for word in words if word in model.wv]
        word2vec_vector = np.mean(word_vectors, axis = 0)
        word2vec_vectors.append(word2vec_vector)

    word2vec_array = np.array(word2vec_vectors, dtype = np.float64).mean(axis = 1)
    new_df = pd.DataFrame({'date': df['date'], 'word2vec': word2vec_array})
    new_df.to_csv('w2v_data.csv', index = False)
    print(len(new_df))
    return new_df

def join_tfidf_w2v(tfidf_data, w2v_data, df):
    new_df = pd.DataFrame({
        'active_wallets': df['active_count'],
        'price_close': df['price_usd_close'],
        'word2vec': w2v_data['word2vec']})
    merged_df = pd.concat([tfidf_data, new_df], axis = 1)
    merged_df.to_csv('final_data/processed_tfidf_w2v.csv')
    print(len(merged_df))
    return merged_df

def perform_grid_search(df):
    features = df.drop(columns = 'price_close')
    target = df['price_close']

    param_grid = {
        'eta': [0.5,0.6,0.7],
        'max_depth': [10,12,14,16],
        'subsample': [0.8],
        'colsample_bytree': [1.0]
    }

    model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric=['rmse'])

    tscv = TimeSeriesSplit(n_splits=3)

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
    scoring='neg_root_mean_squared_error', 
    cv=tscv,
    verbose=2)

    grid_search.fit(features, target)

    print("Best parameters: ", grid_search.best_params_)
    print("Best score on validation set: ", -grid_search.best_score_)
    return grid_search.best_params_, -grid_search.best_score_



text_df = preprocess_text('final_data/joined_day_per_observation.csv')
tfidf = compute_tfidf(text_df)
w2v = compute_word2vec(text_df)
processed_df = join_tfidf_w2v(tfidf, w2v, text_df)
best_params, best_score = perform_grid_search(processed_df)

btc_prices = processed_df['price_close']

btc_std = np.std(btc_prices)
rmse = best_score

if rmse < btc_std:
    print("The model's predictions have a lower error than the standard deviation of Bitcoin prices.")
elif rmse > btc_std:
    print("The model's predictions have a higher error than the standard deviation of Bitcoin prices.")
else:
    print("The model's predictions have a similar error as the standard deviation of Bitcoin prices.")

print("Standard Deviation of Bitcoin Prices:", btc_std)
print("RMSE of Model's Predictions:", rmse)
print("params of best Model:", best_params)