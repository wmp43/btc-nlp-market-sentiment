import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from sklearn.metrics import make_scorer




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
    #pca = PCA(n_components=2)
    #tfidf_reduced = pca.fit_transform(tfidf_matrix)
    #tfidf_df = pd.DataFrame(tfidf_reduced, columns=['pc1', 'pc2'])
    
    tfidf_df = pd.DataFrame(tfidf_matrix)
    tfidf_df['date'] = df.index
    
    tfidf_df.to_csv('tf_idf_df.csv', index=False)
    
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
    return new_df

def join_tfidf_w2v(tfidf_data, w2v_data, df):
    new_df = pd.DataFrame({
        'active_wallets': df['active_count'],
        'price_close': df['price_usd_close'],
        'returns': df['price_usd_close'].pct_change(),
        'returns_tomorrow': df['price_usd_close'].pct_change().shift(-1),
        'word2vec': w2v_data['word2vec']})
    
    merged_df = pd.concat([tfidf_data, new_df], axis = 1)
    merged_df.dropna(inplace=True)
    merged_df.to_csv('final_data/processed_tfidf_w2v.csv')
    return merged_df

def perform_grid_search(df):
    features = df.drop(columns =  'returns_tomorrow')
    target = df['returns_tomorrow']

    param_grid = {
    'n_estimators': [175],
    'max_depth': [4],
    'learning_rate': [0.025],
    'subsample': [0.6],
    'colsample_bytree': [0.6],
    'reg_alpha': [0.2],
    'reg_lambda': [0.4]}

    model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric=['rmse'])

    tscv = TimeSeriesSplit(n_splits=2)
    
    # Define custom scoring metrics
    scoring = {'RMSE': 'neg_root_mean_squared_error',
               'MAPE': make_scorer(mean_absolute_percentage_error, greater_is_better=False),
               'R2': 'r2'}
    
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
    scoring=scoring, 
    cv=tscv,
    verbose=2,
    refit='RMSE')  # Refit an estimator using the best found parameters on the whole dataset.

    grid_search.fit(features, target)

    print("Best parameters: ", grid_search.best_params_)
    print("Best RMSE score on validation set: ", -grid_search.best_score_)
    print("Best MAPE score on validation set: ", -grid_search.cv_results_['mean_test_MAPE'][grid_search.best_index_])
    print("Best R2 score on validation set: ", grid_search.cv_results_['mean_test_R2'][grid_search.best_index_])
    
    # Save the best model as a pickle file
    joblib.dump(grid_search.best_estimator_, 'xgb_btc_model1.pkl')

    return grid_search.best_params_, -grid_search.best_score_, -grid_search.cv_results_['mean_test_MAPE'][grid_search.best_index_], grid_search.cv_results_['mean_test_R2'][grid_search.best_index_]

def evaluate_naive_approach(test_df):
    y_pred_naive = test_df['returns'].shift(-1).dropna()
    y_true_naive = test_df['returns'][:-1]

    rmse_naive = mean_squared_error(y_true_naive, y_pred_naive, squared=False)
    mape_naive = mean_absolute_percentage_error(y_true_naive, y_pred_naive)
    r2_naive = r2_score(y_true_naive, y_pred_naive)


    print("RMSE - Naive Approach:", rmse_naive)
    print("MAPE - Naive Approach:", mape_naive)
    print("R^2 - Naive Approach:", r2_naive)
   
    return rmse_naive, mape_naive, r2_naive

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



text_df = preprocess_text('/Users/owner/myles-personal-env/Projects/text-btc-app/final_data/joined_day_per_observation.csv')
tfidf = compute_tfidf(text_df)
w2v = compute_word2vec(text_df)
processed_df = join_tfidf_w2v(tfidf, w2v, text_df)
best_params, best_score, best_mape, best_r2 = perform_grid_search(processed_df)
evaluate_naive_approach(processed_df)