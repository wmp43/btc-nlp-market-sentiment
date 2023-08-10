import sqlite3
from config import processed_table_name, db_name, table_name, cryptobert_url, hugging_face_token
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import requests
import json
import time
import numpy as np
from transformers import BertTokenizer


def db_ingestion(db_name, processed_table_name):
    '''
    Get Data From DB to process for Model
    '''
    conn = sqlite3.connect(db_name)
    print('opened sqlite connection')

    query = f'SELECT * FROM {processed_table_name}'

    df = pd.read_sql_query(query, conn)
    conn.close()
    print('closed sqlite connection')
    return df


def preprocess_text(df):
    start_time = time.time()
    lemmatizer = WordNetLemmatizer()
    df['combined_text'] = df['titles'] + ' ' + df['content']
    df['tokenized_text'] = df['combined_text'].apply(lambda text: word_tokenize(text.lower()))
    df['lemmatized_tokens'] = df['tokenized_text'].apply(
        lambda tokens: [lemmatizer.lemmatize(token) for token in tokens])
    df['processed_text'] = df['lemmatized_tokens'].apply(lambda tokens: ' '.join(tokens))

    end_time = time.time()
    duration = end_time - start_time
    print(f'Time taken for preprocessing compute: {round((duration / 60.0), 2)} mins')

    return df


def compute_tfidf(df):
    start_time = time.time()
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['processed_text']).toarray()

    pca = PCA(n_components=2)
    reduced_tfidf = pca.fit_transform(tfidf_matrix)
    tfidf_df = pd.DataFrame(reduced_tfidf, columns=['PCA1', 'PCA2'])
    tfidf_df['time_published'] = df['time_published']

    end_time = time.time()  # Get the end time
    duration = end_time - start_time  # Calculate the duration

    print(f'Time taken for TFIDF: {duration} seconds')

    return tfidf_df


def divide_into_chunks(text, tokenizer, max_length=500):
    tokenized_text = tokenizer.tokenize(text)
    chunks = []

    for i in range(0, len(tokenized_text), max_length):
        chunk = tokenized_text[i:i + max_length]
        chunks.append(tokenizer.convert_tokens_to_string(chunk))

    return chunks


def get_sentiment(text, cryptobert_url, hugging_face_token):
    headers = {"Authorization": f"Bearer {hugging_face_token}", "Content-Type": "application/json"}
    payload = {"inputs": text}

    response = requests.post(cryptobert_url, headers=headers, data=json.dumps(payload))
    response_json = response.json()
    sentiment_scores = {'LABEL_0': 0, 'LABEL_1': 0}
    if response.status_code == 200:
        for item in response_json:
            if item['label'] in sentiment_scores.keys():
                sentiment_scores[item['label']] = item['score']
    else:
        print(f'Code: {response.status_code}, Error: {response.text}')

    return sentiment_scores


def update_dataframe_with_sentiment(df, cryptobert_url, hugging_face_token):
    start_time = time.time()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    df['mean_sentiment_score_LABEL_0'] = np.nan
    df['max_sentiment_score_LABEL_0'] = np.nan
    df['min_sentiment_score_LABEL_0'] = np.nan
    df['std_sentiment_score_LABEL_0'] = np.nan

    df['mean_sentiment_score_LABEL_1'] = np.nan
    df['max_sentiment_score_LABEL_1'] = np.nan
    df['min_sentiment_score_LABEL_1'] = np.nan
    df['std_sentiment_score_LABEL_1'] = np.nan

    for idx, row in df.iterrows():
        chunks = divide_into_chunks(row['content'], tokenizer, max_length=500)
        total_chunks = len(chunks)

        chunk_scores_LABEL_0 = []
        chunk_scores_LABEL_1 = []

        for i, chunk in enumerate(chunks):
            sentiment_scores = get_sentiment(chunk, cryptobert_url, hugging_face_token)
            chunk_scores_LABEL_0.append(sentiment_scores['LABEL_0'])
            chunk_scores_LABEL_1.append(sentiment_scores['LABEL_1'])

            print(f"Processed {i + 1}/{total_chunks} chunks for row {idx + 1}.")

        scores_array_LABEL_0 = np.array(chunk_scores_LABEL_0)
        scores_array_LABEL_1 = np.array(chunk_scores_LABEL_1)

        df.loc[idx, 'mean_neg_sentiment'] = np.mean(scores_array_LABEL_0)
        df.loc[idx, 'max_neg_sentiment'] = np.max(scores_array_LABEL_0)
        df.loc[idx, 'min_neg_sentiment'] = np.min(scores_array_LABEL_0)
        df.loc[idx, 'std_neg_sentiment'] = np.std(scores_array_LABEL_0)

        df.loc[idx, 'mean_pos_sentiment'] = np.mean(scores_array_LABEL_1)
        df.loc[idx, 'max_pos_sentiment'] = np.max(scores_array_LABEL_1)
        df.loc[idx, 'min_pos_sentiment'] = np.min(scores_array_LABEL_1)
        df.loc[idx, 'std_pos_sentiment'] = np.std(scores_array_LABEL_1)

    end_time = time.time()
    duration = end_time - start_time

    print(f'Time taken for sentiment compute: {round((duration / 60.0), 2)} mins')

    return df


def add_sentiment_and_tfidf_to_df(original_df, tfidf_data, cryptobert_url, hugging_face_token):
    original_df = update_dataframe_with_sentiment(original_df, cryptobert_url, hugging_face_token)

    merged_df = tfidf_data.merge(original_df, on='time_published', how='inner')

    return merged_df


def processing_df_to_db(df, db_name, feature_engineered_df):
    '''
    Send Data to SQLite db under processed_table_name
    '''
    df = df.drop(['lemmatized_tokens', 'processed_text', 'combined_text', 'tokenized_text'], axis=1)
    conn = sqlite3.connect(db_name)
    df.to_sql(feature_engineered_df, conn, if_exists='append', index=False)
    conn.close()
    print('Check SQLite! df to db func')


def feature_eng_pipeline(db_name, processed_table_name, hf_ts_table_name, cryptobert_url, hugging_face_token):
    start_time = time.time()
    df = db_ingestion(db_name, processed_table_name)
    processed_text_df = preprocess_text(df)

    tfidf_df = compute_tfidf(processed_text_df)

    merged_df = add_sentiment_and_tfidf_to_df(processed_text_df, tfidf_df, cryptobert_url, hugging_face_token)
    processing_df_to_db(merged_df, db_name, hf_ts_table_name)

    end_time = time.time()
    duration = end_time - start_time

    print(f'Time taken for FE Pipeline: {duration} seconds')

    return 'Check SQLite? is it this one? fengpipe'


feature_eng_pipeline(db_name, processed_table_name, table_name, cryptobert_url, hugging_face_token)
