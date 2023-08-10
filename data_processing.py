import sqlite3
import pandas as pd
from config import db_name, raw_table_name, processed_table_name


def db_ingestion(db_name, raw_table_name):
    """
    Get Data From DB to process for Model
    """
    conn = sqlite3.connect(db_name)
    print('opened sqlite connection')

    query = f'SELECT * FROM {raw_table_name}'

    df = pd.read_sql_query(query, conn)
    print(df.dtypes)

    conn.close()
    return df


def join_data_one_day_per_row(df):
    """
    transforms df to concat titles and text
    one day per row --> time series set up
    """
    df['time_published'] = pd.to_datetime(df['time_published'])
    df['content'] = df['content'].astype(str)
    df['titles'] = df['titles'].astype(str)

    timeseries_df = df.groupby('time_published').agg(
        {'titles': ' '.join,
         'content': ' '.join,
         'price': 'mean'}).reset_index()

    # print(timeseries_df.dtypes)
    # print(timeseries_df['time_published'])

    print(f'time series dtypes: {timeseries_df.dtypes}')
    # timeseries_df = timeseries_df.merge(price_df, on='time_published', how='left')
    # timeseries_df = timeseries_df.drop_duplicates(subset=['time_published'])
    return timeseries_df


def lagging_features(df):
    """
    create lagging price, lagging rolling average,
    lagging volatility, rolling average for volume
    """
    df['day_of_week'] = df['time_published'].dt.dayofweek
    df['lagged_price'] = df['price'].shift(1)
    df['returns'] = df['price'].pct_change()

    df['3d_rolling_volatility'] = df['returns'].rolling(window=3).std()
    df['7d_rolling_volatility'] = df['returns'].rolling(window=7).std()

    df['future_returns'] = df['returns'].shift(-1)
    print(df.dtypes)
    return df


def processing_df_to_db(df, db_name, hf_ts_table_name):
    """
    Send Data to SQLite db under raw-btc-price-news
    """
    conn = sqlite3.connect(db_name)
    df.to_sql(hf_ts_table_name, conn, if_exists='append', index=True)
    conn.close()
    print('Check SQLite!')


def process_data_and_save_to_db(db_name, raw_table_name, processed_table_name):
    df = db_ingestion(db_name, raw_table_name)
    df = join_data_one_day_per_row(df)
    df = lagging_features(df)

    processing_df_to_db(df, db_name, processed_table_name)


process_data_and_save_to_db(db_name, raw_table_name, processed_table_name)
