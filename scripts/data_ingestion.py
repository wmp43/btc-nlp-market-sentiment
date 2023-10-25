import requests
import pandas as pd
import time
import sqlite3
from newspaper3k import Article
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import numpy as np


def parse_float_date(date_str):
    print(date_str)
    return pd.to_datetime(date_str, format='%Y-%m-%d').strftime('%Y-%m-%d')


def parse_date(date_str):
    date_str = str(date_str)
    if len(date_str) == 15:
        date_str = date_str[:-7]
    elif len(date_str) == 13:
        date_str = date_str[:-5]

    return pd.to_datetime(date_str, format='%Y%m%d').strftime('%Y-%m-%d')


def get_latest_date_from_db(db_name, raw_table_name):
    conn = sqlite3.connect(db_name)
    query = f'SELECT MAX(time_published) as latest_date FROM {raw_table_name}'
    latest_date = pd.read_sql_query(query, conn)['latest_date'].iloc[0]
    conn.close()
    return latest_date


def format_date_for_api(date_str):
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    formatted_date = date_obj.strftime('%Y%m%dT%H%M')
    return formatted_date


news_data = pd.DataFrame(columns=['titles', 'url', 'time_published'])


def get_news_data(api_key, time_from):
    """
    - Generates Global dataframe
    - Extracts Relevant info: URL, Title, Date/Time
    - Adds new Info to Dataframe
    - Finds last date in that dataframe
    - Updates API call to only include after that date
    - Returns final dataframe with about 3600 rows as of June 2nd 2023
    """
    global news_data
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=COIN,CRYPTO:BTC&time_from={time_from}&sort=earliest&topics=blockchain&limit=1000&apikey={api_key}'
    r = requests.get(url)
    data = r.json()

    if 'feed' in data:
        feed = data['feed']

        urls = []
        time_published = []
        titles = []

        for entry in feed:
            entry_url = entry['url']
            entry_time_published = entry['time_published']
            entry_title = entry['title']

            urls.append(entry_url)
            time_published.append(entry_time_published)
            titles.append(entry_title)

        temp_df = pd.DataFrame({'titles': titles, 'url': urls, 'time_published': time_published})
        news_data = news_data.append(temp_df, ignore_index=True)
        next_start = news_data['time_published'].iloc[-1][:-6] + '0000'
        print(f'news data:{news_data}')

        if len(temp_df) < 10:
            return news_data
        else:
            print('waiting 17 seconds')
            time.sleep(17)
            return get_news_data(api_key, next_start)


def get_historical_btc_data(api_key):
    """
    API Calls for historical BTC DATA
    Adds to dataframe
    Seems like this API is down or has changed
    """
    btc_url = f'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol=BTC&market=USD&apikey={api_key}&outputsize=full'
    btc_price_data = pd.DataFrame(columns=['date', 'price'])
    try:
        response = requests.get(btc_url)
        data = response.json()
        btc_prices = data['Time Series (Digital Currency Daily)']

        for date, price in btc_prices.items():
            close_price = price['4a. close (USD)']
            btc_price_data = btc_price_data.append({'date': date, 'price': close_price}, ignore_index=True)

    except requests.exceptions.RequestException as e:
        print(f'Request Error: {e}')

    return btc_price_data


def merge_data(btc_data, text_data):
    """
    parses date for join on date.
    """
    print(type(btc_data['date'][0]))
    print(btc_data['date'][0])
    text_data['time_published'] = text_data['time_published'].apply(parse_date)
    btc_data['date'] = btc_data['date'].apply(parse_float_date)

    merged_data = pd.merge(text_data, btc_data, left_on='time_published', right_on='date', how='inner')
    merged_data.drop_duplicates(subset='titles', keep='first', inplace=True)
    merged_data.drop('date', axis=1, inplace=True)
    return merged_data


def extract_text(url):
    counter = 0
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except:
        print(f'Not availble @{url}')
        counter += 1
    print(f'{counter} URLS not Available')


def extract_content(df):
    with ThreadPoolExecutor(max_workers=10) as executor:
        df['content'] = list(executor.map(extract_text, df['url']))
    df.dropna(subset=['content'], inplace=True)
    return df


def df_to_db(merged_data, database_name, tablename):
    """
    Send Data to SQLite db under raw-btc-price-news
    """
    merged_data.rename(columns={'Unnamed: 0': 'Index'}, inplace=True)
    merged_data['price'] = np.round(merged_data['price'].astype(float), 2)

    conn = sqlite3.connect(database_name)
    merged_data.to_sql(tablename, conn, if_exists='append', index=False)
    conn.close()


# def ingestion_master(api, timefrom, database_name, tablename):
#     text_data = get_news_data(api, timefrom)
#     btc_data = get_historical_btc_data(api)
#     merged_df = merge_data(btc_data, text_data)
#     all_raw_data = extract_content(merged_df)
#     df_to_db(all_raw_data, database_name, tablename)
#     return print('check SQLite!')


def btc_fear_greed_idx(rapid_api):
    url = "https://fear-and-greed-index.p.rapidapi.com/v1/fgi"
    headers = {
        "X-RapidAPI-Key": rapid_api,
        "X-RapidAPI-Host": "fear-and-greed-index.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers)
    data = response.json()

    fgi_val = data['fgi']['now']['value']
    fgi_txt = data['fgi']['now']['valueText']

    return fgi_val, fgi_txt
# Summaries Should be done before ts data is set up.
# Model will preform better because they are coherent artilces
# Can set max len of the summaries to be (num articles)d/avg len(articles)d where d is a day
# Then Concat the summaries
