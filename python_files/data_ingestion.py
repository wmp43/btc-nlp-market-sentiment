import requests
import pandas as pd
import time
from config import api_key, time_from, brave_binary, driver_path, glassnode_api_key, rapid_api
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from bs4 import BeautifulSoup
import re

news_data = pd.DataFrame(columns=['titles', 'url', 'time_published'])

def get_news_data(api_key, time_from):
    '''
    - Generates Global datafram
    - Extracts Relevant info: URL, Title, Date/Time
    - Adds new Info to Dataframe
    - Finds last date in that dataframe
    - Updates API call to only inlcude after that date
    - Returns final dataframe with about 3600 rows as of June 2nd 2023
    '''
    global news_data
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=COIN,CRYPTO:BTC&time_from={time_from}&sort=earliest&topics=blockchain&apikey={api_key}&limit=200'
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
        news_data = news_data.append(temp_df, ignore_index = True)
        next_start = news_data['time_published'].iloc[-1][:-6] + '0000'

        if len(temp_df)<10:
            return news_data
        else:
            time.sleep(17)
            return get_news_data(api_key, next_start)
            

# def get_historical_btc_data(api_key, output_size):
#     '''
#     API Calls for historical BTC DATA
#     Adds to dataframe
#     Seems like this API is down or has changed
#     '''
#     global news_data

#     btc_url = f'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol=BTC&market=USD&apikey={api_key}&outputsize={output_size}'
#     btc_price_data = pd.DataFrame(columns=['Date', 'Close Price'])
#     try:
#         response = requests.get(btc_url)
#         data = response.json()
#         print(data)
#         btc_prices = data['Time Series (Digital Currency Daily)']
        
#         for date, price in btc_prices.items():
#             close_price = price['4a. close (USD)']
#             btc_price_data = btc_price_data.append({'Date': date, 'Close Price': close_price}, ignore_index=True)
    
#     except requests.exceptions.RequestException as e:
#         print(f'Request Error: {e}')
    
#     btc_price_data.to_csv('data/btc_data', index = False)
#     return btc_price_data

def get_glassnode_data(glassnode_api_key):
    urls = ['https://api.glassnode.com/v1/metrics/addresses/active_count',
    'https://api.glassnode.com/v1/metrics/market/price_usd_close',
    'https://api.glassnode.com/v1/metrics/addresses/sending_to_exchanges_count',
    'https://api.glassnode.com/v1/metrics/addresses/receiving_from_exchanges_count']
    data = []
    for url in urls:
        label = url.split('/')[-1]
        res = requests.get(url, params = {'a':'BTC','api_key': glassnode_api_key})
        df = pd.read_json(res.text, convert_dates=['t'])
        df.set_index('t', inplace = True)
        df.rename(columns = {'v':label}, inplace = True)
        data.append(df)
    
    df = pd.concat(data, axis=1)
    return df


# def fetch_all_data(api_key, time_from, output_size):
#     get_news_data(api_key, time_from)
#     btc_data = get_historical_btc_data(api_key, output_size)
#     news_data.to_csv('data/news_data.csv', index=False)
#     btc_data.to_csv('data/btc_data', index = False)
#     return news_data, btc_data

news_data = pd.read_csv('news_data.csv')

chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--disable-javascript')

chrome_options.binary_location = brave_binary

service = Service(driver_path)
driver = webdriver.Chrome(service=service, options=chrome_options)

def scrape_text(url, index, retry_count=0):
    '''
    - Scrapes URLs and returns the main body of the article
    - Can use lambda function after to apply this function
    '''
    driver.get(url)
    driver.execute_script('window.onload=function(){};')

    print(f"Scraping URL at index {index}: {url}")

    start_time = time.time()

    try:
        # Wait for the body element to be present
        body_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '//body'))
        )
        html_content = body_element.get_attribute('innerHTML')
        soup = BeautifulSoup(html_content, 'html.parser')
        text = soup.get_text(separator=' ')
        text = re.sub(r'\n', '', text)  # Remove newlines
        text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespaces with a single whitespace

        # Extract relevant portion based on title
        title = news_data.loc[index, 'titles']
        if title in text:
            text = text.split(title, 1)[1].strip()
        
    except (NoSuchElementException, TimeoutException) as e:
        print(f"Encountered an error while scraping URL at index {index}: {url}")
        print(f"Error: {str(e)}")
        # Retry the scraping after a delay
        if retry_count < 5:  # Maximum number of retries
            retry_count += 1
            print(f"Retrying in 20 seconds... (Retry count: {retry_count})")
            time.sleep(20)
            return scrape_text(url, index, retry_count)
        else:
            print(f"Maximum number of retries reached. Skipping URL at index {index}: {url}")
            return ""

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time for URL at index {index}: {url} - {execution_time} seconds")

    return text
driver.quit()

chunk_size = 100
total_records = len(news_data)
output_filename = 'scraped_news_data.csv'

def concat_data(total_records, output_filename, chunk_size):
    for i in range(3756, total_records, chunk_size):
        start_index = i
        end_index = min(start_index + chunk_size, total_records)
        output_filename = f"data/scraped_chunks/scraped_news_data_{start_index+1}-{end_index}.csv"  # Dynamic output filename
        chunk = news_data.iloc[start_index:end_index, :]
        chunk['text'] = chunk.apply(lambda x: scrape_text(x['url'], x.name), axis=1)
        chunk.to_csv(output_filename, index=False)
        print(f"Scraped data saved to {output_filename} - Rows: {start_index+1} to {end_index} out of {total_records}")
        return 
  

def btc_fear_greed_idx(rapid_api):
    url = "https://fear-and-greed-index.p.rapidapi.com/v1/fgi"
    headers = {
        "X-RapidAPI-Key": "SIGN-UP-FOR-KEY",
        "X-RapidAPI-Host": "fear-and-greed-index.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers)
    data = response.json()

    # Extract the fear and greed index value
    fgi_now = data['fgi']['now']['value']

    return fgi_now



def fetch_all_data(api_key, time_from, output_size, glassnode_api_key):
    get_news_data(api_key, time_from)
    glassnode_data = get_glassnode_data(glassnode_api_key)

    total_records = len(news_data)
    concat_data(total_records, output_filename, chunk_size)
    fear_greed_index = btc_fear_greed_idx()

    return news_data, glassnode_data, fear_greed_index
