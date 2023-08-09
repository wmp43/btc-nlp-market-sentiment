
def join_data_one_day_per_row(news_data, glassnode_data):
    article_df = news_data
    price_df = glassnode_data
    
    if article_df['date'].dtype != price_df['t'].dtype:
        article_df['date'] = pd.to_datetime(article_df['date'])
        price_df['t'] = pd.to_datetime(price_df['t'])

    article_df['text'] = article_df['text'].astype(str)
    article_df['titles'] = article_df['titles'].astype(str)

    articles_grouped = article_df.groupby('date').agg({'titles': ' '.join, 'text': ' '.join}).reset_index()

    merged_df = pd.merge(articles_grouped, price_df, left_on='date', right_on='t', how='left')
    merged_df['text'] = merged_df['text'].fillna(merged_df['titles'])
    merged_df.drop('t', axis=1, inplace=True)  # Drop the 't' column from merged_df

    #merged_df.to_csv('final_data/joined_day_per_observation.csv', index=False)
    return merged_df


def processing_pipeline(chunk_size, all_data, glassnode_data):
    processed_news_data = combined_function(chunk_size, all_data)
    # Group articles by date and merge with glassnode_data
    final_df = join_data_one_day_per_row(processed_news_data, glassnode_data)

    return final_df







# def join_data_one_article_per_row():
#     article_df = pd.read_csv('data/concatenated_news_data.csv')
#     price_df = pd.read_csv('data/btc_data.csv')
#     if article_df['date'].dtype != price_df['t'].dtype:
#         article_df['date'] = pd.to_datetime(article_df['date'])
#         price_df['t'] = pd.to_datetime(price_df['t'])

#     merged_df = pd.merge(article_df, price_df, left_on='date', right_on='t', how='left')
#     merged_df.drop(['t'], axis = 1)
#     merged_df.to_csv('final_data/joined_article_per_observation.csv', index = False)
#     return merged_df

#join_data_one_article_per_row()

#Tried to find the most common 3 word strings at the start and end, but the result was simply not good enough
#May come back to this later on; however, I think creating manual mappings will be more time consuming but yield a better result
# Additionally, I think that URLS scraped from alpha Vantage are from 20 different domains that I would have already created start and stop points.

# def preprocess_text(text):
#     processed_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
#     processed_text = processed_text.lower()
#     return processed_text

# def find_most_common_ngram(ngrams):
#     counter = Counter(ngrams)
#     most_common = counter.most_common(1)
#     if most_common:
#         return most_common[0][0]
#     else:
#         return ''

# def find_common_start_stop_points(file_path, n_gram=3):
#     # Load the data from the file
#     df = pd.read_csv(file_path)

#     # Convert the 'text' column to string
#     df['text'] = df['text'].astype(str)

#     # Extract the domain names from the URLs
#     df['domain'] = df['url'].apply(lambda url: urlparse(url).netloc)

#     # Exclude Forbes and investors.com domains
#     df = df[~df['domain'].isin(['www.forbes.com', 'www.investors.com'])]

#     # Dictionary to store start and stop points for each domain
#     domain_start_stop = {}

#     for domain in df['domain'].unique():
#         domain_text = df[df['domain'] == domain]['text'].tolist()

#         n_gram_start_points = []
#         n_gram_stop_points = []
#         dates = []

#         for text, date in zip(domain_text, df[df['domain'] == domain]['date']):
#             # Preprocess the text by removing special characters and converting to lowercase
#             processed_text = preprocess_text(text)

#             # Tokenize the processed text into words
#             words = nltk.word_tokenize(processed_text)

#             if len(words) >= n_gram:
#                 n_grams = list(nltk.ngrams(words, n_gram))
#                 n_gram_start_points.extend(n_grams[:10])  # Extract the first 10 n-grams as potential start points
#                 n_gram_stop_points.extend(n_grams[-10:])  # Extract the last 10 n-grams as potential stop points
#                 dates.extend([date] * len(n_grams))  # Store the date for each n-gram

#         common_start = find_most_common_ngram(n_gram_start_points)
#         common_stop = find_most_common_ngram(n_gram_stop_points)

#         domain_start_stop[domain] = (common_start, common_stop, dates)

#     # Create a new DataFrame with the filtered text and date
#     filtered_data = pd.DataFrame(columns=['domain', 'title', 'url', 'filtered_text', 'date'])
#     for domain, (start, stop, dates) in domain_start_stop.items():
#         domain_data = df[df['domain'] == domain]
#         for start, stop, date in zip(start, stop, dates):
#             filtered_text = domain_data[(domain_data['text'].str.contains(start)) & (domain_data['text'].str.contains(stop))]['text'].tolist()
#             filtered_data = filtered_data.append({
#                 'domain': domain,
#                 'title': domain_data['titles'].iloc[0],
#                 'url': domain_data['url'].iloc[0],
#                 'filtered_text': ' '.join(filtered_text),
#                 'date': date
#             }, ignore_index=True)
    
#     filtered_data = filtered_data.drop_duplicates(subset=['url'], keep='first')

#     return filtered_data.to_csv('final_filtered.csv')



# # Example usage
# start_stop_points = find_common_start_stop_points('trial.csv', n_gram=3)

# print(start_stop_points.head())







