'''import pandas as pd
import requests
import schedule
import time
import datetime


# Twitter API credentials


df = pd.DataFrame(columns=['Tweet ID', 'Created At', 'Text', 'Author ID'])

def get_tweets(username, bearer_token, start_date=None):
    user_url = f'https://api.twitter.com/2/users/by/username/{username}'
    headers = {'Authorization': f'Bearer {bearer_token}'}

    # Get the user ID
    response = requests.get(user_url, headers=headers)
    user_id = response.json()['data']['id']

    # Construct the tweets endpoint URL with parameters
    tweets_url = f'https://api.twitter.com/2/users/{user_id}/tweets?tweet.fields=created_at&expansions=author_id&max_results=20'

    if start_date:
    # Convert start_date to ISO 8601 format
        start_date_iso = start_date.isoformat()
        tweets_url += f'&start_time={start_date_iso}'

# Send the request to retrieve the tweets
    response = requests.get(tweets_url, headers=headers)
    tweets_data = response.json()

    # Check if the 'data' key exists in the response
    if 'data' in tweets_data:
        # Extract the relevant information from the response
        tweets = []
        for tweet in tweets_data['data']:
            tweet_id = tweet['id']
            created_at = tweet['created_at']
            text = tweet['text']
            author_id = tweet['author_id']
            tweets.append({'Tweet ID': tweet_id, 'Created At': created_at, 'Text': text, 'Author ID': author_id})

        return tweets
    else:
        print(f"No tweets found for {username}")

    return []

def update_dataframe(username, bearer_token, start_date=None):
    global df  # Declare 'df' as a global variable

    # Get the tweets for the specified username and start date
    tweets = get_tweets(username, bearer_token, start_date)

    # Append new tweets to the existing DataFrame
    if tweets:
        df = df.append(tweets, ignore_index=True)

def run_script(start_date=None):
    usernames = ['VitalikButerin', 'Rewkang', 'WuBlockchain']

    for username in usernames:
        update_dataframe(username, bearer_token, start_date)

    global df  # Declare 'df' as a global variable
    print(df)

# Example usage:
start_date = datetime.datetime(2023, 5, 1)  # Specify your desired start date
run_script(start_date)

# Schedule the script to run every 5 minutes
#schedule.every(5).minutes.do(run_script)

# Keep the script running continuously
#while True:
 #   schedule.run_pending()
  #  time.sleep(1)
'''