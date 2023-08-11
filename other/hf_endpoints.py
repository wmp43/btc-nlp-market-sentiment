'''
- Can't quite get these models functional
- I Think it is something to do with GPUs and Cuda not being up to date/possible given mac
- Tried Using CPUs but it is too slow
'''

import json
import requests

def get_summary(text, trad_url, hugging_face_token):
    headers = {"Authorization": f"Bearer {hugging_face_token}", "Content-Type": "application/json"}
    payload = {
        "inputs": text, 
        "parameters": {
            "do_sample": False}
            }
    data = json.dumps(payload)
    response = requests.post(trad_url, headers=headers, data=data)
    response_json = json.loads(response.content.decode("utf-8"))

    summary = None

    if response.status_code == 200:
        summary = response_json[0]['summary_text']
    else: print(f'Code: {response.status_code}, Error: {response.text}')
    return summary

def get_sentiment(text, cryptobert_url, hugging_face_token):
    headers = {"Authorization": f"Bearer {hugging_face_token}", "Content-Type": "application/json"}
    payload = {"inputs": text}

    response = requests.post(cryptobert_url, headers=headers, data=json.dumps(payload))
    response_json = response.json()
    summary = 0
    if response.status_code == 200:
        sentiment_score = response_json[0]['score'] 
    else: print(f'Code: {response.status_code}, Error: {response.text}')
    return sentiment_score


def apply_summary_and_sentiment(df, trad_url, cryptobert_url, hugging_face_token):
    df['summary'] = df['content'].apply(lambda x: get_summary(x, trad_url, hugging_face_token))
    df['sentiment'] = df['summary'].apply(lambda x: get_sentiment(x, cryptobert_url, hugging_face_token))

    df = df.drop(columns=['titles_content'])
    print(df.dtypes)
    return df

