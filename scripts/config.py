api_key = 'S5UL3XS65JK02IZL'
db_name = 'text-btc-db'

time_from = '20220301T0000'
rapid_api = 'b93b91a249msh8e67e00c32eb7c2p1ac4bcjsn3adf37ba4012'
hugging_face_token = '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'
headers = {"Authorization": f"Bearer {hugging_face_token}"}
cryptobert_url = 'https://ot1scv5vphok4sfb.us-east-1.aws.endpoints.huggingface.cloud'

table0 = 'raw_btc_price_news'
table1 = 'time_series_data'
table2: str = "ts_fe"

clf_path = 'models/xgb_clf_1.0.pkl'
reg_path = 'models/xgb_reg_1.0.pkl'

db_path: str = "db-backup/text-btc-db"


bucket_name = 'btc-nlp'
s3_object_key = 's3://btc-nlp/text-btc-db'
local_file_name = 'ts-fe'
