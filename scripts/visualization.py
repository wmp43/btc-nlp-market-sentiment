import sqlite3
from scripts.config import db_name, table_name
import matplotlib.pyplot as plt
import pandas as pd


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

df = db_ingestion(db_name, table_name)

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Note the change here, pass the values of 'mean_neg_sentiment'
# scatter = ax.scatter(df['PCA1'], df['PCA2'], df['mean_pos_sentiment'], c=df['mean_neg_sentiment'].values, cmap='coolwarm')

# ax.set_xlabel('PCA1')
# ax.set_ylabel('PCA2')
# ax.set_zlabel('Mean Positive Sentiment')
# plt.title('3D Scatter Plot of Principal Components and Sentiment')

# # Add colorbar using the scatter plot as a mappable
# plt.colorbar(scatter, label='Mean Negative Sentiment')

# # Display the plot in Streamlit
# plt.show()

# import matplotlib.dates as mdates

# fig, ax1 = plt.subplots(figsize=(12, 8))


import matplotlib.dates as mdates
fig, ax1 = plt.subplots(figsize=(12, 8))

# Plotting price
df['time_published'] = pd.to_datetime(df['time_published'])
ax1.plot(df['time_published'], df['price'], color='b', label='Price')
ax1.set_xlabel('Year-Month')
ax1.set_ylabel('Price', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Creating a second axis to plot sentiment
ax2 = ax1.twinx()
ax2.plot(df['time_published'],(df['mean_pos_sentiment'] - df['mean_neg_sentiment']).rolling(30).mean(), color='r', label='Sentiment Change')
ax2.set_ylabel('Sentiment Change', color='r')
ax2.tick_params(axis='y', labelcolor='r')

locator = mdates.AutoDateLocator()
ax1.xaxis.set_major_locator(locator)

# Set the formatter to show the dates as YYYY-MM
formatter = mdates.DateFormatter('%Y-%m')
ax1.xaxis.set_major_formatter(formatter)

# Rotating the date labels
plt.xticks(rotation=45)

plt.title('Price vs Sentiment Change Over Time')
plt.show()







print(df['time_published'].dtype)









# # Box plot for a side-by-side comparison
# sentiment_df = df[['mean_neg_sentiment', 'max_neg_sentiment', 'min_neg_sentiment', 'std_neg_sentiment', 'mean_pos_sentiment', 'max_pos_sentiment', 'min_pos_sentiment', 'std_pos_sentiment']]
# sns.boxplot(data=sentiment_df)
# plt.title('Distribution of Negative and Positive Sentiments')
# plt.show()

# # Time series plot (if applicable)
# df.plot(x='time_published', y=['mean_neg_sentiment', 'mean_pos_sentiment'], title='Sentiment Over Time')
# plt.show()
