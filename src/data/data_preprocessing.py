import ns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import openpyxl
import seaborn as sns
import numpy as np
from pytrends.exceptions import TooManyRequestsError
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from pytrends.request import TrendReq
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
import requests
import time


class CreateData:
    def __init__(self, df, keywords, timeframe, geo):
        self.df = df
        self.keywords = keywords
        self.timeframe = timeframe
        self.geo = geo
        self.trends = None

    def format_data(self):
        self.df = self.df.dropna()
        self.df['Next Day Close'] = self.df['Close'].shift(-1)
        self.df = self.df.iloc[:-1]

        numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        numeric_cols.remove('Next Day Close')
        scaler = MinMaxScaler().fit(self.df[numeric_cols])
        df_scaled = scaler.transform(self.df[numeric_cols])
        self.df[numeric_cols] = df_scaled

        return self.df

    def get_trends(self):
        self.keywords = [i.title() for i in self.keywords]

        vader_lexicon_path = '/Users/seanwhite/OneDrive - University of Greenwich/Documents/Group Project/vader_lexicon.txt'
        sia = SentimentIntensityAnalyzer(
            lexicon_file=vader_lexicon_path
        )

        pytrends = TrendReq(hl='en-Uk', tz=60)

        retries = 5
        delay = 10

        for attempt in range(retries):
            try:
                pytrends.build_payload(self.keywords, timeframe=self.timeframe, geo=self.geo)
                self.trends = pytrends.interest_over_time()
                pd.DataFrame(self.trends)
                numeric_cols = self.trends.select_dtypes(include=np.number).columns.tolist()
                scaler = MinMaxScaler().fit(self.trends[numeric_cols])
                trends_scaled = scaler.transform(self.trends[numeric_cols])
                self.trends[numeric_cols] = trends_scaled
                return self.trends
            except TooManyRequestsError as e:
                print(f"Attempt {attempt + 1}/{retries}: Rate limit exceeded. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
            except requests.exceptions.RequestException as e:
                print(f"Request exception: {e}")
                raise e
        raise Exception("Failed to retrieve trends data after multiple attempts")

    def merge_datasets(self, stock_data):
        """
        Required input (param) and output (return) are
            :param stock_data: Stock price data from EIKON
            :param trends: Weekly trends data acquired with PyTrends
            :return: A merged dataset which has the stock price data along with trends search data and keeps the 'New Close Price' as the end column
        """
        # Resetting index for trends data so the date column is no longer used as the index and can instead be used to merge the datasets
        new_trends = self.trends.reset_index()

        # Ensuring both date columns are in datetime format
        stock_data['Exchange Date'] = pd.to_datetime(stock_data['Exchange Date'])
        new_trends['date'] = pd.to_datetime(new_trends['date'])

        # Finding the nearest dates and merging the 'AstraZeneca' column based on these
        stock_data['nearest_date'] = stock_data['Exchange Date'].apply(
            lambda x: new_trends['date'].iloc[(new_trends['date'] - x).abs().argsort()[0]])
        merged_nearest_df = pd.merge(stock_data, new_trends, left_on='nearest_date', right_on='date', how='left')

        # Removing redundant columns
        merged_nearest_df.drop(['nearest_date', 'date', 'isPartial', 'Net', '%Chg', 'Volume', 'Turnover - GBP'], axis=1, inplace=True)

        # Adjusting the columns order to have 'Next Day Close' on the end
        columns = merged_nearest_df.columns.tolist()
        columns.remove('Next Day Close')
        columns.append('Next Day Close')
        merged_nearest_df = merged_nearest_df[columns]

        return merged_nearest_df

    def return_data(self):
        stocks = self.format_data()
        self.get_trends()
        return self.merge_datasets(stocks)


def visualise_correlation(df):
    corr = df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()


weekly_data = pd.read_excel('/Users/seanwhite/OneDrive - University of Greenwich/Documents/Group Project/Ocado Price History.xlsx')
#daily_data = pd.read_excel('/Users/seanwhite/OneDrive - University of Greenwich/Documents/Group Project/Daily Ocado Price History.xlsx')

create_data = CreateData(weekly_data, ['covid', 'quarantine', 'lockdown'], '2019-03-31 2024-03-27', 'GB')
#full_data = create_data.return_data()
#visualise_correlation(full_data)
updated_df = create_data.return_data()
#print(updated_df)


#Ocado & Astra '2019-03-31 2024-03-27'
#Tesla '2019-05-15 2024-05-14'

updated_df.to_excel('Ocado Stock & Trends2.xlsx', index=False)

visualise_correlation(updated_df)
