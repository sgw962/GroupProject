import ns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import openpyxl
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from pytrends.request import TrendReq
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler


df = pd.read_excel('/Users/seanwhite/OneDrive - University of Greenwich/Documents/Group Project/Astrazeneca Price History.xlsx')


class CreateData:
    def __init__(self):
        pass

    def format_data(self, df):
        df = df.dropna()
        df['Next Day Close'] = df['Close'].shift(-1)
        df = df[:-1]

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        numeric_cols.remove('Next Day Close')
        scaler = MinMaxScaler().fit(df[numeric_cols])
        df_scaled = scaler.transform(df[numeric_cols])
        df[numeric_cols] = df_scaled

        return df

    def get_trends(self, keywords, timeframe, geo):
        vader_lexicon_path = '/Users/seanwhite/OneDrive - University of Greenwich/Documents/Group Project/vader_lexicon.txt'
        sia = SentimentIntensityAnalyzer(
            lexicon_file=vader_lexicon_path
        )

        pytrends = TrendReq(hl='en-Uk', tz=360)

        keywords = keywords
        pytrends.build_payload(keywords, timeframe=timeframe, geo=geo)

        trends_data = pytrends.interest_over_time()
        return pd.DataFrame(trends_data)

    def merge_datasets(self, stock_data, trends):
        """
        Required input (param) and output (return) are
            :param stock_data: Stock price data from EIKON
            :param trends: Weekly trends data acquired with PyTrends
            :return: A merged dataset which has the stock price data along with trends search data and keeps the 'New Close Price' as the end column
        """
        # Resetting index for trends data so the date column is no longer used as the index and can instead be used to merge the datasets
        new_trends = trends.reset_index()

        # Ensuring both date columns are in datetime format
        stock_data['Exchange Date'] = pd.to_datetime(stock_data['Exchange Date'])
        new_trends['date'] = pd.to_datetime(new_trends['date'])

        # Finding the nearest dates and merging the 'AstraZeneca' column based on these
        stock_data['nearest_date'] = stock_data['Exchange Date'].apply(
            lambda x: new_trends['date'].iloc[(new_trends['date'] - x).abs().argsort()[0]])
        merged_nearest_df = pd.merge(stock_data, new_trends, left_on='nearest_date', right_on='date', how='left')

        # Removing redundant columns
        merged_nearest_df.drop(['nearest_date', 'date', 'isPartial'], axis=1, inplace=True)

        # Adjusting the columns order to have 'Next Day Close' on the end
        columns = merged_nearest_df.columns.tolist()
        columns[-1], columns[-2] = columns[-2], columns[-1]
        merged_nearest_df = merged_nearest_df[columns]

        return merged_nearest_df


formatted_data = CreateData().format_data(df)
trends = CreateData().get_trends(['Astrazeneca', 'covid'], '2019-03-31 2024-03-27', 'GB')

updated_df = CreateData().merge_datasets(formatted_data, trends)
print(updated_df)
