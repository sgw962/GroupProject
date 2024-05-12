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
from src.data.data_preprocessing import CreateData
from src.data.data_preprocessing import updated_df
from src.data.data_preprocessing import visualise_correlation

class CreateModel():
    def __init__(self, model, df):
        self.model = model
        self.df = df

    def build_model(self):
        X = self.df.iloc[:, 1:12]
        y = self.df.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = self.model(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

        # Fit the model
        self.model.fit(X_train, y_train)

        # Predict the Close prices
        y_pred = self.model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print('Mean Squared Error:', mse, '\n'
            'Root Mean Squared Error:', rmse, '\n'
            'R^2 Score:', r2)

        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.title('Actual vs. Predicted Values')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Line showing perfect predictions
        plt.grid(True)
        plt.show()

    def feature_importance(self):
        feature_names = self.df.drop(['Exchange Date', 'Next Day Close'], axis=1).columns
        importances = self.model.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 7))
        plt.title('GB Feature Importance', fontsize=16)
        plt.bar(range(len(importances)), importances[sorted_indices], align='center')
        plt.xticks(range(len(importances)), feature_names[sorted_indices], rotation=45)
        plt.tight_layout()
        plt.show()


#data = pd.read_excel('/Users/seanwhite/OneDrive - University of Greenwich/Documents/Group Project/EasyJet Price History.xlsx')

#create_data = CreateData(data, ['easy jet', 'cheap flights', 'holidays to europe'], '2019-03-31 2024-03-27', 'GB')
#updated_df = create_data.return_data()

#print(updated_df)
visualise_correlation(updated_df)
create_model = CreateModel(GradientBoostingRegressor, updated_df)
create_model.build_model()
create_model.feature_importance()
