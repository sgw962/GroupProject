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
#from src.data.data_preprocessing import updated_df
from src.data.data_preprocessing import visualise_correlation
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


class CreateModel:
    def __init__(self, df):
        self.df = df
        X = self.df.iloc[:, 1:8]
        y = self.df.iloc[:, -1]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = None
        self.y_pred = None

    def build_model(self):
        self.model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

        # Fit the model
        self.model.fit(self.X_train, self.y_train)

        # Predict the Close prices
        self.y_pred = self.model.predict(self.X_test)

        # Evaluate the model
        mse = mean_squared_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, self.y_pred)

        print(f'Mean Squared Error: {mse}\n Root Mean Squared Error: {rmse}\n R^2 Score: {r2}')
        return

    def plot_model(self):
        if self.y_pred is None:
            print('Cannot plot model as model has not been built yet. Please call build_model() first.')
            return
        else:
            plt.figure(figsize=(8, 6))
            plt.scatter(self.y_test, self.y_pred, alpha=0.5)
            plt.title('Actual vs. Predicted Values')
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'k--', lw=2)  # Line showing perfect predictions
            plt.grid(True)
            plt.show()

    def feature_importance(self):
        if self.model is None:
            print('Cannot show feature importance as model has not been built yet. Please call build_model() first.')
            return
        else:
            feature_names = self.df.drop(['Exchange Date', 'Next Day Close'], axis=1).columns
            importances = self.model.feature_importances_
            sorted_indices = np.argsort(importances)[::-1]
            plt.figure(figsize=(10, 7))
            plt.title('GB Feature Importance', fontsize=16)
            plt.bar(range(len(importances)), importances[sorted_indices], align='center')
            plt.xticks(range(len(importances)), feature_names[sorted_indices], rotation=45)
            plt.tight_layout()
            plt.show()


# data = pd.read_excel('/Users/seanwhite/OneDrive - University of Greenwich/Documents/Group Project/EasyJet Price History.xlsx')

# create_data = CreateData(data, ['easy jet', 'cheap flights', 'holidays to europe'], '2019-03-31 2024-03-27', 'GB')
# updated_df = create_data.return_data()

# print(updated_df)
#visualise_correlation(updated_df)
data = pd.read_excel('/Users/seanwhite/OneDrive - University of Greenwich/Documents/Group Project/group_project_code/src/data/Ocado Stock & trends.xlsx')
create_model = CreateModel(data)
#create_model.build_model()
create_model.plot_model()
create_model.feature_importance()
