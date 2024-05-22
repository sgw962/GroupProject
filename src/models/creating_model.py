import ns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import openpyxl
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from pytrends.request import TrendReq
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
from src.data.data_preprocessing import CreateData
from src.data.data_preprocessing import updated_df
from src.data.data_preprocessing import visualise_correlation
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from datetime import datetime, timedelta


class CreateModel:
    def __init__(self, df):
        self.df = df
        self.df['Exchange Date'] = pd.to_datetime(self.df['Exchange Date'])
        self.df_sorted = self.df.sort_values(by='Exchange Date')

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_val = None
        self.y_val = None

        self.model = None
        self.y_pred = None
        self.y_val_pred = None

    def split_data(self, train_size=0.8, test_size=0.1, val_size=0.1):
        assert train_size + test_size + val_size == 1, "Proportions must sum to 1"

        # Calculate the number of samples for each set
        total_samples = len(self.df_sorted)
        train_end_idx = int(total_samples * train_size)
        test_end_idx = train_end_idx + int(total_samples * test_size)

        # Split the dataset into training, testing, and validation sets
        train_data = self.df_sorted.iloc[:train_end_idx]
        test_data = self.df_sorted.iloc[train_end_idx:test_end_idx]
        val_data = self.df_sorted.iloc[test_end_idx:]

        # Define features and target
        self.X_train = train_data.iloc[:, 1:9]
        self.y_train = train_data.iloc[:, -1]
        self.X_test = test_data.iloc[:, 1:9]
        self.y_test = test_data.iloc[:, -1]
        self.X_val = val_data.iloc[:, 1:9]
        self.y_val = val_data.iloc[:, -1]

        print("Training data length:", len(train_data))
        print("Testing data length:", len(test_data))
        print("Validation data length:", len(val_data))

    def build_model(self, params=None):
        if params is None:
            params = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'random_state': 42}
        self.model = GradientBoostingRegressor(**params)

        # Fit the model
        self.model.fit(self.X_train, self.y_train)

        # Predict the Close prices
        self.y_pred = self.model.predict(self.X_test)
        self.y_val_pred = self.model.predict(self.X_val)

        if np.any(self.y_test == 0):
            print("Warning: There are zero values in the true labels, which can cause high MAPE.")

        # Evaluate the model on the test set
        self.evaluate_model(self.y_test, self.y_pred, "Test Set")

        # Evaluate the model on the validation set
        self.evaluate_model(self.y_val, self.y_val_pred, "\nValidation Set")

    def evaluate_model(self, y_true, y_pred, set_name="Test Set"):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)

        print(
            f'{set_name} - \nMean Squared Error: {mse}\nRoot Mean Squared Error: {rmse}\nR^2 Score: {r2}\nMean Absolute Error: {mae}\nMean Absolute Percentage Error: {mape}%')

    def scatter_plot(self):
        if self.y_pred is None:
            print('Cannot plot model as model has not been built yet. Please call build_model() first.')
            return
        else:
            plt.figure(figsize=(8, 6))
            plt.scatter(self.y_test, self.y_pred, alpha=0.5)
            plt.title('Actual vs. Predicted Values')
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'k--',
                     lw=2)  # Line showing perfect predictions
            plt.grid(True)
            plt.show()

    def line_plot(self):
        plt.plot(range(len(self.y_pred)), self.y_pred, label='Predicted Values')
        plt.plot(range(len(self.y_test)), self.y_test, label='Actual Values')

        plt.title('Actual vs Predicted Values')
        plt.xlabel('Time')
        plt.ylabel('Next Day Close Price')
        plt.legend()
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

    def tune_parameters(self, param_grid):
        grid_search = GridSearchCV(estimator=GradientBoostingRegressor(), param_grid=param_grid,
                                   scoring='neg_mean_squared_error', cv=5, n_jobs=-1, verbose=2,
                                   return_train_score=True)
        grid_search.fit(self.X_train, self.y_train)

        print("Best Parameters found: ", grid_search.best_params_)
        print("Best CV Score: ", -grid_search.best_score_)

        self.best_params_ = grid_search.best_params_
        return self.best_params_

    def retrain_with_validation(self, params=None):
        if params is None:
            params = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'random_state': 42}
        self.model = GradientBoostingRegressor(**params)

        # Combine training and validation data
        X_combined = pd.concat([self.X_train, self.X_val])
        y_combined = pd.concat([self.y_train, self.y_val])

        # Fit the model
        self.model.fit(X_combined, y_combined)

        # Predict the Close prices
        self.y_pred = self.model.predict(self.X_test)

        # Evaluate the model on the test set
        self.evaluate_model(self.y_test, self.y_pred, "Test Set After Retraining with Validation")

        # Evaluate the model on the validation set
        self.y_val_pred = self.model.predict(self.X_val)
        self.evaluate_model(self.y_val, self.y_val_pred, "\nValidation Set After Retraining with Validation")


# data = pd.read_excel('/Users/seanwhite/OneDrive - University of Greenwich/Documents/Group Project/EasyJet Price History.xlsx')

# create_data = CreateData(data, ['easy jet', 'cheap flights', 'holidays to europe'], '2019-03-31 2024-03-27', 'GB')
# updated_df = create_data.return_data()

# print(updated_df)
#visualise_correlation(updated_df)
data = pd.read_excel('/Users/seanwhite/OneDrive - University of Greenwich/Documents/Group Project/group_project_code/src/data/Ocado Stock & trends.xlsx')

param_grid = {
    'loss': 'huber',
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 3,
    'random_state': 42,
    'subsample': [0.8, 1.0],
    #'criterion': ['friedman_mse', 'squared_error'],
    #'min_samples_split': [0.5, 2, 3.5],
    #'min_samples_leaf': [0.5, 1, 3],
    #'min_weight_fraction_leaf': [0.0, 0.25, 0.5],
    #'min_impurity_decrease': [0.0, 2, 5],
    #'init': [None, 'zero'],
    #'max_features': [None, 'sqrt', 'log2'],
    #'alpha': [0.25, 0.5, 0.75],
    #'verbose': [0, 1, 2],
    #'max_leaf_nodes': [None, 2, 10, 100],
    #'warm_start': [True, False],
    #'validation_fraction': [0, 0.5, 1],
    #'n_iter_no_change': [None, 1, 10, 100],
    #'tol': [0.001, 0.01, 0.1, 10],
    #'ccp_alpha': [0.001, 0.01, 0.1, 10]
}

params_list = {
 'loss': 'huber',
 'n_estimators': 100,
 'learning_rate': 0.1,
 'criterion': 'squared_error',
 'init': 'zero',
 'max_depth': 3,
 'min_impurity_decrease': 0.0,
 'subsample': 1.0,
 'max_features': None,
 'max_leaf_nodes': 10,
 'alpha': 0.5,
 'min_samples_leaf': 1,
 'min_samples_split': 2,
 'min_weight_fraction_leaf': 0.0,
 'verbose': 2,
 'warm_start': True,
 'ccp_alpha': 0.001,
 'n_iter_no_change': None,
 'tol': 0.1,
 'validation_fraction': 0.5
 }


def create_sample_data(row_num):
    num_rows = row_num

    start_date = datetime.strptime('2023-04-01', '%Y-%m-%d')
    date_range = [start_date + timedelta(days=i) for i in range(num_rows)]

    np.random.seed(0)

    data_params = {
        'Exchange Date': date_range,
        'Close': np.random.uniform(0.00, 1.00, num_rows),
        'Open': np.random.uniform(0.00, 1.00, num_rows),
        'Low': np.random.uniform(0.00, 1.00, num_rows),
        'High': np.random.uniform(0.00, 1.00, num_rows),
        'Flow': np.random.uniform(0.00, 1.00, num_rows),
        'Covid': np.random.choice([0, 100], num_rows),
        'Online Shop': np.random.choice([0, 100], num_rows),
        'Next Day Close': np.random.uniform(3.000, 29.000, num_rows)
    }

    sample_data = pd.DataFrame(data_params)
    return sample_data

sample_df = create_sample_data(300)



create_model = CreateModel(updated_df)

#best_params = create_model.tune_parameters(param_grid)
#print(best_params.key())

#for mean_train_score, mean_test_score, params in zip(best_params['mean_train_score'], best_params['mean_test_score'], best_params['params']):
#    print(f"Parameters: {params}")
#    print(f"Mean Training Score: {-mean_train_score}")
#    print(f"Mean Validation Score: {-mean_test_score}\n")
#print("Best Hyperparameters:", best_params)

create_model.build_model()
create_model.scatter_plot()
create_model.line_plot()
create_model.new_line_plot()
create_model.feature_importance()
