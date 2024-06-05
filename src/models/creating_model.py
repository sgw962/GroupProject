import ns
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
#from src.data.data_preprocessing import CreateData
#from src.data.data_preprocessing import updated_df
#from src.data.data_preprocessing import visualise_correlation
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from xgboost import XGBRegressor


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

        self.test_dates = None
        self.val_dates = None

        self.metrics = {}

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

        self.test_dates = test_data['Exchange Date']
        self.val_dates = val_data['Exchange Date']

        print("Training data length:", len(train_data))
        print("Testing data length:", len(test_data))
        print("Validation data length:", len(val_data))

    def evaluate_model(self, y_true, y_pred, set_name="Test Set"):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)

        self.metrics[set_name] = {
            'R2': r2,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }

        print(f'{set_name} - \nMean Squared Error: {mse}\nRoot Mean Squared Error: {rmse}\nR^2 Score: {r2}\nMean Absolute Error: {mae}\nMean Absolute Percentage Error: {mape}%')

    def build_model(self, params=None):
        if params is None:
            params = {'booster': 'gblinear', 'feature_selector': 'greedy', 'updater': 'coord_descent'}
        self.model = XGBRegressor(**params)

        # Fit the model
        self.model.fit(self.X_train, self.y_train)

        # Predict the Close prices
        self.y_pred = self.model.predict(self.X_test)
        self.y_val_pred = self.model.predict(self.X_val)

        if np.any(self.y_test == 0):
            print("Warning: There are zero values in the true labels, which can cause high MAPE.")

        # Evaluate the model on the test set
        self.evaluate_model(self.y_test, self.y_pred, '\nTest Set')

        # Evaluate the model on the validation set
        self.evaluate_model(self.y_val, self.y_val_pred, '\nValidation Set')

        self.line_plot('Test Data', self.test_dates, self.y_test, self.y_pred)
        self.line_plot('Validation Data', self.val_dates, self.y_val, self.y_val_pred)

    def get_metrics_dataframe(self):
        return pd.DataFrame(self.metrics).transpose()

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
            plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'k--', lw=2)  # Line showing perfect predictions
            plt.grid(True)
            plt.show()

    def line_plot(self, data_name, time_frame, actual, predicted):
        plt.plot(time_frame, predicted, label='Predicted Values')
        plt.plot(time_frame, actual, label='Actual Values')

        plt.title(f'{data_name} Actual vs Predicted Values')
        plt.xlabel('Exchange Date')
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

            print('Feature Importances:', importances)
            if np.any(importances < 0):
                print('Warning: Some feature importances are negative')

            sorted_indices = np.argsort(importances)[::-1]
            plt.figure(figsize=(10, 7))
            plt.title('GB Feature Importance', fontsize=16)
            plt.bar(range(len(importances)), importances[sorted_indices], align='center')
            plt.xticks(range(len(importances)), feature_names[sorted_indices], rotation=45)
            plt.tight_layout()
            plt.show()

    def tune_parameters(self, params):
        scorers = {
            'R2': make_scorer(r2_score),
            'MSE': make_scorer(mean_squared_error, greater_is_better=False),
            'MAE': make_scorer(mean_absolute_error, greater_is_better=False),
            'MAPE': make_scorer(mean_absolute_percentage_error, greater_is_better=False)
                }

        grid_search = GridSearchCV(estimator=XGBRegressor(), param_grid=params, scoring=scorers, cv=3, n_jobs=-1, verbose=1, refit='R2', error_score='raise')#, n_iter=300)
        grid_search.fit(self.X_train, self.y_train)

        print("Best Parameters found: ", grid_search.best_params_)
        print("Best CV Score: ", -grid_search.best_score_)

        self.best_params_ = grid_search.best_params_
        return self.best_params_

    def retrain_with_validation(self, params=None):
        if params is None:
            params = {'booster': 'gblinear', 'feature_selector': 'greedy', 'updater': 'coord_descent'}
        self.model = XGBRegressor(**params)

        # Combine training and validation data
        X_combined = pd.concat([self.X_train, self.X_val])
        y_combined = pd.concat([self.y_train, self.y_val])

        # Fit the model
        self.model.fit(X_combined, y_combined)

        # Predict the Close prices
        self.y_pred = self.model.predict(self.X_test)

        # Evaluate the model on the test set
        self.evaluate_model(self.y_test, self.y_pred, "\nTest Set After Retraining with Validation")

        # Evaluate the model on the validation set
        self.y_val_pred = self.model.predict(self.X_val)
        self.evaluate_model(self.y_val, self.y_val_pred, "\nValidation Set After Retraining with Validation")

        self.line_plot('Test Data After Retraining with Validation', self.test_dates, self.y_test, self.y_pred)
        self.line_plot('Validation Data After Retraining with Validation', self.val_dates, self.y_val, self.y_val_pred)

    def retrain_without_trends(self, params=None):
        if params is None:
            params = {'booster': 'gblinear', 'feature_selector': 'greedy', 'updater': 'coord_descent'}
        self.model = XGBRegressor(**params)

        columns_to_drop = [5, 6, 7]
        new_train = self.X_train.drop(self.X_train.columns[columns_to_drop], axis=1)
        new_test = self.X_test.drop(self.X_test.columns[columns_to_drop], axis=1)
        new_val = self.X_val.drop(self.X_val.columns[columns_to_drop], axis=1)

        self.model.fit(new_train, self.y_train)
        new_y_pred = self.model.predict(new_test)

        self.evaluate_model(self.y_test, new_y_pred, '\nTest Set Without Trends Data')

        new_y_val_pred = self.model.predict(new_val)
        self.evaluate_model(self.y_val, new_y_val_pred, '\nValidation Set Without Trends Data')

        self.line_plot('Test Set Without Trends Data', self.test_dates, self.y_test, new_y_pred)
        self.line_plot('Validation Data Without Trends Data', self.val_dates, self.y_val, new_y_val_pred)

    def visualise_test(self):
        test = self.y_test
        time_scale = self.test_dates
        plt.plot(time_scale, test)

        plt.title('Test Values Over Time')
        plt.xlabel('Exchange Date')
        plt.ylabel('Y Test')
        plt.show()


# data = pd.read_excel('/Users/seanwhite/OneDrive - University of Greenwich/Documents/Group Project/EasyJet Price History.xlsx')

# create_data = CreateData(data, ['easy jet', 'cheap flights', 'holidays to europe'], '2019-03-31 2024-03-27', 'GB')
# updated_df = create_data.return_data()

# print(updated_df)
#visualise_correlation(updated_df)
data = pd.read_excel('/Users/seanwhite/OneDrive - University of Greenwich/Documents/Group Project/group_project_code/data/stocks & trends/AstraZeneca Stock & trends.xlsx')

def visualise_price(price_df):
    price = price_df['Next Day Close']
    time_scale = price_df['Exchange Date']
    plt.plot(time_scale, price)

    plt.title('Closing Price Over Time')
    plt.xlabel('Exchange Date')
    plt.ylabel('Next Day Close Price')
    plt.show()

def visualise_correlation(df):
    corr_df = df.drop('Exchange Date', axis=1)
    corr = corr_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()

#visualise_price(data)

param_grid = {
    'booster': ['gblinear'],  # Booster type
    'n_estimators': [100, 200, 300, 500, 1000, 1100, 1200, 1400],  # Number of boosting rounds
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4],  # Step size shrinkage
    'lambda': [0, 0.000001, 0.0001, 0.01, 0.1, 1],  # L2 regularization term on weights
    'alpha': [0, 0.000001, 0.0001, 0.01, 0.1, 1],  # L1 regularization term on weights
    'feature_selector': ['thrifty', 'cyclic', 'shuffle', 'random', 'greedy'],  # Feature selector type
    'updater': ['coord_descent'],# 'shotgun'],  # Updater type for linear booster
    'objective': ['reg:squaredlogerror', 'reg:squarederror', ],  # Learning objective
}

params_list = {
    #'alpha': 0.1,
    'booster': 'gblinear',
    'feature_selector': 'thrifty',
    'lambda': 0.001,
    'learning_rate': 0.6,
    'n_estimators': 200,
    #'objective': 'reg:squaredlogerror',
    'updater': 'coord_descent'
    }

boosters = {
    'booster': ['gblinear', 'gbtree', 'dart']
}

#visualise_correlation(data)

create_model = CreateModel(data)
create_model.split_data(0.8, 0.1, 0.1)

#best_params = create_model.tune_parameters(boosters)

create_model.build_model(params_list)
create_model.scatter_plot()
create_model.feature_importance()
create_model.retrain_with_validation(params_list)
create_model.feature_importance()
create_model.retrain_without_trends(params_list)
create_model.feature_importance()
metrics_df = create_model.get_metrics_dataframe()

metrics_df.to_excel('/Users/seanwhite/OneDrive - University of Greenwich/Documents/Group Project/group_project_code/data/metrics tables/AstraZeneca Metrics.xlsx')
