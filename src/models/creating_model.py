import ns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
import seaborn as sns


class CreateModel:
    """
    This class is for applying the XGBRegressor to pre_processed stock price & google trends data.

    It takes the dataset as input and will split it into train, test, validate, then evaluate the performance of
    the model. The output returned will be R2, MSE, RMSE, MAE & MAPE metrics, as well as line graph plotting predicted
    against actual values using both test and val datasets for:
    - The model built with the full X_train data by calling train_model().
    - The model built with the full X_train data + full X_val data by calling retrain_with_validation().
    - The model built and trained with the same data as train_model(), however, with the Google Trends features removed
    by calling retrain_without_trends().

    Additionally, if wanting to test for the best hyperparameters for the model on the given dataset call
    tune_parameters() and add your parameter grid in the brackets. These can then be used each iteration of the model
    by adding best_params when calling the above model building methods.

    If wanting to save the metrics as a table in Excel call get_metrics_dataframe().
    """

    def __init__(self, df, df_name):
        # Creates and adds the aggregated 'Trends' column
        interaction = (df.iloc[:, 6] + df.iloc[:, 7] + df.iloc[:, 8]) / 3
        df.insert(len(df.columns) - 1, 'Trends', interaction)
        # Removes the now redundant separate trends data columns
        columns_to_drop = df.columns[6:9]
        df.drop(columns=columns_to_drop, inplace=True)
        self.df = df
        self.df['Exchange Date'] = pd.to_datetime(self.df['Exchange Date'])
        self.df_sorted = self.df.sort_values(by='Exchange Date')
        self.df_name = df_name

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
        self.X_train = train_data.iloc[:, 1:7]
        self.y_train = train_data.iloc[:, -1]
        self.X_test = test_data.iloc[:, 1:7]
        self.y_test = test_data.iloc[:, -1]
        self.X_val = val_data.iloc[:, 1:7]
        self.y_val = val_data.iloc[:, -1]

        # Creates indexed data for the dates to use in visualisations
        self.test_dates = test_data['Exchange Date']
        self.val_dates = val_data['Exchange Date']

        # Shows the size of each dataset
        print("Training data length:", len(train_data))
        print("Testing data length:", len(test_data))
        print("Validation data length:", len(val_data))

    def build_model(self, parameters=None, training_x=None, training_y=None, testing_x=None):
        """
        Used to apply the XGBR model.
        """
        if training_x is None:
            raise Exception("Please provide training data")

        if parameters is None:
            parameters = {'updater': 'coord_descent', 'n_estimators': 1100, 'learning_rate': 0.3, 'lambda': 0.0001,
                          'feature_selector': 'greedy', 'booster': 'gblinear', 'alpha': 0,
                          'objective': 'reg:squarederror'}
        self.model = XGBRegressor(**parameters)
        self.model.fit(training_x, training_y)
        predicted_y = self.model.predict(testing_x)
        return predicted_y

    def evaluate_model(self, y_true, y_pred, set_name="Test Set"):
        """
        Produces the evaluation metrics.
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)

        self.metrics[set_name] = {'R2': r2, 'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

        print(f'{set_name} - \nMean Squared Error: {mse}\nRoot Mean Squared Error: {rmse}\nR^2 Score: {r2}\nMean Absolute Error: {mae}\nMean Absolute Percentage Error: {mape}%')

    def line_plot(self, data_name, time_frame, actual, predicted):
        """
        Produces the line graphs.
        """
        plt.figure(figsize=(8, 6))
        plt.plot(time_frame, predicted, label='Predicted Values')
        plt.plot(time_frame, actual, label='Actual Values')

        plt.title(f'Actual vs Predicted Values {data_name}')
        plt.xlabel('Exchange Date')
        plt.ylabel('Next Day Close Price')
        plt.legend()
        plt.show()

    def scatter_plot(self, title, test_y, pred_y):
        """
        Produces the scatter plots.
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(test_y, pred_y, alpha=0.5)
        plt.title(f'Actual vs Predicted {title}')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'k--', lw=2)
        plt.show()

    def feature_importance(self, features_name):
        """
        Produces the feature importance bar graphs.
        """
        if self.model is None:
            print('Cannot show feature importance as model has not been built yet. Please call build_model() first.')
            return
        else:
            feature_names = self.df.drop(['Exchange Date', 'Next Day Close'], axis=1).columns
            importances = self.model.feature_importances_

            #print('\nFeature Importances:', importances)
            feature_importances = dict(zip(feature_names, importances))
            print('\nFeature Importances:')
            for feature, importance in feature_importances.items():
                print(f'{feature}: {importance}')
            if np.any(importances < 0):
                print('Warning: Some feature importances are negative')

            sorted_indices = np.argsort(importances)[::-1]
            plt.figure(figsize=(8, 6))
            plt.title(f'{features_name} Feature Importance', fontsize=16)
            plt.bar(range(len(importances)), importances[sorted_indices], align='center')
            plt.xticks(range(len(importances)), feature_names[sorted_indices], rotation=45)
            plt.tight_layout()
            plt.show()

    def tune_parameters(self, params):
        """
        Finds the best parameters options based on R2, MSE, MAE, and MAPE metrics. Currently set to use for random search
        but this can be adjusted for grid search.
        """
        scorers = {
            'R2': make_scorer(r2_score),
            'MSE': make_scorer(mean_squared_error, greater_is_better=False),
            'MAE': make_scorer(mean_absolute_error, greater_is_better=False),
            'MAPE': make_scorer(mean_absolute_percentage_error, greater_is_better=False)
        }

        grid_search = RandomizedSearchCV(estimator=XGBRegressor(), param_distributions=params, scoring=scorers, cv=5,
                                         n_jobs=-1, verbose=1, refit='R2', error_score='raise', n_iter=150)
        grid_search.fit(self.X_train, self.y_train)

        print("Best Parameters found: ", grid_search.best_params_)
        print("Best CV Score: ", -grid_search.best_score_)

        self.best_params_ = grid_search.best_params_
        return self.best_params_

    def train_model(self, params=None):
        """
        Call to assess model performance for test and validation on train dataset.
        """
        self.y_pred = self.build_model(params, self.X_train, self.y_train, self.X_test)
        self.evaluate_model(self.y_test, self.y_pred, '\nTest Set')

        self.y_val_pred = self.build_model(params, self.X_train, self.y_train, self.X_val)
        self.evaluate_model(self.y_val, self.y_val_pred, '\nValidation Set')

        self.line_plot(f'{self.df_name} Test Data', self.test_dates, self.y_test, self.y_pred)
        self.line_plot(f'{self.df_name} Validation Data', self.val_dates, self.y_val, self.y_val_pred)
        self.scatter_plot(f'{self.df_name} Test Data', self.y_test, self.y_pred)
        self.scatter_plot(f'{self.df_name} Validation Data', self.y_val, self.y_val_pred)

        self.feature_importance(f'{self.df_name} X Train Set')

    def retrain_with_validation(self, params=None):
        """
        Call to assess model performance for test data on train with validation dataset.
        """
        X_combined = pd.concat([self.X_train, self.X_val])
        y_combined = pd.concat([self.y_train, self.y_val])

        self.y_pred = self.build_model(params, X_combined, y_combined, self.X_test)
        self.evaluate_model(self.y_test, self.y_pred, "\nTest Set After Retraining with Validation")

        self.line_plot(f'{self.df_name} Test Data After Retraining with Validation', self.test_dates, self.y_test, self.y_pred)
        self.scatter_plot(f'{self.df_name} Test Data After Retraining with Validation', self.y_test, self.y_pred)

        self.feature_importance(f'{self.df_name} X Train with Validation Set')

    def retrain_without_trends(self, params=None):
        """
        Call to assess model performance for test and validation on train (without trends) dataset.
        """
        columns_to_drop = [5]
        new_train = self.X_train.drop(self.X_train.columns[columns_to_drop], axis=1)
        new_test = self.X_test.drop(self.X_test.columns[columns_to_drop], axis=1)
        new_val = self.X_val.drop(self.X_val.columns[columns_to_drop], axis=1)

        self.y_pred = self.build_model(params, new_train, self.y_train, new_test)
        self.evaluate_model(self.y_test, self.y_pred, '\nTest Set Without Trends Data')

        self.y_val_pred = self.build_model(params, new_train, self.y_train, new_val)
        self.evaluate_model(self.y_val, self.y_val_pred, '\nValidation Set Without Trends Data')

        self.line_plot(f'{self.df_name} Test Set Without Trends', self.test_dates, self.y_test, self.y_pred)
        self.line_plot(f'{self.df_name} Validation Data Without Trends', self.val_dates, self.y_val, self.y_val_pred)
        self.scatter_plot(f'{self.df_name} Test Set Without Trends', self.y_test, self.y_pred)
        self.scatter_plot(f'{self.df_name} Validation Data Without Trends', self.y_val, self.y_val_pred)

        self.feature_importance(f'{self.df_name} X Train Set Without Trends')

    def get_metrics_dataframe(self):
        """
        Produces metrics table in Excel format. Adjust file path as needed.
        """
        metrics_table = pd.DataFrame(self.metrics).transpose()
        metrics_table.to_excel(f'/Users/seanwhite/OneDrive - University of Greenwich/Documents/Group Project/group_project_code/data/metrics tables/{self.df_name} Metrics.xlsx')
        return metrics_table
