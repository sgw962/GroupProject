import pandas as pd
from src.data.data_preprocessing import CreateData
from src.models.creating_model import CreateModel

def upload_dataframe(stock_name, keyword_list, timescale):
    data = pd.read_excel(f'/Users/seanwhite/OneDrive - University of Greenwich/Documents/Group Project/{stock_name} Price History.xlsx')

    create_data = CreateData(data, stock_name, keyword_list, timescale)
    updated_df = create_data.return_data()
    create_data.visualise_correlation()
    create_data.visualise_price()
    return updated_df

def xgbregressor(stock_name, data, parameter_grid=None):
    create_model = CreateModel(data, stock_name)
    create_model.split_data(0.8, 0.1, 0.1)

    best_params = None
    if parameter_grid:
        best_params = create_model.tune_parameters(parameter_grid)

    create_model.train_model(best_params)
    create_model.retrain_with_validation(best_params)
    create_model.retrain_without_trends(best_params)
    create_model.get_metrics_dataframe()


if __name__ == '__main__':
    stock_data_name = 'EasyJet'
    trends_keywords = ['covid', 'lockdown', 'quarantine']
    trends_timeframe = '2019-03-31 2024-03-27'
    try:
        df = upload_dataframe(stock_data_name, trends_keywords, trends_timeframe)
    except Exception as e:
        print(f'Error uploading: {e}')
        df = f'/Users/seanwhite/OneDrive - University of Greenwich/Documents/Group Project/group_project_code/data/stocks & trends/{stock_data_name} Stock & trends.xlsx'
    parameter_grid = {'updater': 'coord_descent', 'n_estimators': 1200, 'learning_rate': 0.3, 'lambda': 0.0001,
                          'feature_selector': 'greedy', 'booster': 'gblinear', 'alpha': 0,
                          'objective': 'reg:squarederror'
                        }
    try:
        xgbregressor(stock_data_name, df)
    except Exception as e:
        print(f'Error: {e}')
