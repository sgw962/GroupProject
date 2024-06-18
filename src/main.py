import pandas as pd
from src.data.data_preprocessing import CreateData
from src.models.creating_model import CreateModel

def upload_dataframe(stock_name, keyword_list, timescale, geo):
    """
    Calls CreateDate class. Adjust file path as needed.
    """
    data = pd.read_excel(f'/Users/seanwhite/OneDrive - University of Greenwich/Documents/Group Project/group_project_code/data/price history/{stock_name} Price History.xlsx')
    if geo == 0:
        place = ''
    elif geo == 1:
        place = 'GB'
    else:
        raise ValueError('Invalid geo must be 0 for international or 1 for Britain')

    create_data = CreateData(data, stock_name, keyword_list, timescale, place)
    updated_df = create_data.return_data()
    create_data.visualise_correlation()
    create_data.visualise_price()
    return updated_df

def xgbregressor(stock_name, data, parameter_grid=None):
    """
    Calls CreateModel class.
    """
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
    """
    Main function: The inputs used for this project are:
    - Ocado '2019-03-31 2024-03-27' ['covid', 'quarantine', 'lockdown'] 1
    - AstraZeneca '2019-03-31 2024-03-27' ['tagrisso', 'pharma', 'lynparza'] 0
    - Tesla '2019-05-15 2024-05-14' ['ev', 'elon musk', 'tesla'] 0
    - Diageo '2019-05-16 2024-05-14' ['alcohol', 'bars', 'covid'] 0
    - EasyJet '2019-04-02 2024-03-28' ['flights', 'abroad', 'holiday'] 1
    """
    stock_data_name = 'Ocado'
    trends_keywords = ['covid', 'quarantine', 'lockdown']
    trends_timeframe = '2019-03-31 2024-03-27'
    geo = 1

    try:
        df = upload_dataframe(stock_data_name, trends_keywords, trends_timeframe, geo)
    except Exception as e:
        print(f'Error running data building class: {e}')
        try:
            # If needed adjust file path for use on your own computer
            df = pd.read_excel(f'/Users/seanwhite/OneDrive - University of Greenwich/Documents/Group Project/group_project_code/data/stocks & trends/{stock_data_name} Stock & trends.xlsx')
        except Exception as e:
            print(f'Error loaded pre-processed data: {e}')
    parameter_grid = {'updater': 'coord_descent', 'n_estimators': 1100, 'learning_rate': 0.3, 'lambda': 0.0001,
                    'feature_selector': 'greedy', 'booster': 'gblinear', 'alpha': 0, 'objective': 'reg:squarederror'}
    try:
        xgbregressor(stock_data_name, df)
    except Exception as e:
        print(f'Error Running Model Class: {e}')
