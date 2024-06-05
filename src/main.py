import pandas as pd
from src.data.data_preprocessing import CreateData
from src.models.creating_model import CreateModel

def upload_dataframe(stock_name, keyword_list, trends_timeframe):
    data = pd.read_excel(f'/Users/seanwhite/OneDrive - University of Greenwich/Documents/Group Project/{stock_name} Price History.xlsx')

    create_data = CreateData(data, keyword_list, trends_timeframe)
    updated_df = create_data.return_data()
    create_data.visualise_correlation()
    create_data.visualise_price()

    updated_df.to_excel(f'/Users/seanwhite/OneDrive - University of Greenwich/Documents/Group Project/group_project_code/data/stocks & trends/{stock_name} Stock & Trends.xlsx', index=False)


def xgbregressor(df_name, df, parameter_grid=None):
    create_model = CreateModel(df)
    create_model.split_data(0.8, 0.1, 0.1)

    if parameter_grid is not None:
        best_params = create_model.tune_parameters(parameter_grid)
    else:
        best_params = None

    create_model.train_model(best_params)
    create_model.retrain_with_validation(best_params)
    create_model.retrain_without_trends(best_params)
    metrics_df = create_model.get_metrics_dataframe()
    metrics_df.to_excel(f'/Users/seanwhite/OneDrive - University of Greenwich/Documents/Group Project/group_project_code/data/metrics tables/{df_name} Metrics.xlsx')


if __name__ == '__main__':
    df_name = 'Ocado'
    try:
        df = upload_dataframe(df_name, ['covid', 'lockdown', 'quarantine'], '2019-03-31 2024-03-27')
    except:
        df = f'/Users/seanwhite/OneDrive - University of Greenwich/Documents/Group Project/group_project_code/data/stocks & trends/{df_name} Stock & trends.xlsx'
    parameter_grid = {
                        }
    xgbregressor(df_name, df)
