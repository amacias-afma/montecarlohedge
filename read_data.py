import os
import pandas as pd

def read_data(data_path):
  """
  Reads and preprocesses Hashprice, BTC, and Electricity data from CSV files.

  Parameters:
  data_path (str): Path to the directory containing the CSV files.

  Returns:
  pd.DataFrame: A combined DataFrame containing the preprocessed data for all assets.
  """
  files_pm = {}

  files_pm['hashprice'] = {'file_name': 'bitcoin-hashprice-index-20251110.csv'}
  files_pm['hashprice']['col_date_name'] = 'timestamp'
  files_pm['hashprice']['col_price'] = 'usd_hashprice'

  files_pm['btc'] = {'file_name': 'BTC-USD.csv'}
  files_pm['btc']['col_date_name'] = 'Date'
  files_pm['btc']['col_price'] = 'Adj Close'

  files_pm['electricity'] = {'file_name': 'Electricity data ERCOT.csv'}
  files_pm['electricity']['col_date_name'] = 'Date'
  files_pm['electricity']['col_price'] = 'price'

  df_data = pd.DataFrame([])
  for key, pm in files_pm.items():
    file_name = pm['file_name']
    col_date_name = pm['col_date_name']
    file_path = os.path.join(data_path, file_name)
    df_data_aux = pd.read_csv(file_path)

    df_data_aux[col_date_name] = pd.to_datetime(df_data_aux[col_date_name])
    df_data_aux['date_aux'] = df_data_aux[col_date_name].dt.strftime('%Y-%m-%d')
    df_data_aux['date_aux'] = pd.to_datetime(df_data_aux['date_aux'], format='%Y-%m-%d')
    df_data_aux = df_data_aux.pivot_table(index='date_aux', values=pm['col_price'], aggfunc='mean')

    df_data_aux = df_data_aux[[pm['col_price']]]
    df_data_aux.rename(columns={pm['col_price']: key}, inplace=True)
    df_data_aux.index.name = 'date'
    df_data_aux.sort_index(inplace=True)

    df_data = pd.concat([df_data, df_data_aux], axis=1)
  df_data.dropna(inplace=True)
  return df_data
