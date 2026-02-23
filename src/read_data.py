import os
import pandas as pd

import os
import pandas as pd
import yfinance as yf

def read_data(data_path, is_electricity_constant=False, use_online_data=True, start_date='2020-01-01', end_date=None):
    """
    Reads and preprocesses Hashprice, BTC, and Electricity data.
    Can fetch BTC and Energy data online from Yahoo Finance.

    Parameters:
    data_path (str): Path to the directory containing the CSV files.
    is_electricity_constant (bool): If True, ignores electricity data.
    use_online_data (bool): If True, fetches BTC and Electricity from Yahoo Finance.
    start_date (str): Start date for Yahoo Finance data fetch.
    end_date (str): End date for Yahoo Finance data fetch (None for today).

    Returns:
    pd.DataFrame: A combined DataFrame containing the preprocessed data.
    """
    
    if use_online_data:
        print("Fetching data from Yahoo Finance...")
        
        # 1. Fetch Bitcoin
        btc = yf.download('BTC-USD', start=start_date, end=end_date, progress=False)
        df_btc = btc[['Close']].copy()
        df_btc.columns = ['btc'] # Flatten multi-index if present
        df_btc.index = df_btc.index.normalize() # Remove timezone/time to match just dates
        df_btc.index.name = 'date'
        
        # 2. Fetch Natural Gas (Proxy for ERCOT Electricity Spot)
        if not is_electricity_constant:
            ng = yf.download('NG=F', start=start_date, end=end_date, progress=False)
            df_elec = ng[['Close']].copy()
            df_elec.columns = ['electricity']
            df_elec.index = df_elec.index.normalize()
            df_elec.index.name = 'date'
            
        # 3. Read Hashprice from CSV (Not available on Yahoo Finance)
        # Assuming the file name is standard as in your previous code
        hp_file = os.path.join(data_path, 'bitcoin-hashprice-index-20251110.csv')
        try:
            df_hp = pd.read_csv(hp_file)
            df_hp['timestamp'] = pd.to_datetime(df_hp['timestamp']).dt.normalize()
            df_hp = df_hp.groupby('timestamp')['usd_hashprice'].mean().to_frame()
            df_hp.columns = ['hashprice']
            df_hp.index.name = 'date'
        except FileNotFoundError:
            raise FileNotFoundError(f"Hashprice file not found at {hp_file}. This local file is required because Hashprice is not on Yahoo Finance.")

        # 4. Combine all datasets
        df_data = df_btc.join(df_hp, how='inner')
        if not is_electricity_constant:
            # Join electricity. Traditional markets close on weekends, so we use forward-fill (ffill)
            # to carry Friday's electricity price over the weekend to match Crypto's 24/7 trading.
            df_data = df_data.join(df_elec, how='left')
            df_data['electricity'] = df_data['electricity'].ffill()
            
        df_data.dropna(inplace=True)
        return df_data

    else:
        # --- ORIGINAL OFFLINE CSV LOGIC ---
        print("Loading data from local CSVs...")
        files_pm = {}

        files_pm['hashprice'] = {'file_name': 'bitcoin-hashprice-index-20251110.csv', 'col_date_name': 'timestamp', 'col_price': 'usd_hashprice'}
        files_pm['btc'] = {'file_name': 'BTC-USD.csv', 'col_date_name': 'Date', 'col_price': 'Adj Close'}

        if not is_electricity_constant:
            files_pm['electricity'] = {'file_name': 'Electricity data ERCOT.csv', 'col_date_name': 'Date', 'col_price': 'price'}

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
            
        df_data.fillna(method='ffill', inplace=True)
        df_data.dropna(inplace=True)
        return df_data

# def read_data(data_path, is_electricity_constant=False):
#   """
#   Reads and preprocesses Hashprice, BTC, and Electricity data from CSV files.

#   Parameters:
#   data_path (str): Path to the directory containing the CSV files.

#   Returns:
#   pd.DataFrame: A combined DataFrame containing the preprocessed data for all assets.
#   """
#   files_pm = {}

#   files_pm['hashprice'] = {'file_name': 'bitcoin-hashprice-index-20251110.csv'}
#   files_pm['hashprice']['col_date_name'] = 'timestamp'
#   files_pm['hashprice']['col_price'] = 'usd_hashprice'

#   files_pm['btc'] = {'file_name': 'BTC-USD.csv'}
#   files_pm['btc']['col_date_name'] = 'Date'
#   files_pm['btc']['col_price'] = 'Adj Close'

#   if not is_electricity_constant:
#     files_pm['electricity'] = {'file_name': 'Electricity data ERCOT.csv'}
#     files_pm['electricity']['col_date_name'] = 'Date'
#     files_pm['electricity']['col_price'] = 'price'

#   df_data = pd.DataFrame([])
#   for key, pm in files_pm.items():
#     file_name = pm['file_name']
#     col_date_name = pm['col_date_name']
#     file_path = os.path.join(data_path, file_name)
#     df_data_aux = pd.read_csv(file_path)

#     df_data_aux[col_date_name] = pd.to_datetime(df_data_aux[col_date_name])
#     df_data_aux['date_aux'] = df_data_aux[col_date_name].dt.strftime('%Y-%m-%d')
#     df_data_aux['date_aux'] = pd.to_datetime(df_data_aux['date_aux'], format='%Y-%m-%d')
#     df_data_aux = df_data_aux.pivot_table(index='date_aux', values=pm['col_price'], aggfunc='mean')

#     df_data_aux = df_data_aux[[pm['col_price']]]
#     df_data_aux.rename(columns={pm['col_price']: key}, inplace=True)
#     df_data_aux.index.name = 'date'
#     df_data_aux.sort_index(inplace=True)

#     df_data = pd.concat([df_data, df_data_aux], axis=1)
#   df_data.dropna(inplace=True)
#   return df_data
