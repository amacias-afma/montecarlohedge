def read_electricity_prices(type_data='real'):
    if type_data == 'real':
        file_name = 'Electricity data ERCOT.csv'
        df_electricity = read_data_csv(file_name)
        df_electricity['Date'] = pd.to_datetime(df_electricity['Date'])
    else:
        file_name = 'Electricity data ERCOT.csv'
        df_electricity = read_data_csv(file_name)
        df_electricity['Date'] = pd.to_datetime(df_electricity['Date'])

    df_elec_prices = df_electricity[df_electricity['Date'].dt.hour == 0].pivot_table(index='Date', values='price', columns='zone')
    outliers = [dt.date(2022, 12, 24)]
    # df_prices.loc[dt.date(2021, 5, 1): dt.date(2022, 6, 1)].diff().hist()
    df_elec_prices.loc[dt.date(2021, 5, 1): dt.date(2024, 6, 1)].plot()
    df_electricy_prices = df_elec_prices.mean(axis=1).resample('ME').mean()
    df_electricy_prices.plot()
    
    return df_electricity

def read_bitcoin_prices(type_data='real'):
    if type_data == 'real':
        file_name = 'BTC-USD.csv'
        df_btc_usd = read_data_csv(file_name)
        df_btc_usd.set_index('Date', inplace=True)
        df_btc_usd.index = pd.to_datetime(df_btc_usd.index)
        
        df_btc_usd.rename(columns={'Adj Close': 'Price'}, inplace=True)
        df_btc_usd = df_btc_usd[['Price']]
        df_btc_usd = df_btc_usd.astype(float)
    else:
        file_name = 'BTC-USD.csv'
        df_btc_usd = read_data_csv(file_name)
        df_btc_usd.set_index('Date', inplace=True)
        df_btc_usd.index = pd.to_datetime(df_btc_usd.index)
        
        df_btc_usd.rename(columns={'Adj Close': 'Price'}, inplace=True)
        df_btc_usd = df_btc_usd[['Price']]
        df_btc_usd = df_btc_usd.astype(float)
    return df_btc_usd