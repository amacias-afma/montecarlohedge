import numpy as np
import numpy as np
import datetime as dt
from scipy.stats import norm
import pandas as pd

import src.g_functions as gf

def margrabe_exchange_option_price(Sx, Sy, sigma_x, sigma_y, qx, qy, rho, r, T):
  """
  Calculate the price of an exchange option using Margrabe's formula.

  Parameters:
  Sx (float): Price of asset X.
  Sy (float): Price of asset Y.
  sigma_x (float): Volatility of asset X.
  sigma_y (float): Volatility of asset Y.
  qx (float): Dividend yield of asset X.
  qy (float): Dividend yield of asset Y.
  rho (float): Correlation between asset X and asset Y.
  r (float): Risk-free interest rate.
  T (float): Time to expiration.

  Returns:
  float: Value of the exchange option.
  """

  sigma = np.sqrt(sigma_x**2 + sigma_y**2 - 2 * rho * sigma_x * sigma_y)
  d1 = (np.log(Sx / Sy) + (qy - qx + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
  d2 = d1 - sigma * np.sqrt(T)
  Sx_disc = Sx * np.exp(-qx * T)
  Sy_disc = Sy * np.exp(-qy * T)
  price = Sx_disc * norm.cdf(d1) - Sy_disc * norm.cdf(d2)
  return price

def margrabe_exchange_option_delta(Sx, Sy, sigma_x, sigma_y, qx, qy, rho, r, T):
  """
  Calculate the deltas of an exchange option using Margrabe's formula.

  Parameters:
  Sx (float): Price of asset X.
  Sy (float): Price of asset Y.
  sigma_x (float): Volatility of asset X.
  sigma_y (float): Volatility of asset Y.
  qx (float): Dividend yield of asset X.
  qy (float): Dividend yield of asset Y.
  rho (float): Correlation between asset X and asset Y.
  r (float): Risk-free interest rate.
  T (float): Time to expiration.

  Returns:
  tuple: (delta_x, delta_y)
  """

  sigma = np.sqrt(sigma_x**2 + sigma_y**2 - 2 * rho * sigma_x * sigma_y)
  d1 = (np.log(Sx / Sy) + (qy - qx + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
  d2 = d1 - sigma * np.sqrt(T)

  delta_x = np.exp(-qx * T) * norm.cdf(d1)
  delta_y = np.exp(-qy * T) * norm.cdf(d2)

  return delta_x, delta_y


def blsprice(price, strike, rate, time, volatility):
    """
    Calculate the call option price using the Black-Scholes formula.

    Parameters:
    price (float): Spot price of the underlying asset.
    strike (float): Strike price.
    rate (float): Risk-free interest rate.
    time (float): Time to expiration.
    volatility (float): Volatility of the asset.
    
    Returns:
    float: Call option price.
    """
    d1 = (np.log(price / strike) + (rate + 0.5 * volatility**2) * time) / (volatility * np.sqrt(time))
    d2 = d1 - volatility * np.sqrt(time)
    
    call_price = price * norm.cdf(d1) - strike * np.exp(-rate * time) * norm.cdf(d2)
    
    return call_price

def blsdelta(price, strike, rate, time, volatility):
    """
    Calculate the delta of a call option using the Black-Scholes formula.

    Parameters:
    price (float): Current price of the underlying asset.
    strike (float): Strike (exercise) price of the option.
    rate (float): Continuously compounded risk-free rate of return.
    time (float): Time to expiration of the option (in years).
    volatility (float): Annualized asset price volatility.

    Returns:
    float: Delta of the call option.
    """
    # Calculate d1, which is used to find the option delta
    d1 = (np.log(price / strike) + (rate + 0.5 * volatility**2) * time) / (volatility * np.sqrt(time))

    # Calculate N(d1), the CDF of the standard normal distribution at d1
    N1 = norm.cdf(d1)

    # Delta of the call option
    dCdS = N1

    # # Delta of the put option
    # dPdS = N1 - 1

    return dCdS

def base_bs(df_prices, df_return, delta_t, r, kappa, basis_type=None):
  """
  Generates the basis functions for the regression using Black-Scholes or Margrabe pricing.

  Parameters:
  df_prices (pd.DataFrame): Underlying assets prices (scenarios x assets).
  df_return (pd.DataFrame): Returns of the underlying assets.
  delta_t (float): Time step.
  r (float): Risk-free rate.
  kappa (float): Electricity conversion factor.
  basis_type (dict, optional): Dictionary specifying polynomial order and option basis type.

  Returns:
  pd.DataFrame: A DataFrame containing the basis functions.
  """
  df_return_aux = np.log(1 + df_return / df_prices)

  df_ret_std = df_return_aux.std()
  df_ret_cor = df_return_aux.corr()
  df_ret_mean = df_return_aux.mean()
  df_mean = df_prices.mean()

  assets = {}
  for i, asset in enumerate(df_prices.columns):
    assets[i] = asset
  
  if basis_type is None:
    basis_type = {'pol_order': 2}
    basis_type['option_basis'] = 0

  df_base = pd.DataFrame([], index=df_prices.index)
  df_base['call_po_0'] = 1

  for asset in df_prices.columns:
    for order in range(1, basis_type['pol_order'] + 1):
      df_base[f'call_po_{order}_{asset}'] = (df_prices[asset] - df_mean[asset]) ** order

  if basis_type['option_basis'] == 'black-scholes':
    for asset in df_prices.columns:
      # print('-' * 10)
      # print(df_prices[asset].head())
      # print(asset, df_mean[asset], r, delta_t, df_ret_std[asset])
      df_base['call_option'] = blsprice(df_prices[asset], df_mean[asset], r, delta_t, df_ret_std[asset] * np.sqrt(252))
  elif basis_type['option_basis'] == 'margrabe':
    kappa_aux = df_prices[assets[0]].mean() / df_prices[assets[1]].mean()
    corr = df_ret_cor[assets[0]][assets[1]]
    df_base['call_option'] = margrabe_exchange_option_price(df_prices[assets[0]], kappa_aux * df_prices[assets[1]], \
                                                                   df_ret_std[assets[0]], kappa_aux * df_ret_std[assets[1]], \
                                                                   df_ret_mean[assets[0]], df_ret_mean[assets[1]], \
                                                                   corr, r, delta_t)
  return df_base

def base_bs_delta(df_prices, df_return, delta_t, r, kappa, basis_type=None):
  """
  Generates the delta-weighted basis functions for the regression.

  Parameters:
  df_prices (pd.DataFrame): Underlying assets prices.
  df_return (pd.DataFrame): Returns of the underlying assets.
  delta_t (float): Time step.
  r (float): Risk-free rate.
  kappa (float): Electricity conversion factor.
  basis_type (dict, optional): Basis type configuration.

  Returns:
  pd.DataFrame: A DataFrame containing the delta-weighted basis functions.
  """
  df_return_aux = np.log(1 + df_return / df_prices)

  df_ret_std = df_return_aux.std()
  df_ret_cor = df_return_aux.corr()
  df_ret_mean = df_return_aux.mean()
  df_mean = df_prices.mean()

  if basis_type is None:
    basis_type = {'pol_order': 2}
    basis_type['option_basis'] = 0

  assets = {}
  for i, asset in enumerate(df_prices.columns):
    assets[i] = asset

  df_base = pd.DataFrame([], index=df_prices.index)
  for asset in df_prices.columns: # Fix: Populate assets dictionary correctly
    for order in range(basis_type['pol_order']):
      df_base[f'dl_po_{order}_{asset}'] = (df_prices[asset] - df_mean[asset]) ** order
      df_base[f'dl_po_{order}_{asset}'] *= df_return[asset]

  if basis_type['option_basis'] == 'black-scholes':
    for asset in df_prices.columns:
      df_base[f'dl_option_{asset}'] = blsdelta(df_prices[asset], df_mean[asset], r, delta_t, df_ret_std[asset] * np.sqrt(252))
      df_base[f'dl_option_{asset}'] *= df_return[asset]
  elif basis_type['option_basis'] == 'margrabe':
    corr = df_ret_cor[assets[0]][assets[1]]
    # kappa_aux = df_prices[assets[0]].mean() / df_prices[assets[1]].mean()
    kappa_aux = kappa
    delta_x, delta_y = margrabe_exchange_option_delta(df_prices[assets[0]], kappa_aux * df_prices[assets[1]], \
                                                                   df_ret_std[assets[0]], kappa_aux * df_ret_std[assets[1]], \
                                                                   df_ret_mean[assets[0]], df_ret_mean[assets[1]], \
                                                                   corr, r, delta_t)
    df_base[f'dl_option_{assets[0]}'] = delta_x * df_return[assets[0]]
    print('-------delta_y-----------------')
    print(delta_y)
    print('-------df_return[assets[1]]-----------------')
    print(df_return[assets[1]])
    df_base[f'dl_option_{assets[1]}'] = delta_y * df_return[assets[1]]

  return df_base

def optimization(df_m, df_c):
  """
  Performs the regression optimization to find the continuation value coefficients.

  Parameters:
  df_m (pd.DataFrame): Matrix of basis functions (independent variables).
  df_c (pd.DataFrame): Vector of option values (dependent variable).

  Returns:
  tuple: (mt_delta, opt_value) where mt_delta are the coefficients and opt_value is the residual sum of squares.
  """
  mt_c = np.matrix(df_c)
  mt_m = np.matrix(df_m)
  mt_delta = np.linalg.inv(mt_m.T @ mt_m) @ (mt_m.T @ mt_c)
  mt_res_aux = mt_m @ mt_delta - mt_c
  opt_value = mt_res_aux.T @ mt_res_aux
  return mt_delta, opt_value

def calculate_dates_project_window(other_parameters):
  # --- Extract Parameters ---
  valuation_date = other_parameters['valuation_date']
  optionality_window_days = other_parameters['optionality_window']
  start_days = other_parameters['starting_valuation_window']

  # --- Calculate Monthly Dates ---
  dates_project_window = []
  days_in_month = (365*4 + 1) / 48  # Average month length (~30.416 days)

  current_day_offset = start_days
  end_day_offset = start_days + optionality_window_days

  # Generate dates from start to end of the optionality window
  while current_day_offset <= end_day_offset + 1:
      # Calculate specific date
      date = valuation_date + dt.timedelta(days=int(current_day_offset))
      dates_project_window.append(date)
      # Increment by one month
      current_day_offset += days_in_month
  return dates_project_window

def calculate_mean_prices(prices, dates_project_window):
  mean_prices = {}
  dates_columns = prices['btc'].columns

  for asset, prices_aux in prices.items():
    if isinstance(prices_aux, (pd.DataFrame, pd.Series)):
        mean_prices[asset] = pd.DataFrame([], \
            index=prices_aux.index, columns=dates_project_window)
        date_prev = dates_project_window[0]

        for date in dates_project_window[1:]:
            dates_aux = dates_columns[(dates_columns >= date_prev) & (dates_columns < date)]

            mean_prices[asset][date_prev] = prices_aux[dates_aux].mean(axis=1)
            date_prev = date

        mean_prices[asset].drop(columns=date, inplace=True)
  return mean_prices


def calculate_real_option(prices, basis_type, investment_parameters, other_parameters, g_alpha):
    """
    Calculates the real option value using the Least Squares Monte Carlo (LSM) method.

    Parameters:
    prices (dict): Simulated prices for assets.
    basis_type (dict): Configuration for basis functions.
    dates_project_window (list): List of dates in the project window.
    dates_parameters (dict): Date parameters.
    investment_parameters (dict): Investment parameters.
    other_parameters (dict): Other parameters.

    Returns:
    tuple: (df_option_value, df_call_value, df_intrinsic_value, df_delta, functional_result)
    """
    kappa = investment_parameters['kappa']
    rho = other_parameters['rho']
    delta_time = 1 / 12
    # delta_time = other_parameters['delta_time']
    valuation_date = other_parameters['valuation_date']
    starting_day = other_parameters['starting_valuation_window']

    dates_project_window = calculate_dates_project_window(other_parameters) # Calculate the dates for the project window
    mean_prices = calculate_mean_prices(prices, dates_project_window) # Calculate the mean prices for the project window
    print('--------------------')
    print(dates_project_window)
    dates_project_window = dates_project_window[:-1]
    print(dates_project_window)
    print('--------------------')

    df_project_value = pd.DataFrame([], columns=dates_project_window)
    df_option_value = pd.DataFrame([], columns=dates_project_window)
    df_call_value = pd.DataFrame([], columns=dates_project_window) # This will store the continuation value
    df_intrinsic_value = pd.DataFrame([], columns=dates_project_window)
    df_prices_nxt = pd.DataFrame([])

    # --- Backward Induction Loop ---
    end_date = dates_project_window[-1]
    date_nxt = dates_project_window[-1]
    date_start = dates_project_window[0]

    functional_result = {}
    prices_g = {}
    prices_g['hashprice'] = prices['hashprice']
    prices_g['electricity'] = prices['electricity']

    hedge_prices = {'btc': mean_prices['btc']}
    
    if 'electricity' in mean_prices:
        hedge_prices['electricity'] = mean_prices['electricity']

    for date in reversed(dates_project_window):
        # print(end_date, date, date_nxt)
        # 1. Calculate the project value (NPV - K) and intrinsic value for the current date
        print('date', date)
        df_prices_g = obtain_prices(prices_g, date)
        df_project_value[date] = gf.calculate_g_tilde(df_prices_g, kappa, g_alpha)

        df_intrinsic_value[date] = df_project_value[date].copy()
        df_intrinsic_value.loc[df_intrinsic_value[date] < 0, date] = 0 # Intrinsic value cannot be negative

        # 2. Set the terminal condition at the last date (T)
        if date == end_date:
            df_option_value[date] = df_intrinsic_value[date].copy()
            df_call_value[date] = df_option_value[date]
            df_prices_nxt = obtain_prices(hedge_prices, date)

        # 3. For all other dates, calculate the continuation value
        else:
            df_prices = obtain_prices(hedge_prices, date)
            discount_factor = np.exp(-rho * delta_time)
            df_return = df_prices_nxt * discount_factor - df_prices

            # Known option values from the next time step
            df_option_nxt = df_option_value[[date_nxt]].copy() * discount_factor
            df_option_nxt.rename(columns={date_nxt: 'option'}, inplace=True)

            prices_std = df_prices.std()

            if np.abs(prices_std.sum()) < 1e-10:
                df_return_aux = pd.concat([df_return, df_option_nxt], axis=1)
                df_return_aux_cov = df_return_aux.cov()
                df_return_cov = df_return_aux_cov.loc[df_return.columns][df_return.columns]
                df_option_cov = df_return_aux_cov.loc[df_return.columns][['option']]
                phi = np.linalg.inv(np.matrix(df_return_cov)) @ np.matrix(df_option_cov)

                option_aux = np.matrix(df_option_nxt) - np.matrix(df_return) @ phi
                option_aux = option_aux.mean()
                option_aux = float(option_aux)
            else:
                # Create the basis functions (C_a(x,y) in the paper) for the regression
                df_k = base_bs(df_prices, df_return, delta_time, rho, kappa, basis_type).fillna(0)
                df_h = base_bs_delta(df_prices, df_return, delta_time, rho, kappa, basis_type).fillna(0)
                df_m = pd.concat((df_k, df_h), axis=1)
                print(df_m.head())
                print(df_m.corr())

                # Run regression to find coefficients (γ_a^j) and estimate continuation value
                mt_delta, opt_value = optimization(df_m, df_option_nxt)
                functional_result[date] = opt_value.item()
                df_delta = pd.DataFrame(mt_delta, index=df_m.columns)

                option_aux = np.matrix(df_k) @ mt_delta[: df_k.shape[1], :]
            df_call_value[date] = option_aux # Store the continuation value

            # 4. Determine the option value: V = max(Continuation, Intrinsic)
            df_option_value[date] = option_aux
            if date >= valuation_date + dt.timedelta(days=starting_day):
                df_option_value.loc[df_option_value[date] < df_intrinsic_value[date], date] = df_intrinsic_value[date]

            # Update for the next iteration
            date_nxt = date
            df_prices_nxt = df_prices.copy()
        
    return df_option_value, df_call_value, df_intrinsic_value, df_delta, functional_result

def obtain_prices(prices, date):
    """
    Extracts prices for a specific date from the prices dictionary.

    Parameters:
    prices (dict): Dictionary of price DataFrames.
    date (datetime): The date to extract prices for.

    Returns:
    pd.DataFrame: DataFrame containing prices for the specified date.
    """
    list_dfs = []
    keys = []
    reference_index = None

    # Find a reference index from any DataFrame/Series in prices
    for price, value in prices.items():
        if isinstance(value, (pd.DataFrame, pd.Series)):
            reference_index = value.index
            break
            
    # Default to single row if no DataFrame found (fallback)
    if reference_index is None:
        reference_index = [0]

    for price, value in prices.items():
        keys.append(price)
        if isinstance(value, (pd.DataFrame, pd.Series)):
            list_dfs.append(value[[date]])
        else:
            # Handle constant/scalar value by broadcasting to reference index
            list_dfs.append(pd.DataFrame(value, index=reference_index, columns=[date]))

    df_prices_aux = pd.concat(list_dfs, axis=1)
    df_prices_aux.columns = keys
    return df_prices_aux