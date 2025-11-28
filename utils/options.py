import pandas as pd
import numpy as np
from scipy.stats import norm

def base_bs(df_prices, df_return, delta_t, r):
  """
  df_prices: Underlying assets (scenarios x assets) at a single time point
  investment: Strike price
  r: Mean of the log of the underlying asset
  expiration_time: Time until the option expires

  Returns:
  df_base: A matrix that serves as a base for option pricing
  """

  df_std = df_return.std(axis=0)
  df_mean = df_prices.mean(axis=0)

  # columns = []
  columns = [('k', 'k')]
  for asset in df_prices.columns:
    columns.append((asset, 'price'))
    columns.append((asset, 'price^2'))
    
    columns.append((asset, 'call_option'))

  df_base = pd.DataFrame([], index=df_prices.index, columns=columns)
  df_base[columns[0]] = 1
  for asset in df_prices.columns:
    # First, calculate the mean difference
    df_base[(asset, 'price')] = df_prices[asset] - df_mean[asset]
    # df_base[(asset, 'price')] = df_prices[asset]
    df_base[(asset, 'price^2')] = (df_prices[asset] - df_mean[asset])**2
    
    df_base[(asset, 'call_option')] = blsprice(df_prices[asset], df_mean[asset], r, delta_t, df_std[asset])
  return df_base
    
def base_bs_delta(df_prices, df_return, delta_t, r):
  """
  df_prices: Underlying assets (scenarios x assets) at a single time point
  investment: Strike price
  r: Mean of the log of the underlying asset
  expiration_time: Time until the option expires

  Returns:
  df_base: A matrix that serves as a base for option pricing
  """

  df_std = df_return.std(axis=0)
  df_mean = df_prices.mean(axis=0)

  # columns = [('k', 'k')]
  columns = []
  for asset in df_prices.columns:
    columns.append((asset, 'k'))
    columns.append((asset, 'price'))
    columns.append((asset, 'delta_option'))

  df_base = pd.DataFrame([], index=df_prices.index, columns=columns)
  for asset in df_prices.columns:
    # First, calculate the mean difference
    # df_base[(asset, 'k')] = df_return[asset]
    # df_base[(asset, 'price')] = df_prices[asset] * df_return[asset]

    # df_base[(asset, 'delta_option')] = blsdelta(df_prices[asset], df_mean[asset], r, delta_t, df_std[asset])
    # df_base[(asset, 'delta_option')] *= df_return[asset]
    
    df_base[(asset, 'k')] = df_return[asset]
    df_base[(asset, 'price')] = (df_prices[asset] - df_mean[asset]) * df_return[asset]
    df_base[(asset, 'price^2')] = (df_prices[asset] - df_mean[asset])**2 * df_return[asset]

    df_base[(asset, 'delta_option')] = blsdelta(df_prices[asset], df_mean[asset], r, delta_t, df_std[asset])
    df_base[(asset, 'delta_option')] *= df_return[asset]
      
  return df_base


def blsprice(price, strike, rate, time, volatility):
  """
  price: Spot price of the underlying asset
  strike: Strike price
  rate: Risk-free interest rate
  time: Time to expiration
  volatility: Volatility of the asset

  Returns:
  Call option price calculated using the Black-Scholes formula.
  """
  d1 = (np.log(price / strike) + (rate + 0.5 * volatility**2) * time) / (volatility * np.sqrt(time))
  d2 = d1 - volatility * np.sqrt(time)

  call_price = price * norm.cdf(d1) - strike * np.exp(-rate * time) * norm.cdf(d2)

  return call_price

def blsdelta(price, strike, rate, time, volatility):
    """
    Calculate the delta of a call and put option using the Black-Scholes formula.

    Inputs:
        price: Current price of the underlying asset.
        strike: Strike (exercise) price of the option.
        rate: Continuously compounded risk-free rate of return over the life of the option.
        time: Time to expiration of the option, expressed in years.
        volatility: Annualized asset price volatility (standard deviation of return), expressed as a positive decimal.

    Outputs:
        dCdS: Delta of the call option.
        dPdS: Delta of the put option.
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