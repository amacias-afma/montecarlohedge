import random
import numpy as np
import datetime as dt
import pandas as pd

def estimate_g_alpha(prices, investment_parameters, other_parameters, n_simulations=1000):
    """
    Estimates the alpha coefficients for the g function using a Monte Carlo approach.

    Parameters:
    prices (dict): Dictionary containing 'X' (Hashprice) and 'Y' (Electricity) price DataFrames.
    investment_parameters (dict): Dictionary containing investment parameters (kappa, k, K).
    other_parameters (dict): Dictionary containing other parameters (rho, delta_time).

    Returns:
    numpy.matrix: A matrix containing the estimated alpha coefficients.
    """

    b = []
    A = []
    kappa = investment_parameters['kappa']
    valuation_date = other_parameters['valuation_date']
    n_days_forecast = other_parameters['n_days_forecast']
    project_duration = other_parameters['project_duration']
    
    list_dates = [valuation_date + dt.timedelta(days=i) for i in range(n_days_forecast + 1 - project_duration)]

    for _ in range(n_simulations):
        i = random.choice(range(len(prices['hashprice'])))
        date = random.choice(list_dates)

        x0 = prices['hashprice'].loc[i][date]

        if isinstance(prices['electricity'], (pd.DataFrame, pd.Series)):
            y0 = prices['electricity'].loc[i][date]
        else:
            y0 = prices['electricity']

        g0 = calculate_g_montecarlo(x0, y0, date, prices, investment_parameters, other_parameters)
        v0 = x0 - kappa * y0
        b.append(g0)
        A.append([1, v0, v0**2, v0**3])

    mt_A = np.matrix(A)
    # print('mt_A', mt_A.shape)
    # print(mt_A)
    
    mt_b = np.matrix(b).T
    # print('mt_b', mt_b.shape)
    # print(mt_b)

    mt_A_inv = np.linalg.inv(mt_A.T @ mt_A)
    mt_alpha = mt_A_inv @ (mt_A.T @ mt_b)
    return mt_alpha

def calculate_g_tilde(df_prices, kappa, g_alpha):
    """
    Calculates the g_tilde value based on prices and estimated alpha coefficients.

    Parameters:
    df_prices (pd.DataFrame): DataFrame containing 'X' and 'Y' prices.
    kappa (float): Electricity conversion factor.
    g_alpha (numpy.matrix): Estimated alpha coefficients.

    Returns:
    float: The calculated g_tilde value.
    """
    v = df_prices['hashprice'] - kappa * df_prices['electricity']
    result = g_alpha[0].item()
    for i in range(1, len(g_alpha)):
        result += g_alpha[i].item() * (v**i)
    # print(v.mean(), df_prices['hashprice'].mean(), df_prices['electricity'].mean(), kappa)
    return result


def calculate_g_montecarlo(x0, y0, date, prices, investment_parameters, other_parameters):
    """
    Performs a Monte Carlo simulation to estimate the g value for a specific state.

    Parameters:
    x0 (float): Initial Hashprice.
    y0 (float): Initial Electricity price.
    date (datetime): Current date.
    prices (dict): Dictionary containing price DataFrames.
    investment_parameters (dict): Investment parameters.
    other_parameters (dict): Other parameters.

    Returns:
    float: The estimated g value (NPV - K).
    """

    kappa = investment_parameters['kappa']
    k = investment_parameters['k']
    K = investment_parameters['K']

    rho = other_parameters['rho']
    delta_time = other_parameters['delta_time']
    project_duration = other_parameters['project_duration']

    g_value = 0

    for i_aux in range(1, project_duration + 1):
        end_date = date + dt.timedelta(days=i_aux)
        delta_x = prices['hashprice'][end_date] - prices['hashprice'][date]
        
        if isinstance(prices['electricity'], (pd.DataFrame, pd.Series)):
             delta_y = prices['electricity'][end_date] - prices['electricity'][date]
        else:
             delta_y = 0

        g_aux = x0 + delta_x - kappa * (y0 + delta_y)
        g_aux[g_aux < 0] = 0
        g_aux -= k
        g_aux *= np.exp(-rho * i_aux * delta_time) * delta_time
        g_value += g_aux.mean()
    return float(g_value) - K
