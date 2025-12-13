import random
import numpy as np
import datetime as dt
import pandas as pd

def estimate_g_alpha(prices, dates_parameters, investment_parameters, others_parameters):
    """
    Estimates the alpha coefficients for the g function using a Monte Carlo approach.

    Parameters:
    prices (dict): Dictionary containing 'X' (Hashprice) and 'Y' (Electricity) price DataFrames.
    dates_parameters (dict): Dictionary containing date-related parameters.
    investment_parameters (dict): Dictionary containing investment parameters (kappa, k, K).
    others_parameters (dict): Dictionary containing other parameters (rho, delta_time).

    Returns:
    numpy.matrix: A matrix containing the estimated alpha coefficients.
    """

    b = []
    A = []
    kappa = investment_parameters['kappa']
    valuation_date = dates_parameters['valuation_date']
    n_days_forecast = dates_parameters['n_days_forecast']
    project_duration = dates_parameters['project_duration']
    list_dates = [valuation_date + dt.timedelta(days=i) for i in range(n_days_forecast + 1 - project_duration)]

    for _ in range(1000):
        i = random.choice(range(len(prices['Z'])))
        date = random.choice(list_dates)
        x0 = prices['Z'].loc[i][date]
        if isinstance(prices['Y'], (pd.DataFrame, pd.Series)):
            y0 = prices['Y'].loc[i][date]
        else:
            y0 = prices['Y']

        g0 = calculate_g_montecarlo(x0, y0, date, prices, dates_parameters, investment_parameters, others_parameters)
        v0 = x0 - kappa * y0
        b.append(g0)
        A.append([1, v0, v0**2, v0**3])

    mt_A = np.matrix(A)
    mt_b = np.matrix(b).T
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
    v = df_prices['Z'] - kappa * df_prices['Y']
    return g_alpha[0].item() + g_alpha[1].item() * v + g_alpha[2].item() * v**2 + g_alpha[3].item() * v**3


def calculate_g_montecarlo(x0, y0, date, prices, dates_parameters, investment_parameters, others_parameters):
    """
    Performs a Monte Carlo simulation to estimate the g value for a specific state.

    Parameters:
    x0 (float): Initial Hashprice.
    y0 (float): Initial Electricity price.
    date (datetime): Current date.
    prices (dict): Dictionary containing price DataFrames.
    dates_parameters (dict): Date parameters.
    investment_parameters (dict): Investment parameters.
    others_parameters (dict): Other parameters.

    Returns:
    float: The estimated g value (NPV - K).
    """

    kappa = investment_parameters['kappa']
    k = investment_parameters['k']
    K = investment_parameters['K']

    rho = others_parameters['rho']
    delta_time = others_parameters['delta_time']
    project_duration = dates_parameters['project_duration']

    g_value = 0

    for i_aux in range(1, project_duration + 1):
        end_date = date + dt.timedelta(days=i_aux)
        # print(prices['X'][end_date])
        delta_x = prices['Z'][end_date] - prices['Z'][date]
        
        if isinstance(prices['Y'], (pd.DataFrame, pd.Series)):
             delta_y = prices['Y'][end_date] - prices['Y'][date]
        else:
             delta_y = 0

        g_aux = x0 + delta_x - kappa * (y0 + delta_y)
        g_aux[g_aux < 0] = 0
        g_aux -= k
        g_aux *= np.exp(-rho * i_aux * delta_time)
        g_value += g_aux.mean()
    return float(g_value) - K
