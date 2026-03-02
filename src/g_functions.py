import os
import json
import random
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Helper Data & Simulation Functions
# -------------------------------------------------------------------

def calculate_delta_asset(delta_prices, asset_name, dates, date):
    """
    Calculates the delta (return/difference) for a specific asset over a list of dates.
    
    Parameters:
    delta_prices (dict): A dictionary of DataFrames containing the delta prices for each asset.
    asset_name (str): The name of the asset (e.g., 'hashprice', 'electricity', 'discount_factor').
    dates (list): A list of future dates to calculate the delta against.
    date (datetime): The base date from which deltas are calculated.
    
    Returns:
    pd.DataFrame: A DataFrame containing the deltas.
    """
    delta_prices_rep = pd.concat([delta_prices[asset_name][date]] * len(dates), axis=1)
    delta_prices_rep.columns = dates
    delta_x = delta_prices[asset_name][dates] - delta_prices_rep 
    return delta_x

def calculate_g_montecarlo(x0, y0, date, prices, delta_prices, investment_parameters, other_parameters):
    """
    Performs a Monte Carlo simulation to estimate the g value for a specific state (NPV without K).

    Parameters:
    x0 (float): Initial Hashprice.
    y0 (float): Initial Electricity price.
    date (datetime): Current date.
    prices (dict): Dictionary containing price DataFrames.
    delta_prices (dict): Dictionary containing price return DataFrames.
    investment_parameters (dict): Investment parameters (kappa, k, K).
    other_parameters (dict): Other parameters (rho, delta_time, project_duration).

    Returns:
    float: The estimated g value.
    """
    kappa = investment_parameters['kappa']
    k = investment_parameters['k']
    
    delta_time = other_parameters['delta_time']
    project_duration = other_parameters['project_duration']
    dates = [date + dt.timedelta(days=i_aux) for i_aux in range(1, project_duration + 1)]
    
    # Calculate Hashprice changes
    asset_name = 'hashprice'
    delta_x = calculate_delta_asset(delta_prices, asset_name, dates, date)

    # Calculate Electricity price changes if it is stochastic
    if isinstance(prices['electricity'], (pd.DataFrame, pd.Series)):
        asset_name = 'electricity'
        delta_y = calculate_delta_asset(delta_prices, asset_name, dates, date)
    else:
        delta_y = 0

    # Calculate Discount Factor
    df_discount_factor = np.exp(calculate_delta_asset(delta_prices, 'discount_factor', dates, date))
    
    # Calculate G
    g_aux = x0 * np.exp(delta_x) - kappa * (y0 + delta_y)
    g_aux = g_aux * df_discount_factor
    g_aux[g_aux < 0] = 0
    g_aux -= k
    g_aux *= delta_time

    g_value = g_aux.sum(axis=1).mean()
    
    return float(g_value)

def construct_g_data(n_simulations, prices, delta_prices, list_dates, investment_parameters, other_parameters):
    """
    Constructs the dataset of simulated g values across random paths and dates.
    
    Parameters:
    n_simulations (int): Number of simulations to run.
    prices (dict): Dictionary containing price DataFrames.
    delta_prices (dict): Dictionary containing price return DataFrames.
    list_dates (list): Available dates for simulation sampling.
    investment_parameters (dict): Investment parameters.
    other_parameters (dict): Other parameters.
    
    Returns:
    list: A list of tuples containing (date, x0, y0, g0).
    """
    data = []
    for _ in range(n_simulations):
        i = random.choice(range(len(prices['hashprice'])))
        date = random.choice(list_dates)

        x0 = prices['hashprice'].loc[i][date]

        if isinstance(prices['electricity'], (pd.DataFrame, pd.Series)):
            y0 = prices['electricity'].loc[i][date]
        else:
            y0 = prices['electricity']

        g0 = calculate_g_montecarlo(x0, y0, date, prices, delta_prices, investment_parameters, other_parameters)
        data.append((date, x0, y0, g0))
    return data

def construct_matrix(data, n_basis, kappa):
    """
    Constructs the design matrix A and response vector b for polynomial regression.
    
    Parameters:
    data (list): List of simulated tuples (date, x, y, g).
    n_basis (int): Number of basis functions (polynomial degree).
    kappa (float): Electricity conversion factor.
    
    Returns:
    tuple: (A, b, x_list, y_list)
    """
    A = {i: [] for i in range(1, n_basis + 1)}
    b = []
    x_list, y_list = [], []
    for date, x, y, g in data:
        x_list.append(x)
        y_list.append(y)
        v = x - kappa * y
        for i in A:
            A[i].append([v**j for j in range(i + 1)])
        b.append(g)
    return A, b, x_list, y_list


# -------------------------------------------------------------------
# Model Fitting & Selection
# -------------------------------------------------------------------

def calculate_alpha(A, b):
    """
    Calculates the alpha coefficients using OLS regression for each basis dimension.
    
    Parameters:
    A (dict): Dictionary of design matrices for different polynomial degrees.
    b (list): Response vector.
    
    Returns:
    dict: A dictionary mapping polynomial degree to alpha coefficients.
    """
    mt_b = np.matrix(b).T
    alpha = {}
    for i, A_aux in A.items():
        mt_A_aux = np.matrix(A_aux)
        # Using pseudo-inverse to prevent singular matrix errors
        mt_A_aux_inv = np.linalg.pinv(mt_A_aux.T @ mt_A_aux)
        mt_alpha_aux = mt_A_aux_inv @ (mt_A_aux.T @ mt_b)
        alpha[i] = mt_alpha_aux
    return alpha

def calculate_optimal_basis(A_test, b_test, alpha):
    """
    Determines the optimal number of basis functions by comparing prediction error on a test set.
    
    Parameters:
    A_test (dict): Dictionary of test design matrices.
    b_test (list): Test response vector.
    alpha (dict): Dictionary of estimated alpha coefficients.
    
    Returns:
    tuple: (optimal_basis_degree, results_error_dict)
    """
    results_error = {}

    i_min = len(A_test)
    std_min = np.inf
    for i, A_i in A_test.items():
        mt_A_i = np.matrix(A_i)
        
        mt_alpha_i = alpha[i]
        g0_i = np.squeeze(mt_A_i @ mt_alpha_i).T
        mt_b = np.matrix(b_test).T
        error_i = g0_i - mt_b
        
        std_i = np.std(error_i)
        
        # Select basis if error improves by more than 5%
        if std_i / std_min - 1 < -0.05:
            i_min = i
            std_min = std_i
            
        results_error[i] = np.std(error_i)
    return i_min, results_error


# -------------------------------------------------------------------
# Main Public API
# -------------------------------------------------------------------

def estimate_or_load_g_alpha(prices, delta_prices, investment_parameters, other_parameters, g_simulations_train=1000, g_simulations_test=100, model_type="default", filepath="results/g_alpha.json", force_recalculate=False):
    """
    Main orchestrator function: Estimates the optimal alpha coefficients for the g function, 
    or loads them from a JSON file if they already exist and the parameters match.
    
    Parameters:
    prices (dict): Price DataFrames.
    delta_prices (dict): Price delta DataFrames.
    investment_parameters (dict): Investment configuration.
    other_parameters (dict): Other configuration.
    g_simulations_train (int): Number of training simulations.
    g_simulations_test (int): Number of testing simulations.
    model_type (str): Identifier for the model run.
    filepath (str): Where to save/load the coefficients.
    force_recalculate (bool): If True, bypasses loading and forces recalculation.
    
    Returns:
    numpy.matrix: The optimal alpha coefficients.
    """
    # Helper function to convert parameters to string for JSON serialization
    def serialize_params(params):
        if not isinstance(params, dict):
            return params
        serialized = {}
        for k, v in params.items():
            if isinstance(v, dt.datetime):
                serialized[k] = v.isoformat(timespec='seconds')
            elif isinstance(v, dict):
                serialized[k] = serialize_params(v)
            else:
                serialized[k] = v
        return serialized

    alpha_params = {
        'investment_parameters': serialize_params(investment_parameters),
        'other_parameters': serialize_params(other_parameters),
        'g_simulations_train': g_simulations_train, 
        'g_simulations_test': g_simulations_test,
        'model_type': model_type
    }

    if os.path.exists(filepath) and not force_recalculate:
        try:
            with open(filepath, 'r') as f:
                saved_data = json.load(f)
            
            # Check if parameters match
            if saved_data.get('parameters') == alpha_params:
                print(f"Loading existing g_alpha from {filepath}...")
                return np.matrix(saved_data['g_alpha'])
            else:
                print(f"Parameters in {filepath} do not match current parameters. Recalculating...")
        except Exception as e:
            print(f"Error loading {filepath}: {e}. Recalculating...")
            
    print(f"Estimating optimal G parameters (train simulations={g_simulations_train}, test simulations={g_simulations_test}). This may take some time...")

    valuation_date = other_parameters['valuation_date']
    project_duration = other_parameters['project_duration']
    n_days_forecast = other_parameters['n_days_forecast']

    list_dates = [valuation_date + dt.timedelta(days=i) for i in range(1, n_days_forecast + 1 - project_duration)]

    data_train = construct_g_data(g_simulations_train, prices, delta_prices, list_dates, investment_parameters, other_parameters)
    data_test = construct_g_data(g_simulations_test, prices, delta_prices, list_dates, investment_parameters, other_parameters)

    kappa = investment_parameters['kappa']
    n_basis = 10
    
    A_train, b_train, x_train, y_train = construct_matrix(data_train, n_basis, kappa)
    A_test, b_test, x_test, y_test = construct_matrix(data_test, n_basis, kappa)

    alpha = calculate_alpha(A_train, b_train)

    optimal_basis, results_error = calculate_optimal_basis(A_test, b_test, alpha)
    optimal_alpha = alpha[optimal_basis]
    
    # Plotting the errors across different basis dimensions
    plt.figure(figsize=(8, 5))
    plt.plot(list(results_error.keys()), list(results_error.values()), marker='o', label="Test Error")
    plt.plot(optimal_basis, results_error[optimal_basis], 'xr', markersize=10, label=f"Optimal Basis: {optimal_basis}")
    plt.title("Error Optimization for Basis Choice")
    plt.xlabel("Polynomial Degree (n_basis)")
    plt.ylabel("Standard Deviation of Error")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print(f"Optimal basis chosen: {optimal_basis}, Error Profile: {results_error}")
    
    # Save the new results
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        save_data = {
            'parameters': alpha_params,
            'g_alpha': optimal_alpha.tolist()
        }
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=4)
        print(f"Saved estimated g_alpha to {filepath}.")
    except Exception as e:
        print(f"Failed to save g_alpha to {filepath}: {e}")
        
    return optimal_alpha

def calculate_g_tilde(df_prices, kappa, g_alpha):
    """
    Calculates the g_tilde value based on prices and estimated alpha coefficients.

    Parameters:
    df_prices (pd.DataFrame): DataFrame containing 'hashprice' and 'electricity' prices.
    kappa (float): Electricity conversion factor.
    g_alpha (numpy.matrix): Estimated alpha coefficients.

    Returns:
    float or pd.Series: The calculated g_tilde values.
    """
    v = df_prices['hashprice'] - kappa * df_prices['electricity']
    
    # Initialize result with the intercept (alpha_0)
    result = g_alpha[0].item()
    
    # Add polynomial terms based on the length of g_alpha
    for i in range(1, len(g_alpha)):
        result += g_alpha[i].item() * (v**i)

    return result
