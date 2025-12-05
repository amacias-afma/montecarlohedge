import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg

def fit_electricity(df_data_ret):
  """
  Fits a seasonal model with AR(1) residuals to electricity prices.

  Parameters:
  df_data_ret (pd.DataFrame): DataFrame containing electricity returns/prices.

  Returns:
  tuple: (df_electricity, electricity_params) where df_electricity contains the model components and electricity_params contains the fitted parameters.
  """
  column = 'electricity'
  df_elec_log = df_data_ret[[column]].copy()
  df_elec_log['year'] = df_elec_log.index.year

  T_year = 364
  T_month = 364 / 12

  df_elec_log.index
  dates = df_elec_log.index
  date_min = dates.min()
  df_elec_log['days'] = (dates - date_min).days
  df_elec_log['days'] = df_elec_log['days'].astype(int)
  df_elec_log['sin_year'] = np.sin(2 * np.pi * df_elec_log['days'] / T_year)
  df_elec_log['cos_year'] = np.cos(2 * np.pi * df_elec_log['days'] / T_year)
  df_elec_log['sin_month'] = np.sin(2 * np.pi * df_elec_log['days'] / T_month)
  df_elec_log['cos_month'] = np.cos(2 * np.pi * df_elec_log['days'] / T_month)
  df_elec_log['constant'] = 1

  X = np.matrix(df_elec_log[['constant', 'sin_year', 'cos_year', 'sin_month', 'cos_month']])
  Y = np.matrix(df_elec_log[[column]])

  alpha = np.linalg.inv(X.T @ X) @ (X.T @ Y)

  df_elec_log[column+'_model'] = X @ alpha
  df_elec_log['residual'] = df_elec_log[column] - df_elec_log[column+'_model']

  # Model X_t using an AR(1) process (a simple mean-reverting model)
  # We use trend='n' because the residuals should be zero-mean by definition
  ar_model = AutoReg(df_elec_log['residual'].values, lags=1, trend='n')
  ar_model_fit = ar_model.fit()

  # Get the parameters for our simulation
  # phi is the mean-reversion parameter (autoregressive term)
  phi = ar_model_fit.params[0]

  # sigma_epsilon is the standard deviation of the random shocks
  sigma_epsilon = np.std(ar_model_fit.resid)
  # Generate the shocks
  shocks = np.random.normal(0, sigma_epsilon, len(df_elec_log))

  # Generate the AR(1) process
  df_elec_log['ar'] = phi * df_elec_log['residual'].shift()
  df_elec_log['model'] = df_elec_log['ar'] + df_elec_log[column+'_model']
  df_elec_log['residual_ar'] = df_elec_log[column] - df_elec_log['model']

  df_electricity = df_elec_log[[column, 'model', 'residual_ar']]

  electricity_params = {
    'alpha': alpha,  # Seasonal coefficients
    'phi': phi,  # AR(1) coefficient
    'sigma_epsilon': sigma_epsilon,  # Innovation std dev
    'T_year': T_year,
    'T_month': T_month,
    'date_min': date_min,
    'last_elec_log_price': df_elec_log[column].iloc[-1],
    'last_residual': df_elec_log['residual_ar'].iloc[-1]  # Last residual for AR(1) initialization
  }
  return df_electricity, electricity_params

def calculate_multivariate_params(df_combined):
    """
    Calculates the multivariate parameters (mean, covariance) for the asset returns.

    Parameters:
    df_combined (pd.DataFrame): DataFrame containing returns of all assets.

    Returns:
    dict: A dictionary containing daily and annual means, covariances, and correlations.
    """
    # Calculate multivariate parameters
    mu_daily = df_combined.mean().values
    cov_daily = df_combined.cov().values
    corr_matrix = df_combined.corr().values
    sigma_daily = np.sqrt(np.diag(cov_daily))

    # Annualize
    mu_annual = mu_daily * 365
    cov_annual = cov_daily * 365
    sigma_annual = sigma_daily * np.sqrt(365)

    multivariate_params = {
        'mu_daily': mu_daily,
        'mu_annual': mu_annual,
        'cov_daily': cov_daily,
        'cov_annual': cov_annual,
        'corr_matrix': corr_matrix,
        'sigma_daily': sigma_daily,
        'sigma_annual': sigma_annual,
        'series_names': df_combined.columns.tolist()
    }
    return multivariate_params

def simulate_multivariate_with_electricity(multivariate_params, electricity_params, n_days, 
                                           n_simulations=1000, start_date=None, random_seed=None):
    """
    Simulate multivariate paths including:
    - Hashprice and BTC returns (Brownian motion with Itô correction)
    - Electricity residuals (AR(1) with correlation to hashprice/BTC)
    - Electricity prices (seasonal component + AR(1) residuals)
    
    Parameters:
    -----------
    multivariate_params : dict
        Parameters for multivariate model including hashprice, btc, and electricity_residual
    electricity_params : dict
        Parameters for electricity model (seasonal, AR(1))
    n_days : int
        Number of days to simulate
    n_simulations : int
        Number of simulation paths
    start_date : pd.Timestamp, optional
        Start date for simulation (for seasonal component)
    random_seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    dict : Dictionary with simulated paths
        - 'hashprice': returns paths
        - 'btc': returns paths
        - 'electricity_residual': residual paths
        - 'electricity': price paths (log prices)
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    # Get indices for each series
    series_names = multivariate_params['series_names']
    hashprice_idx = series_names.index('hashprice')
    btc_idx = series_names.index('btc')
    electricity_idx = series_names.index('electricity')

    # Extract parameters
    mu_daily = multivariate_params['mu_daily']
    cov_daily = multivariate_params['cov_daily']

    # Electricity parameters
    phi = electricity_params['phi']
    sigma_epsilon = electricity_params['sigma_epsilon']
    alpha = electricity_params['alpha']
    T_year = electricity_params['T_year']
    T_month = electricity_params['T_month']
    date_min = electricity_params['date_min']
    last_elec_log_price = electricity_params['last_elec_log_price']
    last_elec_residual = electricity_params['last_residual']

    # Set start date for seasonal component (use last date from training data)
    if start_date is None:
        start_date = df_data_ret.index[-1] + pd.Timedelta(days=1)

    # Initialize arrays
    n_series = len(series_names)
    paths = np.zeros((n_simulations, n_series, n_days + 1))

    # Initialize electricity residuals with last observed residual
    paths[:, electricity_idx, 0] = last_elec_residual

    # Cholesky decomposition for correlated shocks
    try:
        L = np.linalg.cholesky(cov_daily)
    except np.linalg.LinAlgError:
        print("Warning: Covariance matrix not positive definite. Adding regularization.")
        cov_daily = cov_daily + np.eye(len(cov_daily)) * 1e-8
        L = np.linalg.cholesky(cov_daily)

    # Extract variances for Itô correction (only for Brownian motion series)
    sigma_squared_daily = np.diag(cov_daily)

    # Generate independent standard normal random variables
    Z = np.random.normal(0, 1, size=(n_simulations, n_days, n_series))

    # Transform to correlated random variables
    correlated_shocks = np.dot(Z, L.T)

    # Simulate paths
    for t in range(n_days):
        # For hashprice and btc: independent period returns with Itô correction
        # For electricity_residual: AR(1) with correlated innovation
        
        # Generate period returns for hashprice and btc
        period_returns = (mu_daily - 0.5 * sigma_squared_daily) + correlated_shocks[:, t, :]
        paths[:, hashprice_idx, t+1] = period_returns[:, hashprice_idx]
        paths[:, btc_idx, t+1] = period_returns[:, btc_idx]
        
        # For electricity residual: AR(1) process
        # ε_t = φ * ε_{t-1} + σ_ε * innovation_t
        # The innovation is correlated with hashprice/btc shocks via Cholesky
        # The correlated_shock already has the right correlation structure
        # But we need to scale it by sigma_epsilon for the AR(1) innovation
        # Note: The covariance matrix includes electricity_residual, so the correlated shock
        # already accounts for correlation. We just need to use it as the innovation.
        elec_innovation = correlated_shocks[:, t, electricity_idx]
        # Scale by sigma_epsilon to match the AR(1) innovation variance
        # The correlated shock has std dev from covariance, we need to normalize
        elec_std_from_cov = np.sqrt(cov_daily[electricity_idx, electricity_idx])
        normalized_innovation = elec_innovation / elec_std_from_cov * sigma_epsilon
        paths[:, electricity_idx, t+1] = (
            phi * paths[:, electricity_idx, t] + 
            normalized_innovation
        )

    # Reconstruct electricity prices from residuals
    # Calculate dates for seasonal component
    dates = pd.date_range(start=start_date, periods=n_days+1, freq='D')

    # Initialize electricity price paths (log prices)
    electricity_prices = np.zeros((n_simulations, n_days + 1))

    # Get last observed electricity log price
    # last_elec_price = df_data_train['electricity'].iloc[-1]
    electricity_prices[:, 0] = last_elec_log_price

    # Calculate seasonal component and reconstruct prices for each time step
    for t in range(n_days + 1):
        # Calculate days from date_min for seasonal component
        days = (dates[t] - date_min).days
        
        sin_year = np.sin(2 * np.pi * days / T_year)
        cos_year = np.cos(2 * np.pi * days / T_year)
        sin_month = np.sin(2 * np.pi * days / T_month)
        cos_month = np.cos(2 * np.pi * days / T_month)
        
        seasonal = (
            alpha[0, 0] +  # constant
            alpha[1, 0] * sin_year +
            alpha[2, 0] * cos_year +
            alpha[3, 0] * sin_month +
            alpha[4, 0] * cos_month
        )
        
        # Electricity log price = seasonal + residual
        if t == 0:
            # Initial price: use last observed price (should match seasonal + last residual)
            electricity_prices[:, t] = last_elec_log_price
        else:
            # Update: seasonal component + AR(1) residual
            electricity_prices[:, t] = seasonal + paths[:, electricity_idx, t]
    
    return {
            'hashprice': paths[:, hashprice_idx, :],
            'btc': paths[:, btc_idx, :],
            'electricity': electricity_prices
        }

def convert_to_price(simulations, df_data_train):
    """
    Converts simulated log returns (or log prices) back to price levels.

    Parameters:
    simulations (dict): Dictionary of simulated paths (returns or log prices).
    df_data_train (pd.DataFrame): Training data to get the last observed prices.

    Returns:
    dict: Dictionary of simulated price paths.
    """
    # Convert simulated returns back to price levels
    # Since we used log returns: log(P_t) - log(P_{t-1}) = log(P_t / P_{t-1})
    # To get prices: P_t = P_{t-1} * exp(return_t)
    price_simulations = {}

    for key in simulations.keys():
        if key == 'electricity':
            price_paths = np.exp(simulations[key])
            price_simulations[key] = price_paths
        else:
            # Get the last actual price from training data
            last_price = df_data_train[key].iloc[-1]
            
            # Get simulated returns paths
            returns_paths = simulations[key]
            
            # Convert to price paths
            # Start with last price, then apply cumulative returns
            price_paths = np.zeros_like(returns_paths)
            price_paths[:, 0] = last_price
            
            for t in range(1, returns_paths.shape[1]):
                # P_t = P_{t-1} * exp(return_t)
                price_paths[:, t] = price_paths[:, t-1] * np.exp(returns_paths[:, t])
            
            price_simulations[key] = price_paths
    return price_simulations
