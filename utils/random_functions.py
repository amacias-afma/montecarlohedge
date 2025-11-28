import numpy as np

def multivariable_random_walk(start_prices, drift, covariance_matrix, num_steps, num_simulations):
    """
    Generates price paths for multiple variables using a multivariable random walk.

    Args:
        start_prices (np.ndarray): Array of initial prices for each variable.
        drift (np.ndarray): Array of drift values (mean of daily returns) for each variable.
        covariance_matrix (np.ndarray): Covariance matrix of the daily returns.
        num_steps (int): The number of time steps (e.g., days) to simulate.
        num_simulations (int): The number of independent random walk paths to generate.

    Returns:
        np.ndarray: A 3D array of shape (num_simulations, num_steps + 1, num_variables)
                    containing the simulated price paths. The first dimension represents
                    each simulation, the second dimension represents the time step,
                    and the third dimension represents the variable.
    """
    num_variables = len(start_prices)
    simulated_paths = np.zeros((num_simulations, num_steps + 1, num_variables))
    simulated_paths[:, 0, :] = start_prices

    # Generate random daily shocks using the multivariate normal distribution
    # We generate (num_simulations * num_steps) samples and then reshape
    # for efficiency compared to generating within the loop.
    daily_shocks = np.random.multivariate_normal(np.zeros(num_variables), covariance_matrix, size=(num_simulations * num_steps))
    daily_shocks = daily_shocks.reshape((num_simulations, num_steps, num_variables))

    for t in range(num_steps):
        # The daily return includes a drift term and a stochastic term
        daily_returns = drift + daily_shocks[:, t, :]

        # Update prices
        simulated_paths[:, t+1, :] = simulated_paths[:, t, :] * np.exp(daily_returns)

    return simulated_paths
