import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_distribution(x_values, data, title, ylabel, xlabel='Years', 
                     show_min_line=False, min_data=None, min_label='Minimum',
                     percentiles=(5, 95)):
    """
    Plots the distribution of simulation results including mean and confidence interval.

    Parameters:
    x_values (array-like): X-axis values (e.g., years).
    data (pd.DataFrame or np.array): Simulation data (n_simulations, n_time_steps) or DataFrame with columns as time steps.
    title (str): Plot title.
    ylabel (str): Y-axis label.
    xlabel (str): X-axis label.
    show_min_line (bool): Whether to show a line for minimum values (or another comparison line).
    min_data (array-like): Data for the minimum line if show_min_line is True.
    min_label (str): Label for the minimum line.
    percentiles (tuple): Percentiles for the confidence interval (lower, upper). Default (5, 95).
    """
    
    # Ensure data is consistent format (simulations x time_steps)
    if isinstance(data, pd.DataFrame):
        # If DataFrame columns are time steps, we transpose if needed or just use values
        # Usually simulation results in dataframes here have columns as dates
        values = data.values
        # If the input is transposed (time_steps x simulations), we transpose back
        if values.shape[0] == len(x_values) and values.shape[1] > len(x_values):
             values = values.T
    else:
        values = data

    # Calculate statistics
    # Assuming values shape is (n_simulations, n_time_steps)
    # If not, transpose
    if values.shape[1] != len(x_values):
        values = values.T

    mean_values = np.mean(values, axis=0)
    p_lower = np.percentile(values, percentiles[0], axis=0)
    p_upper = np.percentile(values, percentiles[1], axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot shaded area (Confidence Interval)
    ax.fill_between(x_values, p_lower, p_upper, color='#e6ffe6', alpha=1.0, 
                    label=f'{percentiles[1]-percentiles[0]}% Percentile', edgecolor='black', linewidth=0.5)

    # Plot Mean
    ax.plot(x_values, mean_values, color='blue', linewidth=2, label=f'Mean {ylabel}' if 'NPV' in ylabel else 'Mean')

    # Plot Min line if requested
    if show_min_line and min_data is not None:
         ax.plot(x_values, min_data, color='yellow', linewidth=3, label=min_label)

    # Styling
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='black')
    ax.grid(False) # The reference plots don't show grid lines prominently
    
    # Adjust margins to look like the reference
    ax.margins(x=0) 
    
    return fig, ax
