import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from src.export import export_figure

# --- Professional Style Configuration ---
# Call this function at the start of your notebook
def set_professional_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif',           # Serif fonts (Times/Palatino) are standard for papers
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.spines.top': False,         # Remove top border
        'axes.spines.right': False,       # Remove right border
        'axes.grid': True,
        'grid.color': '#E6E6E6',          # Lighter grid
        'grid.linestyle': '--',
        'legend.frameon': False,          # Clean legend
        'figure.figsize': (10, 6)
    })

def plot_distribution(x_values, data, title, ylabel, xlabel='Days Forecast', 
                     show_min_line=False, min_data=None, min_label='Minimum',
                     percentiles=(5, 95), ax=None, is_return=False):
    """
    Plots simulation distribution with professional formatting.
    
    Parameters:
    -----------
    is_return : bool
        If True, formats Y-axis as percentage (e.g., 5.0%).
        If False, formats Y-axis with thousands separators (e.g., 1,000).
    """
    
    # Data Handling
    if isinstance(data, pd.DataFrame):
        values = data.values
        if values.shape[0] == len(x_values) and values.shape[1] > len(x_values):
             values = values.T
    else:
        values = data

    if values.shape[1] != len(x_values):
        values = values.T

    # Statistics
    mean_values = np.mean(values, axis=0)
    p_lower = np.percentile(values, percentiles[0], axis=0)
    p_upper = np.percentile(values, percentiles[1], axis=0)

    # Axis Handling
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # Plot Fan Chart
    # Use a professional grey/blue palette instead of bright green
    ax.fill_between(x_values, p_lower, p_upper, color='#4A90E2', alpha=0.15, 
                    label=f'{percentiles[1]-percentiles[0]}% Confidence', edgecolor=None)

    ax.plot(x_values, mean_values, color='#003366', linewidth=1.5, 
            label=f'Mean {ylabel}' if 'NPV' in ylabel else 'Mean')

    if show_min_line and min_data is not None:
         ax.plot(x_values, min_data, color='#D0021B', linewidth=2, linestyle='--', label=min_label)

    # Formatting
    ax.set_title(title, pad=15)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    
    # --- The "Professional" Axis Formatting ---
    if is_return:
        # Format as Percentage (e.g., 0.05 -> 5%)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=1))
    else:
        # Format with Commas (e.g., 10000 -> 10,000)
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
        
    ax.xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))

    ax.legend(loc='upper left')
    ax.margins(x=0) 
    
    return fig, ax

def plot_prices(df_train, assets_type='bitcoin'):
    # A. Dual-Axis Plot (BTC vs Hashprice)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    bitcoin_pm = {'color': 'tab:orange', 'label': 'Bitcoin Price'}
    electricity_pm = {'color': 'tab:blue', 'label': 'Electricity Price'}
    
    if assets_type == 'bitcoin':
        asset_1_pm = {'label': 'Bitcoin Price', 'unit': '($)', 'asset': 'btc'}
        asset_2_pm = {'label': 'Hashprice', 'unit': '($/PH/Day)', 'asset': 'hashprice'}
    elif assets_type == 'electricity':
        asset_1_pm = {'label': 'Electricity Price', 'unit': '($/MWh)', 'asset': 'electricity'}
        asset_2_pm = {'label': 'Electricity Price', 'unit': '($/MWh)', 'asset': 'electricity'}
    else:
        raise ValueError('assets_type must be either "bitcoin" or "electricity"')
    
    assets_pm = {'asset_1': asset_1_pm, 'asset_2': asset_2_pm}

    # Plot Bitcoin (Left Axis)
    color = 'tab:orange'
    ax1.set_xlabel('Date')
    ax1.set_ylabel(f'{asset_1_pm["label"]} {asset_1_pm["unit"]}', color=color)
    print(df_train.columns)
    print(asset_1_pm['asset'] in df_train.columns)
    ax1.plot(df_train.index, df_train[asset_1_pm['asset']], color=color, label=asset_1_pm['label'])
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Plot Hashprice (Right Axis)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel(f'{asset_2_pm["label"]} {asset_2_pm["unit"]}', color=color)
    ax2.plot(df_train.index, df_train[asset_2_pm['asset']], color=color, label=asset_2_pm['label'])
    ax2.tick_params(axis='y', labelcolor=color)

    # Title & Layout
    plt.title(f'Historical Dynamics: {asset_1_pm["label"]} vs. {asset_2_pm["label"]}')
    fig.tight_layout()
    export_figure(fig, f"historical_{assets_type}_dual")

def plot_project_value(possible_exits, pv_exit_curve, max_project_value, optimal_exit_month):
    # 4. Visualization
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(possible_exits, pv_exit_curve, label='Project NPV (Ops + Resale)', color='green', linewidth=2)
    ax.axhline(0, color='black', linestyle='--', linewidth=1, label='Breakeven')

    # Highlight Optimum
    if max_project_value > -np.inf: # Check validity
        ax.scatter(optimal_exit_month, max_project_value, color='red', zorder=5)
        ax.annotate(f'Optimal Exit: Month {optimal_exit_month}\nNPV: ${max_project_value:,.0f}', 
                    xy=(optimal_exit_month, max_project_value), 
                    xytext=(optimal_exit_month+2, max_project_value),
                    arrowprops=dict(facecolor='black', shrink=0.05))

    ax.set_title("Deterministic Project Value by Exit Timing")
    ax.set_xlabel("Exit Month (T)")
    ax.set_ylabel("Net Present Value ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    export_figure(fig, "pv_exit_curve_static")
