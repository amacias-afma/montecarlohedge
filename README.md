# Monte Carlo Hedge Simulation

## Overview
This project provides a Python-based framework for simulating financial asset prices and evaluating hedging strategies using Monte Carlo methods. It is designed to analyze the effectiveness of hedging (e.g., Delta hedging) under various market conditions and stochastic processes.

The core functionality includes:
- **Monte Carlo Simulations:** Generating multiple future price paths for underlying assets.
- **Option Pricing:** Valuation of options using simulated paths.
- **Hedging Strategies:** Simulating and testing the performance of dynamic hedging strategies over time.

## Project Structure

The project is organized as follows:

```text
├── main.ipynb                  # Main demonstration notebook showing usage and results
├── Electricity data ERCOT.csv  # Dataset (e.g., electricity prices) used for analysis
├── utils/                      # Helper modules
│   ├── g_functions.py          # Functions for calculating Option Greeks (Delta, Gamma, etc.)
│   ├── options.py              # Classes defining Option contracts (Call, Put, etc.)
│   ├── read_data.py            # Utilities for loading and preprocessing data
│   └── simulations.py          # Core Monte Carlo simulation algorithms (e.g., GBM, MRJD)
├── olds/                       # Archive of previous notebook versions and experiments
└── images/                     # Generated plots and visualizations
