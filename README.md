# Optimal Investment in Bitcoin Mining Farms: A Hedged Monte-Carlo Approach

## Overview

This project implements a **Real Options** framework for evaluating investments in Bitcoin mining facilities. It uses the **Hedged Monte Carlo (HMC)** method to solve the optimal stopping problem of when to invest, considering:

- **Stochastic Bitcoin Hashprice** (correlated GBM)
- **Stochastic Electricity Costs** (seasonal Fourier + AR(1) model)
- **Option to Defer** investment (timing flexibility)
- **Option to Suspend** mining when unprofitable (operational flexibility)

## Project Structure

```
MonteCarloHedge/
├── config.py                    # Centralized paths and configuration
├── requirements.txt             # Python dependencies
│
├── article/                     # LaTeX article
│   ├── article.tex              # Main paper
│   ├── bibli.bib                # General bibliography
│   ├── bitcoin_refs.bib         # Bitcoin-specific references
│   └── figures/                 # Article figures
│
├── src/                         # Python source modules
│   ├── __init__.py
│   ├── read_data.py             # Data loading and preprocessing
│   ├── simulations.py           # Monte Carlo simulation (GBM, AR(1), Cholesky)
│   ├── options.py               # Option pricing (Margrabe, BS, HMC regression)
│   ├── g_functions.py           # NPV estimation (G function)
│   ├── visualization.py         # Distribution and result plots
│   └── export.py                # Results export (figures, tables, LaTeX macros)
│
├── notebooks/                   # Jupyter notebooks
│   └── main.ipynb               # Main analysis notebook
│
├── data/
│   ├── raw/                     # Original CSV data (see data/raw/README.md)
│   └── cache/                   # Generated/cached simulation data
│
├── results/                     # Generated outputs (notebook → article pipeline)
│   ├── figures/                 # Exported plots (PDF/PNG)
│   ├── tables/                  # Exported LaTeX table snippets
│   └── params.json              # Key numerical results
│
└── archive/                     # Old notebooks and experiments
```

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure data path**: Edit `DATA_PATH` in `config.py` or set the environment variable:
   ```bash
   export MCHEDGE_DATA_PATH=/path/to/your/csv/data/
   ```

3. **Run the notebook**:
   ```bash
   jupyter notebook notebooks/main.ipynb
   ```

## Data Requirements

See [`data/raw/README.md`](data/raw/README.md) for details on required CSV files and their sources.

## Results Pipeline

The notebook exports figures, tables, and parameters to `results/`. The LaTeX article includes them via:

```latex
\input{../results/tables/params_macros.tex}       % Key numbers as \newcommand
\input{../results/tables/case_parameters.tex}      % Parameter tables
\includegraphics{../results/figures/boundary.pdf}   % Figures
```

Re-run the notebook → recompile LaTeX → article auto-updates.
