# Data Directory

## Raw Data Files

This project requires the following data files in this directory:

| File | Description | Source |
|------|-------------|--------|
| `bitcoin-hashprice-index-20251110.csv` | Bitcoin Hashprice Index (daily) | [Hashrate Index](https://data.hashrateindex.com/network-data/bitcoin-hashprice-index) |
| `BTC-USD.csv` | BTC/USD daily prices | [Yahoo Finance](https://finance.yahoo.com/quote/BTC-USD/history) |
| `Electricity data ERCOT.csv` | ERCOT day-ahead electricity prices | [EnergyOnline](http://www.energyonline.com/Data/GenericData.aspx?DataId=23) |

## Setup

By default, the project looks for data files at the path configured in `config.py`:

```python
DATA_PATH = os.environ.get("MCHEDGE_DATA_PATH", "C:/Users/fe_ma/Data/")
```

**To use your own data location**, either:
1. Set the environment variable: `export MCHEDGE_DATA_PATH=/your/data/path/`
2. Or edit `DATA_PATH` in `config.py`

Place the CSV files in that directory.
