# Stat Arb Engine

A statistical arbitrage trading model that scans the S&P 500 for mean-reverting pairs using regression analyses, Engle-Granger tests, z-score signals, R² filters, and cointegration.  

**Current Stage:**
Data validation - ✅  
Creating Database of 26972 Pairs (Pair permutations of same-sector stocks under S&P500) - Done ✅

Currently: Backtesting Model on 10 years of data, 26972. One backtest considers different regimes.

**Next Step:**
Go through trades in Excel, create pivot tables and optimize strategy  
Grid search if required  
Finalize Strategy (Kalman Filter)  
Create dashboard of pairs to monitor - close to reaching our trade thresholds  
Integrate Alpaca API, collecting data continuously on pairs to be monitored  
Thresholds are met - execute trades, log data  
Monitor PnL, pipeline trade data to live dashboard  
