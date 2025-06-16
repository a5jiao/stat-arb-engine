# Stat Arb Engine

A statistical arbitrage trading model that scans the S&P 500 for mean-reverting pairs using ADF tests, z-score signals, R² filters, and cointegration. Built for backtesting and potential live deployment.  

**Current Stage:**
Data validation - ✅  
Creating Database of 26972 Pairs (Pair permutations of same-sector stocks under S&P500) - Currently Running!  

**Next Step:**
Backtest Model on 10 years of data, 26972 pairs  
Go through trades in Excel, create pivot tables and optimize strategy  
Grid search if required  
Finalize Strategy (Kalman Filter)  
Create dashboard of pairs to monitor - close to reaching our trade thresholds  
Integrate Alpaca API, collecting data continuously on pairs to be monitored  
Thresholds are met - execute trades, log data  
Monitor PnL, pipeline trade data to live dashboard  
