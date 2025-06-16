import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import pandas as pd
import numpy as np
from itertools import permutations
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
import sqlite3
import os
import openpyxl
from concurrent.futures import as_completed, ThreadPoolExecutor
import threading

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


PRICES = pd.read_csv('data/bbg_tot_ret.csv', index_col=0)
PRICES.index = pd.to_datetime(PRICES.index)
MKT_CAP = pd.read_csv('data/cur_mkt_cap.csv', index_col=0)
MKT_CAP.index = pd.to_datetime(MKT_CAP.index)
TURNOVER = pd.read_csv('data/turnover.csv', index_col=0)
TURNOVER.index = pd.to_datetime(TURNOVER.index)

EARNINGS = pd.read_csv('data/earnings_dates.csv')
# Convert all earnings dates to datetime, handling NaN values
for col in EARNINGS.columns:
    EARNINGS[col] = pd.to_datetime(EARNINGS[col], errors='coerce')

SECTORS = pd.read_csv('/Users/aj/stat-arb-engine/data/sp500_sectors.csv')
SECTOR_MAP = dict(zip(SECTORS['ticker'], SECTORS['sector']))

def load_tickers(filepath="data/sp500_tickers.csv"):
    with open(filepath) as f:
        return [line.strip() for line in f if line.strip()]

TICKS = load_tickers()

def get_next_earnings_date(ticker, current_date):
    
    if ticker not in EARNINGS.columns:
        return pd.NaT
    
    earnings_dates = EARNINGS[ticker].dropna()
    
    future_earnings = earnings_dates[earnings_dates > current_date]
    
    if len(future_earnings) > 0:
        return future_earnings.min()  # Return the next (earliest) earnings date
    else:
        return pd.NaT

def preprocess_earnings_data():

    # Pre-process earnings data into sorted arrays for fast lookup
    # Only needs to be run once at startup

    earnings_lookup = {}
    for ticker in EARNINGS.columns:
        # Get all earnings dates for this ticker, sorted
        earnings_dates = EARNINGS[ticker].dropna().sort_values()
        earnings_lookup[ticker] = earnings_dates.values  # Convert to numpy array for speed
    return earnings_lookup

def get_next_earnings_fast(ticker, current_date, earnings_lookup):

    # Fast earnings lookup using binary search

    if ticker not in earnings_lookup:
        return pd.NaT
    
    earnings_array = earnings_lookup[ticker]
    if len(earnings_array) == 0:
        return pd.NaT
    
    # Use numpy searchsorted for fast binary search
    idx = np.searchsorted(earnings_array, current_date, side='right')
    
    if idx < len(earnings_array):
        return pd.Timestamp(earnings_array[idx])
    else:
        return pd.NaT

def add_earnings_vectorized(df, stock1, stock2, earnings_lookup):
    # Fixed vectorized earnings - no more float error!
    df['nextErn1'] = pd.NaT
    df['nextErn2'] = pd.NaT
    
    if stock1 in earnings_lookup:
        earnings1 = pd.to_datetime(earnings_lookup[stock1])  # Convert to proper datetime
        for i, current_date in enumerate(df.index):
            future_earnings = earnings1[earnings1 > current_date]
            if len(future_earnings) > 0:
                df.iloc[i, df.columns.get_loc('nextErn1')] = future_earnings.min()
  
    if stock2 in earnings_lookup:
        earnings2 = pd.to_datetime(earnings_lookup[stock2])  # Convert to proper datetime
        for i, current_date in enumerate(df.index):
            future_earnings = earnings2[earnings2 > current_date]
            if len(future_earnings) > 0:
                df.iloc[i, df.columns.get_loc('nextErn2')] = future_earnings.min()



def generate_same_sector_pairs():
    # Group tickers by sector
    sector_groups = {}
    for ticker in TICKS:
        sector = SECTOR_MAP.get(ticker, 'Unknown')
        if sector != 'Unknown':
            if sector not in sector_groups:
                sector_groups[sector] = []
            sector_groups[sector].append(ticker)
    
    # Generate pairs within each sector
    same_sector_pairs = []
    for sector, tickers in sector_groups.items():
        if len(tickers) >= 2:  # Need at least 2 stocks for pairs
            # Use permutations for both directions since cointegration is directional
            sector_pairs = list(permutations(tickers, 2))
            same_sector_pairs.extend(sector_pairs)
    
    return same_sector_pairs

def z_score(curr, mean, std):
    if std == 0:
        return 0
    return (curr-mean)/std

def linear_regression(series1, series2):
    X = series1
    y = series2
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    results = model.fit()
    y_pred = results.predict(X)
    rsq = results.rsquared
    intercept, slope = results.params 
    return y_pred, intercept, slope, rsq


def get_coint(series1, series2): # Engle-Granger test, good for regression based relationship
    try:
        _, pvalue, _ = coint(series1, series2)
        return pvalue
    except:
        return np.nan

def create_df(series1, series2, stock1, stock2, earnings_lookup, look_back=60):
    common_dates = series1.index.intersection(series2.index)
    series1 = series1.loc[common_dates]
    series2 = series2.loc[common_dates]
    
    # Skip if insufficient data
    if len(series1) < look_back + 30: # buffer for calculations
        return pd.DataFrame()
    
    df = pd.DataFrame(index=common_dates)
        
    df[f'{stock1}_price'] = series1
    df[f'{stock2}_price'] = series2
    df[f'{stock1}_sector'] = SECTOR_MAP.get(stock1, 'Unknown')
    df[f'{stock2}_sector'] = SECTOR_MAP.get(stock2, 'Unknown')
    df[f'{stock1}_ln_price'] = np.log(series1)
    df[f'{stock2}_ln_price'] = np.log(series2)
    add_earnings_vectorized(df, stock1, stock2, earnings_lookup)
    df['count'] = range(1, len(df) + 1)

    # Initialize rolling calculation columns
    df['coint_p_value'] = np.nan
    df['slope'] = np.nan
    df['y_intercept'] = np.nan
    df['r_squared'] = np.nan
    df['y_implied'] = np.nan
    df['curr_residual'] = np.nan
    df['z_residual'] = np.nan # Z-score of RESIDUAL, NOT PRICE

    df['ratio'] = series1/series2
    df['logRatio'] = np.log(series1)/np.log(series2)
    df['avg_ratio'] = df['logRatio'].shift(1).rolling(window=60, min_periods=60).mean() # 60 day avg
    df['std_dev'] = np.nan
    df['z_ratio'] = np.nan # Z-score of PRICE ratio
    df['Mkt_cap1'] = np.nan
    df['Mkt_cap2'] = np.nan
    df['30D_Turnover1'] = np.nan
    df['30D_Turnover2'] = np.nan
    # df['nextErn1'] = pd.NaT
    # df['nextErn2'] = pd.NaT

    

    for date in df.index:
        # df.loc[date, 'nextErn1'] = get_next_earnings_date(stock1, date)
        # df.loc[date, 'nextErn2'] = get_next_earnings_date(stock2, date)
        try:
            if date in MKT_CAP.index and stock1 in MKT_CAP.columns:
                df.loc[date, 'Mkt_cap1'] = MKT_CAP.loc[date, stock1]
            if date in MKT_CAP.index and stock2 in MKT_CAP.columns:
                df.loc[date, 'Mkt_cap2'] = MKT_CAP.loc[date, stock2]
            if date in TURNOVER.index and stock1 in TURNOVER.columns:
                df.loc[date, '30D_Turnover1'] = TURNOVER.loc[date, stock1]
            if date in TURNOVER.index and stock2 in TURNOVER.columns:
                df.loc[date, '30D_Turnover2'] = TURNOVER.loc[date, stock2]
        except (KeyError, IndexError):
            pass  # Skip if data missing

    for i in range(look_back, len(df)):
        start_idx = i - look_back
        end_idx = i

        look_back_ln1 = df[f'{stock1}_ln_price'].iloc[start_idx:end_idx]
        look_back_ln2 = df[f'{stock2}_ln_price'].iloc[start_idx:end_idx]
        df.iloc[i,df.columns.get_loc('coint_p_value')] = get_coint(look_back_ln1, look_back_ln2)
        try:
            y_pred, y_int, slope, rsq = linear_regression(look_back_ln1, look_back_ln2)
            df.iloc[i,df.columns.get_loc('slope')] = slope
            df.iloc[i,df.columns.get_loc('y_intercept')] = y_int
            df.iloc[i,df.columns.get_loc('r_squared')] = rsq
            y_imp = slope*df[f'{stock1}_ln_price'].iloc[i] + y_int
            df.iloc[i,df.columns.get_loc('y_implied')] = y_imp
            resid = df[f'{stock2}_ln_price'].iloc[i]-y_imp
            df.iloc[i,df.columns.get_loc('curr_residual')] = resid
            past60_resid = look_back_ln2 - y_pred
            df.iloc[i,df.columns.get_loc('z_residual')] = z_score(resid, past60_resid.mean(), past60_resid.std())
        except: 
            continue
        lookback_lnratio = df['logRatio'].iloc[start_idx:end_idx]
        df.iloc[i,df.columns.get_loc('std_dev')] = lookback_lnratio.std()
        df.iloc[i,df.columns.get_loc('z_ratio')] = z_score(df['logRatio'][i], lookback_lnratio.mean(), lookback_lnratio.std())

    return df


if __name__ == "__main__":
    look_back = 60
    pairs = generate_same_sector_pairs()
    processed_count = 0
    
    print("Pre-processing earnings data...")
    earnings_lookup = preprocess_earnings_data()
    print(f"Earnings data ready for {len(earnings_lookup)} tickers")

    print(f"Processing {len(pairs)} same-sector pairs...")

    for pair in tqdm(pairs, desc="Processing pairs"):  # Remove [:5] for full run
        stock1 = pair[0]
        stock2 = pair[1]
        try:
            series1 = PRICES[stock1].dropna()
            series2 = PRICES[stock2].dropna()

            df = create_df(series1, series2, stock1, stock2, earnings_lookup)

            if not df.empty:
                # Use connection context manager for safety
                with sqlite3.connect('data/pairs_database.db') as conn:
                    table_name = f'pair_{stock1}_{stock2}'
                    df.to_sql(table_name, conn, if_exists='replace', index=True)
                processed_count += 1
                
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} pairs successfully")

        except Exception as e:
            print(f'Error processing pair {pair}: {str(e)}')
            continue
    
    print(f"Total pairs processed successfully: {processed_count}")
    print("Database saved as: data/pairs_database.db")

    #         if not df.empty:
    #             # Save to CSV file
    #             filename = f'{stock1}_{stock2}_60days.xlsx'
    #             filepath = os.path.join('/Users/aj/stat-arb-engine/data/pairs', filename)
    #             df.to_excel(filepath, index=True)
    #             processed_count += 1
                
    #             if processed_count % 100 == 0:
    #                 print(f"Processed {processed_count} pairs successfully")

    #     except Exception as e:
    #         print(f'Error processing pair {pair}: {str(e)}')
    
    # print(f"Total pairs processed successfully: {processed_count}")
    # print(f"CSV files saved to: {'/Users/aj/stat-arb-engine/data/pairs'}")

