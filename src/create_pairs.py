import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import pandas as pd
import numpy as np
from itertools import permutations
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
import sqlite3
import os

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


PRICES = pd.read_csv('data/historical_prices.csv', index_col=0)
PRICES.index = pd.to_datetime(PRICES.index)
SECTORS = pd.read_csv('/Users/aj/stat-arb-engine/data/sp500_sectors.csv')
SECTOR_MAP = dict(zip(SECTORS['ticker'], SECTORS['sector']))

def load_tickers(filepath="data/sp500_tickers.csv"):
    with open(filepath) as f:
        return [line.strip() for line in f if line.strip()]

TICKS = load_tickers()

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

def create_df(series1, series2, stock1, stock2, look_back=60):
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
    df['avg_ratio'] = df['ratio'].rolling(window=60).mean() # 60 day avg
    df['std_dev'] = np.nan
    df['z_ratio'] = np.nan # Z-score of PRICE ratio

    for i in range(look_back, len(df)):
        start_idx = i - look_back
        end_idx = i + 1

        look_back_ln1 = df[f'{stock1}_ln_price'].iloc[start_idx:end_idx]
        look_back_ln2 = df[f'{stock2}_ln_price'].iloc[start_idx:end_idx]
        df.iloc[i,df.columns.get_loc('coint_p_value')] = get_coint(look_back_ln1, look_back_ln2)
        try:
            y_pred, y_int, slope, rsq = linear_regression(look_back_ln1, look_back_ln2)
            df.iloc[i,df.columns.get_loc('slope')] = slope
            df.iloc[i,df.columns.get_loc('y_intercept')] = y_int
            df.iloc[i,df.columns.get_loc('r_squared')] = rsq
            df.iloc[i,df.columns.get_loc('y_implied')] = y_pred[-1]
            resid = look_back_ln2-y_pred
            df.iloc[i,df.columns.get_loc('curr_residual')] = resid[-1]
            df.iloc[i,df.columns.get_loc('z_residual')] = z_score(resid[-1], resid.mean(), resid.std())
        except: 
            continue
        lookback_lnratio = df['logRatio'].iloc[start_idx:end_idx]
        df.iloc[i,df.columns.get_loc('std_dev')] = lookback_lnratio.std()
        df.iloc[i,df.columns.get_loc('z_ratio')] = z_score(lookback_lnratio[-1], lookback_lnratio.mean(), lookback_lnratio.std())

    return df


if __name__ == "__main__":
    look_back = 60
    pairs = list(permutations(TICKS,2))
    processed_count = 0
    
    # Create SQLite database connection
    conn = sqlite3.connect('data/pairs_database.db')
    print("Created database connection")

    output_dir = '/Users/aj/stat-arb-engine/data/pairs'

    for pair in tqdm(pairs[:5], desc="Processing pairs"):
        stock1 = pair[0]
        stock2 = pair[1]
        try:
            series1 = PRICES[stock1].dropna()
            series2 = PRICES[stock2].dropna()

            df = create_df(series1, series2, stock1, stock2)
    #         if not df.empty:
    #             # Save to CSV file
    #             filename = f'{stock1}_{stock2}_60days.csv'
    #             filepath = os.path.join(output_dir, filename)
    #             df.to_csv(filepath, index=True)
    #             processed_count += 1
                
    #             if processed_count % 100 == 0:
    #                 print(f"Processed {processed_count} pairs successfully")

    #     except Exception as e:
    #         print(f'Error processing pair {pair}: {str(e)}')
    
    # print(f"Total pairs processed successfully: {processed_count}")
    # print(f"CSV files saved to: {output_dir}")


            if not df.empty:
                # Save to SQLite database
                table_name = f'pair_{stock1}_{stock2}'
                df.to_sql(table_name, conn, if_exists='replace', index=True)
                processed_count += 1
                
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} pairs successfully")

        except Exception as e:
            print(f'Error processing pair {pair}: {str(e)}')
    
    # Close database connection
    conn.close()
    print(f"Total pairs processed successfully: {processed_count}")
    print("Database saved as: data/pairs_database.db")
