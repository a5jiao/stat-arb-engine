import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Strategy parameters - BASE PARAMETERS (will be adjusted by regime)
BASE_ENTRY_Z_THRESHOLD = 2.5
BASE_EXIT_Z_THRESHOLD = 0.0
MIN_IMPLIED_RETURN = 0.03  # 3%
BASE_RSQ_THRESHOLD = 0.8
BASE_COINT_THRESHOLD = 0.005
EARNINGS_BUFFER_DAYS = 20
EARNINGS_EXIT_DAYS = 7
EARNINGS_RECENCY_DAYS = 90

POSITION_SIZE = 500

# *** NEW: Regime detection parameters ***
USE_REGIME_DETECTION = True  # Set to False for baseline comparison
VIX_LOW_THRESHOLD = 20
VIX_HIGH_THRESHOLD = 30

def load_vix_data(vix_path='data/vix_data.csv'):
    # Load VIX data for regime detection
    try:
        vix_df = pd.read_csv(vix_path, index_col=0, parse_dates=True)
        vix_df.columns = ['VIX']  # Ensure column name is 'VIX'
        return vix_df
    except Exception as e:
        print(f"Warning: Could not load VIX data from {vix_path}: {e}")
        print("Proceeding without regime detection...")
        return pd.DataFrame()

def get_regime_thresholds(date, vix_data):
    # Get regime-adjusted thresholds based on VIX level
    if vix_data.empty or not USE_REGIME_DETECTION:
        # Return base parameters if no VIX data or regime detection disabled
        return {
            'entry_z': BASE_ENTRY_Z_THRESHOLD,
            'exit_z': BASE_EXIT_Z_THRESHOLD,
            'rsq_threshold': BASE_RSQ_THRESHOLD,
            'coint_threshold': BASE_COINT_THRESHOLD,
            'regime': 'baseline'
        }
    
    try:
        # Get VIX level for this date (or closest available)
        if date in vix_data.index:
            vix_level = vix_data.loc[date, 'VIX']
        else:
            # Find closest date
            closest_date = vix_data.index[vix_data.index <= date][-1] if len(vix_data.index[vix_data.index <= date]) > 0 else vix_data.index[0]
            vix_level = vix_data.loc[closest_date, 'VIX']
        
        # Determine regime and adjust thresholds
        if vix_level < VIX_LOW_THRESHOLD:  # Low volatility regime
            return {
                'entry_z': 2.0,      # Tighter entry (was 2.5)
                'exit_z': 0.2,       # Tighter exit (was 0.0)
                'rsq_threshold': 0.85,  # Higher R² requirement
                'coint_threshold': 0.003,  # Stricter cointegration
                'regime': 'low_vol',
                'vix_level': vix_level
            }
        elif vix_level > VIX_HIGH_THRESHOLD:  # High volatility regime
            return {
                'entry_z': 3.0,      # Wider entry (was 2.5)
                'exit_z': 0.5,       # Wider exit (was 0.0)
                'rsq_threshold': 0.75,  # Lower R² tolerance
                'coint_threshold': 0.01,   # Relaxed cointegration
                'regime': 'high_vol',
                'vix_level': vix_level
            }
        else:  # Medium volatility regime
            return {
                'entry_z': BASE_ENTRY_Z_THRESHOLD,
                'exit_z': BASE_EXIT_Z_THRESHOLD,
                'rsq_threshold': BASE_RSQ_THRESHOLD,
                'coint_threshold': BASE_COINT_THRESHOLD,
                'regime': 'medium_vol',
                'vix_level': vix_level
            }
    except Exception as e:
        print(f"Error getting regime for {date}: {e}")
        # Fallback to base parameters
        return {
            'entry_z': BASE_ENTRY_Z_THRESHOLD,
            'exit_z': BASE_EXIT_Z_THRESHOLD,
            'rsq_threshold': BASE_RSQ_THRESHOLD,
            'coint_threshold': BASE_COINT_THRESHOLD,
            'regime': 'error_fallback'
        }


def get_all_pairs(db_path='data/pairs_database.db'):
    # GEt list of all pairs in database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'pair_%'")
    pairs = [row[0] for row in cursor.fetchall()]
    conn.close()
    return pairs

def load_pair_data(pair_name, db_path='data/pairs_database.db'):
    # Load and prepare data for a specific pair
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql(
            f'SELECT * FROM {pair_name}', 
            conn, 
            index_col='index', 
            parse_dates=['index']
        )
        conn.close()
        
        # Ensure data is sorted by date
        df = df.sort_index()
        
        # Get symbols from pair name
        symbols = pair_name.replace('pair_', '').split('_')
        if len(symbols) != 2:
            return pd.DataFrame(), None, None
        
        symbol1, symbol2 = symbols[0], symbols[1]
        
        # Calculate implied return using your column names
        df['y_implied_price'] = np.exp(df['y_implied'])
        df['implied_return'] = abs(df[f'{symbol2}_price'] - df['y_implied_price']) / df[f'{symbol2}_price']
        
        # Calculate days to earnings
        df['days_to_earnings1'] = calculate_days_to_earnings(df, 'nextErn1')
        df['days_to_earnings2'] = calculate_days_to_earnings(df, 'nextErn2')
        
        return df, symbol1, symbol2
        
    except Exception as e:
        conn.close()
        print(f"Error loading {pair_name}: {e}")
        return pd.DataFrame(), None, None

def calculate_days_to_earnings(df, earnings_col):
    days_to_earnings = pd.Series(index=df.index, dtype=float)
    
    for date, row in df.iterrows():
        if pd.notna(row[earnings_col]) and row[earnings_col] != 'None':
            try:
                earnings_date = pd.to_datetime(row[earnings_col])
                days_diff = (earnings_date - date).days
                days_to_earnings.loc[date] = days_diff
            except:
                days_to_earnings.loc[date] = np.nan
        else:
            days_to_earnings.loc[date] = np.nan
            
    return days_to_earnings

def check_earnings_filter(row, trade_type='entry'):
    # Check if trade should be filtered due to earnings
    days1 = row['days_to_earnings1']
    days2 = row['days_to_earnings2']
    
    if trade_type == 'entry':
        for days in [days1, days2]:
            if pd.notna(days):
                # Block if earnings within next 20 days
                if 0 <= days <= EARNINGS_BUFFER_DAYS:
                    return True
                # Block if earnings were within past 90 days
                if -EARNINGS_RECENCY_DAYS <= days < 0:
                    return True
                # Block if earnings are more than 90 days away
                if days > EARNINGS_RECENCY_DAYS:
                    return True
        # Only allow trading when earnings are 21-89 days away for both stocks
    
    elif trade_type == 'exit':
        # Force exit if earnings within 7 days
        for days in [days1, days2]:
            if pd.notna(days) and 0 <= days <= EARNINGS_EXIT_DAYS:
                return True
                
    return False

def generate_signals(df, vix_data):
    # Generate entry and exit signals for a pair
    signals = pd.DataFrame(index=df.index)
    
    # Entry conditions
    for date in df.index:
        # Get regime-specific thresholds for this date
        thresholds = get_regime_thresholds(date, vix_data)
        
        row = df.loc[date]
        
        # Entry conditions with regime-adjusted thresholds
        long_condition = (
            (row['z_residual'] <= -thresholds['entry_z']) &  # Y underpriced
            (row['implied_return'] >= MIN_IMPLIED_RETURN) &
            (row['r_squared'] >= thresholds['rsq_threshold']) &
            (row['coint_p_value'] <= thresholds['coint_threshold']) &
            (not check_earnings_filter(row, 'entry'))
        )
        
        short_condition = (
            (row['z_residual'] >= thresholds['entry_z']) &   # Y overpriced
            (row['implied_return'] >= MIN_IMPLIED_RETURN) &
            (row['r_squared'] >= thresholds['rsq_threshold']) &
            (row['coint_p_value'] <= thresholds['coint_threshold']) &
            (not check_earnings_filter(row, 'entry'))
        )
        
        earnings_exit = check_earnings_filter(row, 'exit')
        
        signals.loc[date, 'entry_long'] = long_condition
        signals.loc[date, 'entry_short'] = short_condition
        signals.loc[date, 'exit_earnings'] = earnings_exit
        signals.loc[date, 'regime'] = thresholds['regime']
        signals.loc[date, 'entry_z_threshold'] = thresholds['entry_z']
        signals.loc[date, 'exit_z_threshold'] = thresholds['exit_z']
        if 'vix_level' in thresholds:
            signals.loc[date, 'vix_level'] = thresholds['vix_level']
    
    return signals

def open_position(row, date, direction, symbol1, symbol2, regime_info):
    # Create a new position record
    entry_x_price = row[f'{symbol1}_price']
    entry_y_price = row[f'{symbol2}_price']
    
    # Calculate shares for position in each stock
    x_shares = POSITION_SIZE / entry_x_price
    y_shares = POSITION_SIZE / entry_y_price

    return {
        'entry_date': date,
        'direction': direction,
        'entry_z_score': row['z_residual'],
        'entry_implied_return': row['implied_return'],
        'entry_coint': row['coint_p_value'],
        'entry_x_price': row[f'{symbol1}_price'],
        'entry_y_price': row[f'{symbol2}_price'],
        'entry_y_implied': row['y_implied'],
        'x_shares': x_shares,
        'y_shares': y_shares,
        'position_size': POSITION_SIZE,
        'slope': row['slope'],
        'r_squared': row['r_squared'],
        'symbol1': symbol1,
        'symbol2': symbol2,
        'sector': row[f'{symbol1}_sector'],
        'entry_market_cap1': row.get('Mkt_cap1', np.nan),
        'entry_market_cap2': row.get('Mkt_cap2', np.nan),
        'days_to_earnings1': row['days_to_earnings1'],
        'days_to_earnings2': row['days_to_earnings2'],
        # *** NEW: Regime information ***
        'entry_regime': regime_info.get('regime', 'unknown'),
        'entry_vix_level': regime_info.get('vix_level', np.nan),
        'entry_z_threshold_used': regime_info.get('entry_z', BASE_ENTRY_Z_THRESHOLD),
        'exit_z_threshold_used': regime_info.get('exit_z', BASE_EXIT_Z_THRESHOLD)
    }

def close_position(position, row, date, exit_reason, symbol1, symbol2):
    # Close a position and calculate P&L
    holding_period = (date - position['entry_date']).days
    
    exit_x_price = row[f'{symbol1}_price']
    exit_y_price = row[f'{symbol2}_price']

    x_shares = position['x_shares']
    y_shares = position['y_shares']
    
    # Calculate P&L based on direction
    if position['direction'] == 'long':
        # Long Y, Short X
        y_dollar_pnl = y_shares * (exit_y_price - position['entry_y_price'])
        x_dollar_pnl = x_shares * (position['entry_x_price'] - exit_x_price)
    else:  # short
        # Short Y, Long X
        y_dollar_pnl = y_shares * (position['entry_y_price'] - exit_y_price)
        x_dollar_pnl = x_shares * (exit_x_price - position['entry_x_price'])
    
    
    # Weight X P&L by hedge ratio (slope)
    # total_pnl = y_pnl + (position['slope'] * x_pnl)

    # DOllar neutral
    total_dollar_pnl = y_dollar_pnl + x_dollar_pnl
    return_pct = total_dollar_pnl / (2 * POSITION_SIZE)
    
    trade = {
        # Entry information
        'pair': f"{symbol1}_{symbol2}",
        'symbol1': symbol1,
        'symbol2': symbol2,
        'sector': position['sector'],
        'entry_date': position['entry_date'],
        'exit_date': date,
        'entry_z_score': position['entry_z_score'],
        'entry_coint': position['entry_coint'],
        'entry_implied_return': position['entry_implied_return'],
        'entry_x_price': position['entry_x_price'],
        'entry_y_price': position['entry_y_price'],
        'entry_y_implied': position['entry_y_implied'],
        'slope': position['slope'],
        'r_squared': position['r_squared'],
        'direction': position['direction'],
        'entry_market_cap1': position['entry_market_cap1'],
        'entry_market_cap2': position['entry_market_cap2'],
        'entry_days_to_earnings1': position['days_to_earnings1'],
        'entry_days_to_earnings2': position['days_to_earnings2'],
        'entry_regime': position['entry_regime'],
        'entry_vix_level': position['entry_vix_level'],
        'entry_z_threshold_used': position['entry_z_threshold_used'],
        'exit_z_threshold_used': position['exit_z_threshold_used'],
        
        # Exit information
        'exit_z_score': row['z_residual'],
        'exit_x_price': exit_x_price,
        'exit_y_price': exit_y_price,
        'exit_reason': exit_reason,
        'holding_period': holding_period,
        
        # P&L calculation
        'y_dollar_pnl': y_dollar_pnl,
        'x_dollar_pnl': x_dollar_pnl,
        'total_dollar_pnl': total_dollar_pnl,
        'return_pct': return_pct,
        
        # Additional metrics
        'realized_implied_return': abs(position['entry_y_price'] - exit_y_price) / position['entry_y_price']
    }
    
    return trade

def backtest_pair(pair_name, vix_data, db_path='data/pairs_database.db'):
    # Backtest a single pair and return list of completed trades
    df, symbol1, symbol2 = load_pair_data(pair_name, db_path)
    
    if df.empty or len(df) < 100:
        return []
    
    signals = generate_signals(df, vix_data)
    trades = []
    current_position = None
    
    for date, row in df.iterrows():
        signal_row = signals.loc[date]
        
        # Check for exit signals first
        if current_position is not None:
            should_exit = False
            exit_reason = None
            
            # Get exit threshold for this date/regime
            exit_threshold = signal_row['exit_z_threshold']
            
            # Check for mean reversion exit based on direction and regime-adjusted threshold
            if current_position['direction'] == 'long':
                if row['z_residual'] >= exit_threshold:
                    should_exit = True
                    exit_reason = 'mean_revert'
            elif current_position['direction'] == 'short':
                if row['z_residual'] <= -exit_threshold:  # Note: negative for short
                    should_exit = True
                    exit_reason = 'mean_revert'
            
            if should_exit:
                trade = close_position(current_position, row, date, exit_reason, symbol1, symbol2)
                trades.append(trade)
                current_position = None
        
        # Check for entry signals (only if no current position)
        elif current_position is None:
            if signal_row['entry_long']:
                regime_info = {
                    'regime': signal_row['regime'],
                    'vix_level': signal_row.get('vix_level', np.nan),
                    'entry_z': signal_row['entry_z_threshold'],
                    'exit_z': signal_row['exit_z_threshold']
                }
                current_position = open_position(row, date, 'long', symbol1, symbol2, regime_info)
            elif signal_row['entry_short']:
                regime_info = {
                    'regime': signal_row['regime'],
                    'vix_level': signal_row.get('vix_level', np.nan),
                    'entry_z': signal_row['entry_z_threshold'],
                    'exit_z': signal_row['exit_z_threshold']
                }
                current_position = open_position(row, date, 'short', symbol1, symbol2, regime_info)
    
    return trades

def run_full_backtest(db_path='data/pairs_database.db', start_pair=0, max_pairs=None, sample_pairs=None):
    
    # Load VIX data for regime detection
    vix_data = load_vix_data()
    
    print(f"Strategy parameters:")
    if USE_REGIME_DETECTION and not vix_data.empty:
        print(f"  REGIME DETECTION ENABLED")
        print(f"  VIX thresholds: Low < {VIX_LOW_THRESHOLD}, High > {VIX_HIGH_THRESHOLD}")
        print(f"  Low vol regime: Entry ±2.0, Exit ±0.2, R² ≥ 85%, Coint ≤ 0.3%")
        print(f"  Medium vol regime: Entry ±2.5, Exit 0.0, R² ≥ 80%, Coint ≤ 0.5%")
        print(f"  High vol regime: Entry ±3.0, Exit ±0.5, R² ≥ 75%, Coint ≤ 1.0%")
    else:
        print(f"  BASELINE MODE (No regime detection)")
        print(f"  Entry z-score threshold: ±{BASE_ENTRY_Z_THRESHOLD}")
        print(f"  Exit z-score threshold: {BASE_EXIT_Z_THRESHOLD}")
        print(f"  Minimum R²: {BASE_RSQ_THRESHOLD:.1%}")
        print(f"  Maximum cointegration p-value: {BASE_COINT_THRESHOLD:.3%}")
        print(f"  Minimum implied return: {MIN_IMPLIED_RETURN:.1%}")

    print(f"  Earnings buffer: {EARNINGS_BUFFER_DAYS} days")
    print(f"  Earnings exit buffer: {EARNINGS_EXIT_DAYS} days (DISABLED)")
    print(f"  Earnings recency filter: {EARNINGS_RECENCY_DAYS} days")
    
    # Get pairs to process
    if sample_pairs:
        pairs_to_process = sample_pairs
    else:
        all_pairs = get_all_pairs(db_path)
        end_pair = start_pair + max_pairs if max_pairs else len(all_pairs)
        pairs_to_process = all_pairs[start_pair:end_pair]
    
    print(f"\nProcessing {len(pairs_to_process)} pairs...")
    
    all_trades = []
    
    # Process each pair with progress bar
    for pair_name in tqdm(pairs_to_process, desc="Backtesting pairs"):
        try:
            pair_trades = backtest_pair(pair_name, vix_data, db_path)
            all_trades.extend(pair_trades)
        except Exception as e:
            print(f"Error processing {pair_name}: {e}")
            continue
    
    print(f"\nBacktest complete!")
    print(f"Total trades generated: {len(all_trades)}")
    
    if all_trades:
        total_dollar_pnl = sum(trade['total_dollar_pnl'] for trade in all_trades)
        total_capital_deployed = len(all_trades) * 2 * POSITION_SIZE  # Each trade deploys $1000
        portfolio_return = total_dollar_pnl / total_capital_deployed if total_capital_deployed > 0 else 0
        
        profitable_trades = sum(1 for trade in all_trades if trade['total_dollar_pnl'] > 0)
        win_rate = profitable_trades / len(all_trades)
        avg_return = np.mean([trade['return_pct'] for trade in all_trades])
        
        print(f"Total dollar P&L: ${total_dollar_pnl:,.2f}")
        print(f"Portfolio return: {portfolio_return:.2%}")
        print(f"Win rate: {win_rate:.2%}")
        print(f"Average return per trade: {avg_return:.2%}")
    
        if USE_REGIME_DETECTION and not vix_data.empty:
            trades_df = pd.DataFrame(all_trades)
            regime_breakdown = trades_df.groupby('entry_regime').agg({
                'total_dollar_pnl': ['count', 'sum', 'mean'],
                'return_pct': 'mean',
                'entry_vix_level': 'mean'
            }).round(4)
            print(f"\nRegime breakdown:")
            print(regime_breakdown)

    return all_trades

def create_trades_dataframe(all_trades):
    # Convert trades list to DataFrame
    if not all_trades:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_trades)
    
    # Convert dates
    df['entry_date'] = pd.to_datetime(df['entry_date'])
    df['exit_date'] = pd.to_datetime(df['exit_date'])
    
    # Add analysis columns
    df['entry_year'] = df['entry_date'].dt.year
    df['entry_month'] = df['entry_date'].dt.month
    df['entry_quarter'] = df['entry_date'].dt.quarter
    df['is_profitable'] = df['total_dollar_pnl'] > 0
    
    return df

def generate_summary_stats(df):
    # Generate summary statistics
    if df.empty:
        return pd.DataFrame()
    
    stats = {
        'Total_Trades': len(df),
        'Profitable_Trades': sum(df['is_profitable']),
        'Win_Rate': df['is_profitable'].mean(),
        'Total_Dollar_PnL': df['total_dollar_pnl'].sum(),
        'Average_Dollar_PnL_Per_Trade': df['total_dollar_pnl'].mean(), 
        'Average_Return_Per_Trade': df['return_pct'].mean(),
        'Average_Holding_Period': df['holding_period'].mean(),
        'Max_Return': df['return_pct'].max(),
        'Min_Return': df['return_pct'].min(),
        'Std_Return': df['return_pct'].std(),
        'Unique_Pairs_Traded': df['pair'].nunique(),
        'Long_Trades': sum(df['direction'] == 'long'),
        'Short_Trades': sum(df['direction'] == 'short'),
        'Mean_Revert_Exits': sum(df['exit_reason'] == 'mean_revert'),
        'Earnings_Exits': sum(df['exit_reason'] == 'earnings'),
    }
    
    return pd.DataFrame(list(stats.items()), columns=['Metric', 'Value'])

def export_to_excel(all_trades, filename='statistical_arbitrage_trades.xlsx'):
    # Export trades to Excel
    df = create_trades_dataframe(all_trades)
    
    if df.empty:
        print("No trades to export")
        return
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Main trades sheet
        df.to_excel(writer, sheet_name='All_Trades', index=False)
        
        # Summary statistics
        summary_stats = generate_summary_stats(df)
        summary_stats.to_excel(writer, sheet_name='Summary_Stats')
        
        # Monthly performance
        monthly_perf = df.groupby(['entry_year', 'entry_month']).agg({
            'total_dollar_pnl': ['count', 'sum', 'mean'],
            'return_pct': 'mean',
            'is_profitable': 'mean'
        }).round(4)
        monthly_perf.to_excel(writer, sheet_name='Monthly_Performance')

        if USE_REGIME_DETECTION and 'entry_regime' in df.columns:
            regime_perf = df.groupby('entry_regime').agg({
                'total_dollar_pnl': ['count', 'sum', 'mean'],
                'return_pct': ['mean', 'std'],
                'holding_period': 'mean',
                'entry_vix_level': 'mean',
                'is_profitable': 'mean'
            }).round(4)
            regime_perf.to_excel(writer, sheet_name='Regime_Performance')
    
    
    print(f"Trades exported to {filename}")
    print(f"Exported {len(df)} trades across {len(df['pair'].unique())} unique pairs")

if __name__ == "__main__":
    # Test with small sample first
    all_pairs = get_all_pairs()
    sample_pairs = all_pairs
    
    # Run backtest
    trades = run_full_backtest(sample_pairs=sample_pairs)
    
    # Export results
    export_to_excel(trades, 'regime_27k_trades.xlsx')
    
    # Show sample trades
    trades_df = create_trades_dataframe(trades)
    if not trades_df.empty:
        print("\nSample trades:")
        print(trades_df[['pair', 'entry_date', 'direction', 'return_pct', 'entry_regime', 'entry_vix_level']].head())
    
