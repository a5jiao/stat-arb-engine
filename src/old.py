import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Strategy parameters
ENTRY_Z_THRESHOLD = 2.5
EXIT_Z_THRESHOLD = 0.0
MIN_IMPLIED_RETURN = 0.03  # 3%
RSQ_THRESHOLD = 0.8
COINT_THRESHOLD = 0.005
EARNINGS_BUFFER_DAYS = 20
EARNINGS_EXIT_DAYS = 7
EARNINGS_RECENCY_DAYS = 90

def get_all_pairs(db_path='data/pairs_database.db'):
    """Get list of all pair tables in database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'pair_%'")
    pairs = [row[0] for row in cursor.fetchall()]
    conn.close()
    return pairs

def load_pair_data(pair_name, db_path='data/pairs_database.db'):
    """Load and prepare data for a specific pair"""
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
    """Calculate days until next earnings"""
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
    """Check if trade should be filtered due to earnings"""
    days1 = row['days_to_earnings1']
    days2 = row['days_to_earnings2']
    
    if trade_type == 'entry':
        # Don't enter if earnings within 20 days or if earnings were within last 90 days
        for days in [days1, days2]:
            if pd.notna(days):
                if 0 <= days <= EARNINGS_BUFFER_DAYS:  # Earnings coming up
                    return True
                if -EARNINGS_RECENCY_DAYS <= days < 0:  # Recent earnings
                    return True
    
    elif trade_type == 'exit':
        # Force exit if earnings within 7 days
        for days in [days1, days2]:
            if pd.notna(days) and 0 <= days <= EARNINGS_EXIT_DAYS:
                return True
                
    return False

def generate_signals(df):
    """Generate entry and exit signals for a pair"""
    signals = pd.DataFrame(index=df.index)
    
    # Entry conditions
    long_conditions = (
        (df['z_residual'] <= -ENTRY_Z_THRESHOLD) &  # Y underpriced
        (df['implied_return'] >= MIN_IMPLIED_RETURN) &
        (df['r_squared'] >= RSQ_THRESHOLD) &
        (df['coint_p_value'] <= COINT_THRESHOLD) &
        (~df.apply(lambda row: check_earnings_filter(row, 'entry'), axis=1))
    )
    
    short_conditions = (
        (df['z_residual'] >= ENTRY_Z_THRESHOLD) &   # Y overpriced
        (df['implied_return'] >= MIN_IMPLIED_RETURN) &
        (df['r_squared'] >= RSQ_THRESHOLD) &
        (df['coint_p_value'] <= COINT_THRESHOLD) &
        (~df.apply(lambda row: check_earnings_filter(row, 'entry'), axis=1))
    )
    
    # Exit conditions
    mean_revert_exit = (
        (df['z_residual'] <= EXIT_Z_THRESHOLD) & (df['z_residual'] >= -EXIT_Z_THRESHOLD)
    )
    
    earnings_exit = df.apply(lambda row: check_earnings_filter(row, 'exit'), axis=1)
    
    signals['entry_long'] = long_conditions
    signals['entry_short'] = short_conditions
    signals['exit_mean_revert'] = mean_revert_exit
    signals['exit_earnings'] = earnings_exit
    signals['exit'] = mean_revert_exit #| earnings_exit
    
    return signals

def open_position(row, date, direction, symbol1, symbol2):
    """Create a new position record"""
    return {
        'entry_date': date,
        'direction': direction,
        'entry_z_score': row['z_residual'],
        'entry_implied_return': row['implied_return'],
        'entry_coint': row['coint_p_value'],
        'entry_x_price': row[f'{symbol1}_price'],
        'entry_y_price': row[f'{symbol2}_price'],
        'entry_y_implied': row['y_implied'],
        'slope': row['slope'],
        'r_squared': row['r_squared'],
        'symbol1': symbol1,
        'symbol2': symbol2,
        'sector': row[f'{symbol1}_sector'],
        'entry_market_cap1': row.get('Mkt_cap1', np.nan),
        'entry_market_cap2': row.get('Mkt_cap2', np.nan),
        'days_to_earnings1': row['days_to_earnings1'],
        'days_to_earnings2': row['days_to_earnings2']
    }

def close_position(position, row, date, exit_reason, symbol1, symbol2):
    """Close a position and calculate P&L"""
    holding_period = (date - position['entry_date']).days
    
    exit_x_price = row[f'{symbol1}_price']
    exit_y_price = row[f'{symbol2}_price']
    
    # Calculate P&L based on direction
    if position['direction'] == 'long':
        # Long Y, Short X
        y_pnl = exit_y_price - position['entry_y_price']
        x_pnl = position['entry_x_price'] - exit_x_price
    else:  # short
        # Short Y, Long X
        y_pnl = position['entry_y_price'] - exit_y_price
        x_pnl = exit_x_price - position['entry_x_price']
    
    # Weight X P&L by hedge ratio (slope)
    total_pnl = y_pnl + (position['slope'] * x_pnl)
    return_pct = total_pnl / position['entry_y_price']
    
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
        
        # Exit information
        
        'exit_z_score': row['z_residual'],
        'exit_x_price': exit_x_price,
        'exit_y_price': exit_y_price,
        'exit_reason': exit_reason,
        'holding_period': holding_period,
        
        # P&L calculation
        'y_pnl': y_pnl,
        'x_pnl': x_pnl,
        'total_pnl': total_pnl,
        'return_pct': return_pct,
        
        # Additional metrics
        'realized_implied_return': abs(position['entry_y_price'] - exit_y_price) / position['entry_y_price']
    }
    
    return trade

def backtest_pair(pair_name, db_path='data/pairs_database.db'):
    """Backtest a single pair and return list of completed trades"""
    df, symbol1, symbol2 = load_pair_data(pair_name, db_path)
    
    if df.empty or len(df) < 100:
        return []
    
    signals = generate_signals(df)
    trades = []
    current_position = None
    
    for date, row in df.iterrows():
        signal_row = signals.loc[date]
        
        # Check for exit signals first
        if current_position is not None:
            if signal_row['exit']:
                exit_reason = 'mean_revert' if signal_row['exit_mean_revert'] else 'earnings'
                trade = close_position(current_position, row, date, exit_reason, symbol1, symbol2)
                trades.append(trade)
                current_position = None
        
        # Check for entry signals (only if no current position)
        elif current_position is None:
            if signal_row['entry_long']:
                current_position = open_position(row, date, 'long', symbol1, symbol2)
            elif signal_row['entry_short']:
                current_position = open_position(row, date, 'short', symbol1, symbol2)
    
    # Close any remaining position at end of data
    if current_position is not None:
        final_row = df.iloc[-1]
        final_date = df.index[-1]
        trade = close_position(current_position, final_row, final_date, 'end_of_data', symbol1, symbol2)
        trades.append(trade)
    
    return trades

def run_full_backtest(db_path='data/pairs_database.db', start_pair=0, max_pairs=None, sample_pairs=None):
    """Run backtest on all pairs or a subset"""
    print(f"Strategy parameters:")
    print(f"  Entry z-score threshold: ±{ENTRY_Z_THRESHOLD}")
    print(f"  Exit z-score threshold: {EXIT_Z_THRESHOLD}")
    print(f"  Minimum implied return: {MIN_IMPLIED_RETURN:.1%}")
    print(f"  Earnings buffer: {EARNINGS_BUFFER_DAYS} days")
    print(f"  Earnings exit buffer: {EARNINGS_EXIT_DAYS} days")
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
            pair_trades = backtest_pair(pair_name, db_path)
            all_trades.extend(pair_trades)
        except Exception as e:
            print(f"Error processing {pair_name}: {e}")
            continue
    
    print(f"\nBacktest complete!")
    print(f"Total trades generated: {len(all_trades)}")
    
    if all_trades:
        profitable_trades = sum(1 for trade in all_trades if trade['total_pnl'] > 0)
        win_rate = profitable_trades / len(all_trades)
        avg_return = np.mean([trade['return_pct'] for trade in all_trades])
        print(f"Win rate: {win_rate:.2%}")
        print(f"Average return per trade: {avg_return:.2%}")
    
    return all_trades

def create_trades_dataframe(all_trades):
    """Convert trades list to DataFrame"""
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
    df['is_profitable'] = df['total_pnl'] > 0
    
    return df

def generate_summary_stats(df):
    """Generate summary statistics"""
    if df.empty:
        return pd.DataFrame()
    
    stats = {
        'Total_Trades': len(df),
        'Profitable_Trades': sum(df['is_profitable']),
        'Win_Rate': df['is_profitable'].mean(),
        'Total_PnL': df['total_pnl'].sum(),
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
        'End_of_Data_Exits': sum(df['exit_reason'] == 'end_of_data')
    }
    
    return pd.DataFrame(list(stats.items()), columns=['Metric', 'Value'])

def export_to_excel(all_trades, filename='statistical_arbitrage_trades.xlsx'):
    """Export trades to Excel for analysis"""
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
            'total_pnl': ['count', 'sum', 'mean'],
            'return_pct': 'mean',
            'is_profitable': 'mean'
        }).round(4)
        monthly_perf.to_excel(writer, sheet_name='Monthly_Performance')
    
    print(f"Trades exported to {filename}")
    print(f"Exported {len(df)} trades across {len(df['pair'].unique())} unique pairs")

if __name__ == "__main__":
    # Test with small sample first
    print("Testing with first 30 pairs...")
    all_pairs = get_all_pairs()
    sample_pairs = all_pairs[:30]
    
    # Run backtest
    trades = run_full_backtest(sample_pairs=sample_pairs)
    
    # Export results
    export_to_excel(trades, 'test_trades.xlsx')
    
    # Show sample trades
    trades_df = create_trades_dataframe(trades)
    if not trades_df.empty:
        print("\nSample trades:")
        print(trades_df[['pair', 'entry_date', 'direction', 'return_pct', 'exit_reason']].head())
