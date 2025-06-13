import blpapi
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from tqdm import tqdm

class BloombergEarningsFetcher:
    def __init__(self):
        """Initialize Bloomberg API session"""
        self.session = None
        self.ref_data_service = None
        
    def start_session(self):
        """Start Bloomberg API session"""
        try:
            # Create session options
            session_options = blpapi.SessionOptions()
            session_options.setServerHost('localhost')
            session_options.setServerPort(8194)
            
            # Create and start session
            self.session = blpapi.Session(session_options)
            if not self.session.start():
                raise Exception("Failed to start Bloomberg session")
                
            # Open reference data service
            if not self.session.openService("//blp/refdata"):
                raise Exception("Failed to open reference data service")
                
            self.ref_data_service = self.session.getService("//blp/refdata")
            print("Bloomberg session started successfully")
            
        except Exception as e:
            print(f"Error starting Bloomberg session: {e}")
            raise
    
    def stop_session(self):
        """Stop Bloomberg API session"""
        if self.session:
            self.session.stop()
            print("Bloomberg session stopped")
    
    def get_earnings_dates(self, ticker, years_back=10):
        """
        Get historical earnings dates for a single ticker
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL US Equity')
            years_back: Number of years of historical data
            
        Returns:
            List of earnings dates
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years_back * 365)
            
            # Create request
            request = self.ref_data_service.createRequest("HistoricalDataRequest")
            request.getElement("securities").appendValue(ticker)
            request.getElement("fields").appendValue("DY895")  # Earnings announcement date
            request.set("startDate", start_date.strftime("%Y%m%d"))
            request.set("endDate", end_date.strftime("%Y%m%d"))
            request.set("periodicitySelection", "DAILY")
            
            # Send request
            self.session.sendRequest(request)
            
            # Process response
            earnings_dates = []
            while True:
                event = self.session.nextEvent(500)  # 500ms timeout
                
                if event.eventType() == blpapi.Event.RESPONSE or event.eventType() == blpapi.Event.PARTIAL_RESPONSE:
                    for msg in event:
                        security_data = msg.getElement("securityData")
                        field_data = security_data.getElement("fieldData")
                        
                        for i in range(field_data.numValues()):
                            field_data_point = field_data.getValue(i)
                            if field_data_point.hasElement("DY895"):
                                date_val = field_data_point.getElement("date").getValue()
                                earnings_val = field_data_point.getElement("DY895").getValue()
                                
                                if earnings_val and not pd.isna(earnings_val):
                                    earnings_dates.append(pd.to_datetime(earnings_val))
                
                if event.eventType() == blpapi.Event.RESPONSE:
                    break
            
            # Remove duplicates and sort
            earnings_dates = sorted(list(set(earnings_dates)))
            return earnings_dates
            
        except Exception as e:
            print(f"Error getting earnings dates for {ticker}: {e}")
            return []
    
    def fetch_all_earnings_dates(self, tickers, years_back=10, output_csv="data/earnings_dates.csv"):
        """
        Fetch earnings dates for all tickers and save as DataFrame/CSV
        
        Args:
            tickers: List of ticker symbols
            years_back: Number of years of historical data
            output_csv: Output CSV file path
            
        Returns:
            DataFrame with earnings dates
        """
        try:
            self.start_session()
            
            # Dictionary to store earnings dates for each ticker
            earnings_data = {}
            
            # Process each ticker
            for ticker in tqdm(tickers, desc="Fetching earnings dates"):
                # Bloomberg format: add ' US Equity' if not already formatted
                if ' US Equity' not in ticker:
                    bloomberg_ticker = f"{ticker} US Equity"
                else:
                    bloomberg_ticker = ticker
                
                # Get earnings dates
                earnings_dates = self.get_earnings_dates(bloomberg_ticker, years_back)
                
                # Store in dictionary (clean ticker name as key)
                clean_ticker = ticker.replace(' US Equity', '')
                earnings_data[clean_ticker] = earnings_dates
                
                # Brief pause to avoid rate limiting
                import time
                time.sleep(0.1)
            
            # Convert to DataFrame format
            # Find the maximum number of earnings dates across all stocks
            max_earnings = max(len(dates) for dates in earnings_data.values()) if earnings_data else 0
            
            # Create DataFrame with earnings dates as rows, tickers as columns
            df_data = {}
            for ticker, dates in earnings_data.items():
                # Pad with NaN if fewer earnings dates than max
                padded_dates = dates + [pd.NaT] * (max_earnings - len(dates))
                df_data[ticker] = padded_dates
            
            # Create DataFrame
            earnings_df = pd.DataFrame(df_data)
            
            # Sort each column (ticker) by date, with NaN values at the end
            for col in earnings_df.columns:
                sorted_dates = earnings_df[col].dropna().sort_values()
                nan_count = earnings_df[col].isna().sum()
                earnings_df[col] = pd.concat([sorted_dates, pd.Series([pd.NaT] * nan_count)]).reset_index(drop=True)
            
            # Save to CSV
            earnings_df.to_csv(output_csv, index=False)
            print(f"Earnings dates saved to {output_csv}")
            print(f"Shape: {earnings_df.shape}")
            print(f"Columns: {list(earnings_df.columns)}")
            
            return earnings_df
            
        except Exception as e:
            print(f"Error in fetch_all_earnings_dates: {e}")
            return pd.DataFrame()
            
        finally:
            self.stop_session()

def load_tickers(filepath="data/sp500_tickers.csv"):
    """Load ticker list from CSV file"""
    with open(filepath) as f:
        return [line.strip() for line in f if line.strip()]

# Main execution
if __name__ == "__main__":
    # Load S&P 500 tickers
    tickers = load_tickers()
    print(f"Loaded {len(tickers)} tickers")
    
    # Initialize fetcher
    fetcher = BloombergEarningsFetcher()
    
    # Fetch earnings dates for all tickers
    earnings_df = fetcher.fetch_all_earnings_dates(
        tickers=tickers,
        years_back=10,
        output_csv="data/earnings_dates.csv"
    )
    
    # Display sample results
    if not earnings_df.empty:
        print("\nSample earnings dates:")
        print(earnings_df.head(10))
        
        # Show earnings count per ticker
        earnings_counts = earnings_df.notna().sum()
        print(f"\nEarnings dates per ticker (sample):")
        print(earnings_counts.head(10))
        
        print(f"\nTotal earnings dates collected: {earnings_df.notna().sum().sum()}")
        