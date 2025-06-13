import pandas as pd

def create_sp500_sector_mapping(output_csv="data/sp500_sectors.csv"):
    try:
        # Get S&P 500 data from Wikipedia
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        df = tables[0] 
        
        # Clean up the data
        df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)  # Fix for yfinance compatibility
        
        # Select relevant columns and rename for clarity
        sector_df = df[['Symbol', 'GICS Sector']].copy()
        sector_df.columns = ['ticker', 'sector']
        
        sector_df['sector'] = sector_df['sector'].str.strip()
        
        sector_df = sector_df.sort_values('ticker').reset_index(drop=True)
        
        # Save to CSV
        sector_df.to_csv(output_csv, index=False)
        
        print(f"Saved {len(sector_df)} ticker-sector mappings to {output_csv}")
        print(f"\nSector breakdown:")
        print(sector_df['sector'].value_counts())
        
        return sector_df
        
    except Exception as e:
        print(f"Error creating sector mapping: {e}")
        return None

if __name__ == "__main__":
    create_sp500_sector_mapping()