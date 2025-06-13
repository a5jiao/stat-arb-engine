import pandas as pd

def download_sp500_tickers(output_csv="data/sp500_tickers.csv"):
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    df = pd.read_html(url)[0]  # Read first table on page
    tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()  # Fix for yfinance compatibility

    # Save to CSV (one ticker per line, no headers/index)
    pd.Series(tickers).to_csv(output_csv, index=False, header=False)
    print(f"Saved {len(tickers)} tickers to {output_csv}")

if __name__ == "__main__":
    download_sp500_tickers()
