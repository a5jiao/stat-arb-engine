import yfinance as yf
import pandas as pd
import time

def load_tickers(filepath="data/sp500_tickers.csv"):
    with open(filepath) as f:
        return [line.strip() for line in f if line.strip()]

def download_price_data(tickers, start="2014-01-01", end=None, save_path="data/historical_prices.csv"):
    all_data = pd.DataFrame()
    
    for i, ticker in enumerate(tickers):
        try:
            print(f"Downloading {ticker} ({i+1}/{len(tickers)})...")
            data = yf.download(ticker, start=start, end=end, auto_adjust=True)
            data = data['Close']  # adjusted close series
            data.name = ticker   # set series name to ticker (not rename)
            all_data = pd.concat([all_data, data], axis=1)
            time.sleep(0.1)
        except Exception as e:
            print(f"Failed to download {ticker}: {e}")

    all_data.to_csv(save_path)
    print(f"Saved price data to {save_path}")

if __name__ == "__main__":
    tickers = load_tickers()
    download_price_data(tickers)
