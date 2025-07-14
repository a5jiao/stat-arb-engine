import yfinance as yf

ticker = "^VIX"
start_date = "2015-06-16"
end_date = "2025-06-13"

vix_data = yf.download(ticker, start=start_date, end=end_date)['Close']

vix_data.to_csv("data/vix_data.csv")

