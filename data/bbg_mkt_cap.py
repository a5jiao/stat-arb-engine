# src/download_market_cap_bbg.py

import blpapi
import pandas as pd
import datetime

from blpapi import SessionOptions, Session

# --- SETTINGS ---
TICKERS_FILE = "data/sp500_tickers.csv"
SAVE_PATH = "data/market_cap_bbg.csv"
START_DATE = (datetime.datetime.now() - datetime.timedelta(days=365 * 10)).strftime("%Y%m%d")
END_DATE = datetime.datetime.now().strftime("%Y%m%d")

# --- Bloomberg Request Helper ---
def fetch_market_cap(tickers):
    options = SessionOptions()

    session = Session(options)
    if not session.start():
        print("Failed to start session.")
        return []

    if not session.openService("//blp/refdata"):
        print("Failed to open service.")
        return []

    service = session.getService("//blp/refdata")
    request = service.createRequest("HistoricalDataRequest")

    for ticker in tickers:
        request.append("securities", f"{ticker} US Equity")

    request.append("fields", "CUR_MKT_CAP")
    request.set("startDate", START_DATE)
    request.set("endDate", END_DATE)
    request.set("periodicitySelection", "DAILY")
    request.set("adjustmentSplit", True)

    session.sendRequest(request)
    data = []

    while True:
        ev = session.nextEvent()
        for msg in ev:
            if msg.messageType() == "HistoricalDataResponse":
                sec_data = msg.getElement("securityData")
                security = sec_data.getElementAsString("security")
                ticker = security.split()[0]

                field_data = sec_data.getElement("fieldData")
                for i in range(field_data.numValues()):
                    record = field_data.getValueAsElement(i)
                    date = record.getElementAsDatetime("date").date()
                    try:
                        mkt_cap = record.getElementAsFloat64("CUR_MKT_CAP")
                        data.append({
                            "Ticker": ticker,
                            "Date": date,
                            "Market Cap": mkt_cap
                        })
                    except:
                        continue

        if ev.eventType() == blpapi.Event.RESPONSE:
            break

    return data

# --- MAIN ---
if __name__ == "__main__":
    with open(TICKERS_FILE) as f:
        tickers = [line.strip() for line in f if line.strip()]

    print(f"Fetching 10Y daily market cap for {len(tickers)} tickers...")

    all_data = fetch_market_cap(tickers)
    df = pd.DataFrame(all_data)

    # Pivot to format: Dates as rows, Tickers as columns
    pivot_df = df.pivot(index="Date", columns="Ticker", values="Market Cap")

    # Sort by date
    pivot_df = pivot_df.sort_index()

    # Save
    pivot_df.to_csv(SAVE_PATH)
    print(f"Saved market cap history to: {SAVE_PATH}")
