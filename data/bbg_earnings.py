# src/download_earnings_bbg.py

# FIELD DY895

import blpapi
import pandas as pd
import datetime
import time

from blpapi import SessionOptions, Session

# --- SETTINGS ---
TICKERS_FILE = "data/sp500_tickers.csv"
SAVE_PATH = "data/earnings_dates_bbg.csv"
START_DATE = (datetime.datetime.now() - datetime.timedelta(days=365 * 10)).strftime("%Y%m%d")
END_DATE = datetime.datetime.now().strftime("%Y%m%d")

# --- Bloomberg Request Helper ---
def fetch_earnings_dates(tickers):
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

    request.append("fields", "ERN_ANN_DT")
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
                        earnings_date = record.getElementAsDatetime("ERN_ANN_DT").date()
                        data.append({
                            "Ticker": ticker,
                            "Record Date": date,
                            "Earnings Date": earnings_date
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

    print(f"Fetching 10Y earnings dates for {len(tickers)} tickers...")

    all_data = fetch_earnings_dates(tickers)

        # Build pivoted DataFrame: Tickers as columns, earnings dates as rows
    df = pd.DataFrame(all_data)

    # Group by ticker, sort each group by earnings date
    df = df.sort_values(by=["Ticker", "Earnings Date"])

    # For each ticker, get the list of earnings dates (sorted)
    grouped = df.groupby("Ticker")["Earnings Date"].apply(list)

    # Combine into a DataFrame
    earnings_pivot = pd.DataFrame.from_dict(grouped.to_dict(), orient="columns")

    # Transpose so tickers are columns
    earnings_pivot = earnings_pivot.transpose()

    # Pad shorter lists with NaT and transpose back
    earnings_pivot = earnings_pivot.apply(lambda col: pd.Series(col), axis=1)

    # Save to CSV
    earnings_pivot.to_csv(SAVE_PATH, index=False)
    print(f"Saved earnings history to: {SAVE_PATH}")

