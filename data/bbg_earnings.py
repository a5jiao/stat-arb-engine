# earnings_date_download.py

import blpapi
import pandas as pd
from blpapi import SessionOptions, Session, Service, Request
import datetime
import time

# Bloomberg field for earnings announcement date
FIELD = "DY895"
YEARS_BACK = 10
INPUT_CSV = "data/sp500_tickers.csv"
OUTPUT_CSV = "data/earnings_dates.csv"

def init_session():
    options = SessionOptions()
    options.setServerHost("localhost")
    options.setServerPort(8194)
    session = Session(options)
    if not session.start():
        raise RuntimeError("Failed to start session.")
    if not session.openService("//blp/refdata"):
        raise RuntimeError("Failed to open //blp/refdata")
    return session

def fetch_earnings_dates(session, tickers):
    refDataSvc = session.getService("//blp/refdata")
    request = refDataSvc.createRequest("HistoricalDataRequest")
    request.set("periodicitySelection", "DAILY")
    request.set("startDate", (datetime.datetime.now() - datetime.timedelta(days=365 * YEARS_BACK)).strftime('%Y%m%d'))
    request.set("endDate", datetime.datetime.now().strftime('%Y%m%d'))
    request.getElement("fields").appendValue(FIELD)
    
    for ticker in tickers:
        request.getElement("securities").appendValue(ticker)

    session.sendRequest(request)
    earnings_data = {ticker: [] for ticker in tickers}

    while True:
        ev = session.nextEvent(500)
        for msg in ev:
            if msg.hasElement("securityData"):
                securityData = msg.getElement("securityData")
                ticker = securityData.getElementAsString("security")
                if securityData.hasElement("fieldData"):
                    for item in securityData.getElement("fieldData").values():
                        if item.hasElement(FIELD):
                            earnings_date = item.getElementAsDatetime(FIELD)
                            earnings_data[ticker].append(earnings_date.strftime('%Y-%m-%d'))
        if ev.eventType() == blpapi.Event.RESPONSE:
            break

    # Transpose into desired DataFrame format
    max_len = max(len(v) for v in earnings_data.values())
    for ticker in earnings_data:
        earnings_data[ticker] += [''] * (max_len - len(earnings_data[ticker]))
    df = pd.DataFrame(earnings_data)
    return df

def main():
    tickers = pd.read_csv(INPUT_CSV, header=None)[0].tolist()
    session = init_session()
    df = fetch_earnings_dates(session, tickers[:5])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Earnings dates saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
