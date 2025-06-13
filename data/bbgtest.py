# src/download_prices_bbg.py

import blpapi
from blpapi import SessionOptions, Session, Request
import pandas as pd
import datetime
import time

TICKER_PATH = r"C:\Users\a5jiao\stat-arb-engine\data\sp500_tickers.csv"
SAVE_PATH = r"C:\Users\a5jiao\stat-arb-engine\data\sp500_prices_10y.csv"
FIELD = "PX_LAST"
START_DATE = (datetime.datetime.now() - datetime.timedelta(days=365 * 10)).strftime("%Y%m%d")
END_DATE = datetime.datetime.now().strftime("%Y%m%d")

def init_session():
    options = SessionOptions()
    options.setServerHost("localhost")
    options.setServerPort(8194)
    session = Session(options)
    if not session.start():
        raise RuntimeError("Failed to start Bloomberg session.")
    if not session.openService("//blp/refdata"):
        raise RuntimeError("Failed to open //blp/refdata service.")
    return session

def get_price_data(tickers, session):
    refDataService = session.getService("//blp/refdata")
    request = refDataService.createRequest("HistoricalDataRequest")

    for ticker in tickers:
        request.getElement("securities").appendValue(ticker)
    request.getElement("fields").appendValue(FIELD)
    request.set("startDate", START_DATE)
    request.set("endDate", END_DATE)
    request.set("periodicitySelection", "DAILY")
    request.set("adjustmentSplit", True)
    request.set("adjustmentNormal", True)

    session.sendRequest(request)

    # Dictionary to store prices
    data_dict = {}

    while True:
        ev = session.nextEvent(500)
        for msg in ev:
            if msg.hasElement("securityData"):
                secData = msg.getElement("securityData")
                ticker = secData.getElementAsString("security")
                if secData.hasElement("fieldData"):
                    fieldDataArray = secData.getElement("fieldData")
                    dates, prices = [], []
                    for i in range(fieldDataArray.numValues()):
                        fieldData = fieldDataArray.getValueAsElement(i)
                        if fieldData.hasElement(FIELD):
                            date = fieldData.getElementAsDatetime("date").date()
                            price = fieldData.getElementAsFloat(FIELD)
                            dates.append(date)
                            prices.append(price)
                    data_dict[ticker] = pd.Series(prices, index=dates)

        if ev.eventType() == blpapi.Event.RESPONSE:
            break

    return pd.DataFrame(data_dict)

def main():
    tickers_df = pd.read_csv(TICKER_PATH, header=None)
    tickers = tickers_df[0].tolist()
    session = init_session()
    print("Session initialized. Fetching data...")
    df = get_price_data(tickers[:5], session)
    print("Data fetched. Saving to CSV...")
    df.to_csv(SAVE_PATH, index=True)
    print(f"Saved to {SAVE_PATH}")

if __name__ == "__main__":
    main()
