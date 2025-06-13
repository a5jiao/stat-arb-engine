import blpapi
from blpapi import Session, SessionOptions
import pandas as pd
import datetime
from datetime import timedelta
import numpy as np
from tqdm import tqdm

def init_session():
    options = SessionOptions()
    options.setServerHost("localhost")
    options.setServerPort(8194)
    session = Session(options)
    if not session.start():
        raise RuntimeError("Failed to start session.")
    if not session.openService("//blp/refdata"):
        raise RuntimeError("Failed to open //blp/refdata.")
    return session

def get_px_last(ticker, session):
    refDataService = session.getService("//blp/refdata")
    request = refDataService.createRequest("HistoricalDataRequest")
    request.append("securities", ticker)
    request.append("fields", "PX_LAST")  

    end_date = datetime.datetime.now().strftime('%Y%m%d')
    start_date = (datetime.datetime.now()-timedelta(365*10)).strftime('%Y%m%d')

    request.set("startDate", start_date)
    request.set("endDate", end_date)

    session.sendRequest(request)

    values = {}
    while True:
        event = session.nextEvent()
        for msg in event:
            if msg.hasElement("securityData"):
                #print(msg)
                securityData = msg.getElement("securityData")
                fieldData = securityData.getElement("fieldData")
                for i in range(fieldData.numValues()):
                    data = fieldData.getValue(i)
                    values[data['date']] = data['PX_LAST']
                        
        if event.eventType() == blpapi.Event.RESPONSE:
            break

    return values

from tqdm import tqdm
if __name__ == "__main__":
    session = init_session()

    # Load tickers from CSV
    ticker_df = pd.read_csv("data/sp500_tickers.csv", header=None)
    tickers = ticker_df[0].tolist()

    all_data = {}
    for i, ticker in tqdm(enumerate(tickers)):
        try:
            dates = get_px_last(f'{ticker} US EQUITY', session)
            all_data[ticker] = dates
        except Exception as e:
            print(f"Failed to get data for {ticker}: {e}")
            all_data[ticker] = []

    # Convert to DataFrame (columns = tickers, rows = earninbgs dates)
    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in all_data.items()]))
    df.to_csv('data/bbg_prices.csv')
    print('Data saved to data/bbg_prices.csv')
