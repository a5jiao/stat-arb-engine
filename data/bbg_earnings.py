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

def get_earnings_dates(ticker, session):
    refDataService = session.getService("//blp/refdata")
    request = refDataService.createRequest("ReferenceDataRequest")
    request.append("securities", ticker)
    request.append("fields", "DY895")  # Earnings announcement dates

    session.sendRequest(request)

    earnings_dates = []
    ten_years_ago = datetime.date.today() - timedelta(days=365 * 10)

    while True:
        event = session.nextEvent()
        for msg in event:
            if msg.hasElement("securityData"):
                #print(msg)
                securityDataArray = msg.getElement("securityData")
                for i in range(securityDataArray.numValues()):
                    securityData = securityDataArray.getValueAsElement(i)
                    fieldData = securityData.getElement("fieldData")
                    if fieldData.hasElement("DY895"):
                        dy895Data = fieldData.getElement("DY895")
                        # DY895 can be an array of dates
                        for j in range(dy895Data.numValues()):
                            data = dy895Data.getValueAsElement(j)
                            date = data['Announcement Date']
                            if date < ten_years_ago:
                                break
                            earnings_dates.append(date)
        if event.eventType() == blpapi.Event.RESPONSE:
            break

    earnings_dates.reverse()
    return earnings_dates

if __name__ == "__main__":
    session = init_session()

    # Load tickers from CSV
    ticker_df = pd.read_csv("data/sp500_tickers.csv", header=None)
    tickers = ticker_df[0].tolist()

    all_data = {}
    for i, ticker in tqdm(enumerate(tickers)):
        try:
            print(f"[{i+1}/{len(tickers)}] Getting earnings for {ticker}")
            dates = get_earnings_dates(f'{ticker} US EQUITY', session)
            all_data[ticker] = dates
        except Exception as e:
            print(f"Failed to get data for {ticker}: {e}")
            all_data[ticker] = []

    # Convert to DataFrame (columns = tickers, rows = earnings dates)
    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in all_data.items()]))
    df.to_csv('data/earnings_dates.csv')
    print('Data saved to data/earnings_dates.csv')
