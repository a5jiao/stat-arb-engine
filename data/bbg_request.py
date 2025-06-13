import blpapi
import pandas as pd
import datetime
from blpapi import SessionOptions, Session

def init_session():
    session = Session()
    if not session.start():
        print("Failed to start session")
        return
    if not session.openService("//blp/refdata"):
        print("Failed to open service.")
        return
    return session
