import statsmodels as sm
import pandas as pd
import numpy as np
from itertools import permutations
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map


PRICES = pd.read_csv('data/historical_prices.csv')

def load_tickers(filepath="data/sp500_tickers.csv"):
    with open(filepath) as f:
        return [line.strip() for line in f if line.strip()]

TICKS = load_tickers()

if __name__ == "__main__":
    pairs = permutations(TICKS,2)
    for i in pairs:
        print(i)