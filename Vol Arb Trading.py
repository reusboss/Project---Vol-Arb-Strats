import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import bisect, brentq
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def Black76_call(sigma, forward, risk_free, expiry, strike_price):
    sd = sigma * np.sqrt(expiry)
    d1 = np.log(forward / strike_price)
    d2 = d1 - sd
    return np.exp(-risk_free * expiry) * (forward * norm.cdf(d1) - strike_price * norm.cdf(d2))


def Black76_put(sigma, forward, risk_free, expiry, strike_price):
    sd = sigma * np.sqrt(expiry)
    d1 = np.log(forward / strike_price)
    d2 = d1 - sd
    return np.exp(-risk_free * expiry) * (strike_price * norm.cdf(-d2) - forward * norm.cdf(-d1))



def IVofCall(call_price, forward, risk_free, strike_price, expiry):
    lowerbound = np.max([0,(forward-strike_price) * np.exp(-risk_free * expiry)])
    if call_price < lowerbound:
        return np.nan
    if call_price == lowerbound:
        return 0
    if call_price >= forward * np.exp(-risk_free * expiry):
        return np.nan
    hi = 0.2
    while Black76_call(hi, forward, risk_free, expiry, strike_price) > call_price:
        hi = hi / 2
    while Black76_call(hi, forward, risk_free, expiry, strike_price) < call_price:
        hi = hi * 2
    lo = hi / 2
    IV = bisect(lambda x: Black76_call(x, forward, risk_free, expiry, strike_price) - call_price, lo, hi)
    return IV


def IVofPut(put_price, forward, risk_free, strike_price, expiry):
    lowerbound = np.max([0,(strike_price-forward) * np.exp(-risk_free * expiry)])
    if put_price < lowerbound:
        return np.nan
    if put_price == lowerbound:
        return 0
    if put_price >= forward * np.exp(-risk_free * expiry):
        return np.nan
    hi = 0.2
    while Black76_put(hi, forward, risk_free, expiry, strike_price) > put_price:
        hi = hi / 2
    while Black76_put(hi, forward, risk_free, expiry, strike_price) < put_price:
        hi = hi * 2
    lo = hi / 2
    IV = bisect(lambda x: Black76_put(x, forward, risk_free, expiry, strike_price) - put_price, lo, hi)
    return IV



if __name__ == "__main__":
    print(IVofCall(33.3, 1132.759389, 0.02, 1100, 4/252))
    print(IVofPut(0.325, 1132.759389, 0.02, 1050, 4/365))