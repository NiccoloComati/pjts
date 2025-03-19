import numpy as np
import pandas as pd
import statsmodels.api as sm

def CAPM(rets, factors):
    rf = factors[:, 0]
    mkt = factors[:, 1]
    X = sm.add_constant(mkt)
    xrets = rets - rf
    model = sm.OLS(xrets, X).fit()
    return model

def FF3(rets, factors):
    rf = factors[:, 0]
    mkt = factors[:, 1]
    smb = factors[:, 2]
    hml = factors[:, 3]
    X = sm.add_constant(np.column_stack((mkt, smb, hml)))
    xrets = rets - rf
    model = sm.OLS(xrets, X).fit()
    return model

def FF5(rets, factors):
    rf = factors[:, 0]
    mkt = factors[:, 1]
    smb = factors[:, 2]
    hml = factors[:, 3]
    rmw = factors[:, 4]
    cma = factors[:, 5]
    X = sm.add_constant(np.column_stack((mkt, smb, hml, rmw, cma)))
    xrets = rets - rf
    model = sm.OLS(xrets, X).fit()
    return model

def signal_cleaner(rets, factors, signal):
    rf = factors[:, 0]
    mkt = factors[:, 1]
    smb = factors[:, 2]
    hml = factors[:, 3]
    signal = signal
    X = sm.add_constant(np.column_stack((mkt, smb, hml, signal)))
    xrets = rets - rf
    model = sm.OLS(xrets, X).fit()
    return model

################# DOUBLE SORT FUNCTION IN PROGRESS #################
# def double_sort(array):