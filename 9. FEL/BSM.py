import numpy as np
from scipy import stats
from math import log, sqrt, exp
import AssetModels

def d1(S0, K, T, r, sigma, q=0):
    if T == 0:
        return float('inf') if S0 > K else float('-inf') if S0 < K else 0.0
    return (log(S0 / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))

def d2(S0, K, T, r, sigma, q=0):
    return d1(S0, K, T, r, sigma, q) - sigma * sqrt(T) if T != 0 else 0

def delta(S0, K, T, r, sigma, option_type, q=0):
    d1_val = d1(S0, K, T, r, sigma, q)
    if option_type == 'call':
        return stats.norm.cdf(d1_val)
    elif option_type == 'put':
        return stats.norm.cdf(d1_val) - 1

def gamma(S0, K, T, r, sigma):
    d1_val = d1(S0, K, T, r, sigma)
    return stats.norm.pdf(d1_val) / (S0 * sigma * sqrt(T))

def vega(S0, K, T, r, sigma):
    d1_val = d1(S0, K, T, r, sigma)
    return S0 * stats.norm.pdf(d1_val) * sqrt(T)

def theta(S0, K, T, r, sigma, option_type):
    d1_val = d1(S0, K, T, r, sigma)
    d2_val = d2(S0, K, T, r, sigma)
    return -S0 * stats.norm.pdf(d1_val) * sigma / (2 * sqrt(T)) - r * K * exp(-r * T) * stats.norm.cdf(d2_val) if option_type == 'call' else -S0 * stats.norm.pdf(d1_val) * sigma / (2 * sqrt(T)) + r * K * exp(-r * T) * stats.norm.cdf(-d2_val)

def rho(S0, K, T, r, sigma, option_type):
    d2_val = d2(S0, K, T, r, sigma)
    return K * T * exp(-r * T) * stats.norm.cdf(d2_val) if option_type == 'call' else -K * T * exp(-r * T) * stats.norm.cdf(-d2_val)

def price(S0, K, T, r, sigma, option_type, q=0):
    if option_type not in ['call', 'put']:
        raise ValueError("option_type must be 'call' or 'put'")
    if sigma <= 0 or T < 0 or K <= 0 or S0 <= 0:
        raise ValueError("Invalid parameters")
    d1_val = d1(S0, K, T, r, sigma, q)
    d2_val = d2(S0, K, T, r, sigma, q)
    if option_type == 'call':
        return S0 * exp(-q * T) * stats.norm.cdf(d1_val) - K * exp(-r * T) * stats.norm.cdf(d2_val)
    elif option_type == 'put':
        return K * exp(-r * T) * stats.norm.cdf(-d2_val) - S0 * exp(-q * T) * stats.norm.cdf(-d1_val)

def delta_hedge(S0, K, T, r, sigma, option_type, mu, dt, option_pos, path=None, sigma_h=None, sigma_a=None):
    sigma_h = sigma_h if sigma_h is not None else sigma
    sigma_a = sigma_a if sigma_a is not None else sigma

    S = AssetModels.GBM(S0, mu, sigma_a, T, dt) if path is None else np.array(path)
    option_price = price(S0, K, T, r, sigma, option_type)
    N = len(S)
    t = np.linspace(0, T, N)
    T_i = T - t

    d1_vals = np.where(
        T_i[:-1] == 0,
        np.where(S[:-1] > K, float('inf'), np.where(S[:-1] < K, float('-inf'), 0.0)),
        (np.log(S[:-1] / K) + (r + 0.5 * sigma_h ** 2) * T_i[:-1]) / (sigma_h * np.sqrt(T_i[:-1]))
    )
    d1_vals = np.append(
        d1_vals, 
        float('inf') if S[-1] > K else float('-inf') if S[-1] < K else 0.0
    )

    delta_vals = stats.norm.cdf(d1_vals) if option_type == 'call' else stats.norm.cdf(d1_vals) - 1

    delta_diff = np.diff(delta_vals, prepend=0)
    cashflows = delta_diff * option_pos * S
    comp_factors = np.exp(r * (T - t)) - 1
    interests = cashflows * comp_factors
    pnl = np.sum(interests) + np.sum(cashflows)

    if option_type == 'call' and S[-1] > K:
        pnl -= K * option_pos
    elif option_type == 'put' and S[-1] < K:
        pnl += K * option_pos

    return pnl * np.exp(-r * T) - option_price * option_pos

def dh_path(S0, K, T, r, sigma, option_type, mu, dt, option_pos, path=None, sigma_h=None, sigma_a=None):
    sigma_h = sigma_h if sigma_h is not None else sigma
    sigma_a = sigma_a if sigma_a is not None else sigma

    S = AssetModels.GBM(S0, mu, sigma_a, T, dt) if path is None else np.array(path)
    option_price = price(S0, K, T, r, sigma, option_type)
    N = len(S)
    t = np.linspace(0, T, N)
    T_i = T - t

    d1_vals = np.where(
        T_i[:-1] == 0,
        np.where(S[:-1] > K, float('inf'), np.where(S[:-1] < K, float('-inf'), 0.0)),
        (np.log(S[:-1] / K) + (r + 0.5 * sigma_h ** 2) * T_i[:-1]) / (sigma_h * np.sqrt(T_i[:-1]))
    )
    d1_vals = np.append(
        d1_vals, 
        float('inf') if S[-1] > K else float('-inf') if S[-1] < K else 0.0
    )

    delta_vals = stats.norm.cdf(d1_vals) if option_type == 'call' else stats.norm.cdf(d1_vals) - 1

    delta_diff = np.diff(delta_vals, prepend=0)
    cashflows = delta_diff * option_pos * S
    comp_factors = np.exp(r * (T - t)) - 1
    interests = cashflows * comp_factors
    pnl_path = np.cumsum(interests + cashflows)

    if option_type == 'call' and S[-1] > K:
        pnl_path[-1] -= K * option_pos
    elif option_type == 'put' and S[-1] < K:
        pnl_path[-1] += K * option_pos

    pnl_path -= option_price * option_pos
    pnl_path *= np.exp(-r * T)

    return pnl_path




def MC_pnl(S0, K, T, r, sigma, option_type, mu, dt, option_pos, nsim=1000, path=None, sigma_h=None, sigma_a=None):
    
    prices = np.zeros(nsim)
    for i in range(nsim):
        prices[i] = delta_hedge(S0=S0, K=K, T=T, r=r, sigma=sigma, option_type=option_type, mu=mu, dt=dt, option_pos=option_pos, sigma_h=sigma_h, sigma_a=sigma_a)
    
    return np.mean(prices)





















# ################## WORK IN PROGRESS ################
# def gamma_hedge(S0, K, T, r, sigma, option_type, mu, dt, option_pos, path=None, sigma_h=None, sigma_a=None):
    
#     sigma_h = sigma_h if sigma_h is not None else sigma
#     sigma_a = sigma_a if sigma_a is not None else sigma

#     S = AssetModels.GBM(S0, mu, sigma_a, T, dt) if path is None else np.array(path)
#     N = len(S)
#     t = np.linspace(0, T, N)
#     T_i = T - t

#     hedge_K = S0

#     d1_vals = np.where(
#         T_i[:-1] == 0,
#         np.where(S[:-1] > K, float('inf'), np.where(S[:-1] < K, float('-inf'), 0.0)),
#         (np.log(S[:-1] / K) + (r + 0.5 * sigma_h ** 2) * T_i[:-1]) / (sigma_h * np.sqrt(T_i[:-1]))
#     )
#     d1_vals = np.append(
#         d1_vals, 
#         float('inf') if S[-1] > K else float('-inf') if S[-1] < K else 0.0
#     )

#     d1_vals_hedge = np.where(
#         T_i[:-1] == 0,
#         np.where(S[:-1] > hedge_K, float('inf'), np.where(S[:-1] < hedge_K, float('-inf'), 0.0)),
#         (np.log(S[:-1] / hedge_K) + (r + 0.5 * sigma_h ** 2) * T_i[:-1]) / (sigma_h * np.sqrt(T_i[:-1]))
#     )
#     d1_vals_hedge = np.append(
#         d1_vals_hedge, 
#         float('inf') if S[-1] > hedge_K else float('-inf') if S[-1] < hedge_K else 0.0
#     )

#     delta_vals = stats.norm.cdf(d1_vals) if option_type == 'call' else stats.norm.cdf(d1_vals) - 1
#     delta_vals_hedge = stats.norm.cdf(d1_vals_hedge) if option_type == 'call' else stats.norm.cdf(d1_vals_hedge) - 1

#     gamma_vals = np.where(
#         (T_i > 0) & (S > 0), 
#         stats.norm.pdf(d1_vals) / (S * sigma_h * np.sqrt(T_i)),
#         0.0  # Gamma is 0 if T_i or S is 0
#     )

#     gamma_vals_hedge = np.where(
#         (T_i > 0) & (S > 0), 
#         stats.norm.pdf(d1_vals_hedge) / (S * sigma_h * np.sqrt(T_i)),
#         0.0
#     )

#     gamma_hedge_units = np.where(
#         gamma_vals_hedge == 0,
#         0.0,  # Avoid division by zero
#         - (gamma_vals * option_pos) / gamma_vals_hedge
#     )

#     total_delta = delta_vals * option_pos + delta_vals_hedge * gamma_hedge_units

#     delta_diff = np.diff(total_delta, prepend=0)

#     cashflows = delta_diff * S

#     comp_factors = np.exp(r * (T - t)) - 1
#     interests = cashflows * comp_factors

#     pnl = np.sum(interests) + np.sum(cashflows)

#     if option_type == 'call' and S[-1] > K:
#         pnl -= K * option_pos
#     elif option_type == 'put' and S[-1] < K:
#         pnl += K * option_pos

#     return pnl * np.exp(-r * T) - price(S0, K, T, r, sigma, option_type) * option_pos
# ################################################
