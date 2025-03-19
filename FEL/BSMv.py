import numpy as np
from scipy import stats
import AssetModels

def d1(S0, K, T, r, sigma):
    """
    Vectorized d1 calculation for Black-Scholes-Merton model.
    Handles scalar and array inputs for all parameters.
    
    Returns inf/-inf for T=0 based on moneyness (S0 vs K).
    """
    # Handle T=0 case first
    with np.errstate(divide='ignore'):
        d1_val = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    # Replace values where T=0
    is_zero_T = (T == 0)
    if np.any(is_zero_T):
        d1_val = np.where(is_zero_T,
                         np.where(S0 > K, np.inf,
                                np.where(S0 < K, -np.inf, 0.0)),
                         d1_val)
    return d1_val

def d2(S0, K, T, r, sigma):
    """
    Vectorized d2 calculation. Uses d1 result and adjusts for volatility term.
    """
    return d1(S0, K, T, r, sigma) - sigma * np.sqrt(T)

def price(S0, K, T, r, sigma, option_type):
    """
    Vectorized option price calculation.
    Parameters can be scalars or arrays (must be broadcastable).
    
    option_type: 'call' or 'put'
    """
    if np.any(sigma <= 0) or np.any(T < 0) or np.any(K <= 0) or np.any(S0 <= 0):
        raise ValueError("Invalid parameters")
        
    d1_val = d1(S0, K, T, r, sigma)
    d2_val = d2(S0, K, T, r, sigma)
    
    if option_type == 'call':
        return S0 * stats.norm.cdf(d1_val) - K * np.exp(-r * T) * stats.norm.cdf(d2_val)
    elif option_type == 'put':
        return K * np.exp(-r * T) * stats.norm.cdf(-d2_val) - S0 * stats.norm.cdf(-d1_val)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

def delta(S0, K, T, r, sigma, option_type):
    """
    Vectorized delta calculation.
    """
    d1_val = d1(S0, K, T, r, sigma)
    if option_type == 'call':
        return stats.norm.cdf(d1_val)
    elif option_type == 'put':
        return stats.norm.cdf(d1_val) - 1

def gamma(S0, K, T, r, sigma):
    """
    Vectorized gamma calculation.
    """
    d1_val = d1(S0, K, T, r, sigma)
    return stats.norm.pdf(d1_val) / (S0 * sigma * np.sqrt(T))

def vega(S0, K, T, r, sigma):
    """
    Vectorized vega calculation.
    """
    d1_val = d1(S0, K, T, r, sigma)
    return S0 * stats.norm.pdf(d1_val) * np.sqrt(T)

def theta(S0, K, T, r, sigma, option_type):
    """
    Vectorized theta calculation.
    """
    d1_val = d1(S0, K, T, r, sigma)
    d2_val = d2(S0, K, T, r, sigma)
    disc_factor = r * K * np.exp(-r * T)
    
    common_term = -S0 * stats.norm.pdf(d1_val) * sigma / (2 * np.sqrt(T))
    
    if option_type == 'call':
        return common_term - disc_factor * stats.norm.cdf(d2_val)
    else:  # put
        return common_term + disc_factor * stats.norm.cdf(-d2_val)

def rho(S0, K, T, r, sigma, option_type):
    """
    Vectorized rho calculation.
    """
    d2_val = d2(S0, K, T, r, sigma)
    disc_factor = K * T * np.exp(-r * T)
    
    if option_type == 'call':
        return disc_factor * stats.norm.cdf(d2_val)
    else:  # put
        return -disc_factor * stats.norm.cdf(-d2_val)

def delta_hedge(S0, K, T, r, sigma, option_type, mu, dt, option_pos, path=None, sigma_h=None, sigma_a=None):
    """
    Vectorized delta hedging simulation.
    Can handle both single paths and arrays of paths.
    
    Returns PnL for each path.
    """
    sigma_h = sigma if sigma_h is None else sigma_h
    sigma_a = sigma if sigma_a is None else sigma_a
    
    # Generate or use provided path(s)
    S = AssetModels.GBM(S0, mu, sigma_a, T, dt) if path is None else np.asarray(path)
    if S.ndim == 2:
        S = S.squeeze()
    
    # Calculate initial option price
    option_price = price(S0, K, T, r, sigma, option_type)
    
    # Time grid
    N = len(S)
    t = np.linspace(0, T, N)
    T_remaining = T - t
    
    # Calculate deltas along the path
    d1_vals = d1(S[:-1], K, T_remaining[:-1], r, sigma_h)
    d1_vals = np.append(d1_vals, 
                       np.where(S[-1] > K, np.inf,
                              np.where(S[-1] < K, -np.inf, 0.0)))
    
    delta_vals = stats.norm.cdf(d1_vals) if option_type == 'call' else stats.norm.cdf(d1_vals) - 1
    
    # Calculate PnL components
    delta_diff = np.diff(delta_vals, prepend=0)
    cashflows = delta_diff * option_pos * S
    comp_factors = np.exp(r * T_remaining) - 1
    interests = cashflows * comp_factors
    
    # Calculate final PnL
    pnl = np.sum(interests) + np.sum(cashflows)
    
    # Adjust for option payoff
    if option_type == 'call':
        pnl -= np.where(S[-1] > K, K * option_pos, 0)
    else:  # put
        pnl += np.where(S[-1] < K, K * option_pos, 0)
    
    return pnl * np.exp(-r * T) - option_price * option_pos

def dh_path(S0, K, T, r, sigma, option_type, mu, dt, option_pos, path=None, sigma_h=None, sigma_a=None):
    """
    Vectorized delta hedging simulation that returns the full PnL path.
    Can handle both single paths and arrays of paths.
    
    Returns the cumulative PnL at each time step.
    """
    sigma_h = sigma if sigma_h is None else sigma_h
    sigma_a = sigma if sigma_a is None else sigma_a
    
    # Generate or use provided path(s)
    S = AssetModels.GBM(S0, mu, sigma_a, T, dt) if path is None else np.asarray(path)
    if S.ndim == 2:
        S = S.squeeze()
    
    # Calculate initial option price
    option_price = price(S0, K, T, r, sigma, option_type)
    
    # Time grid
    N = len(S)
    t = np.linspace(0, T, N)
    T_remaining = T - t
    
    # Calculate deltas along the path
    d1_vals = d1(S[:-1], K, T_remaining[:-1], r, sigma_h)
    d1_vals = np.append(d1_vals, 
                       np.where(S[-1] > K, np.inf,
                              np.where(S[-1] < K, -np.inf, 0.0)))
    
    delta_vals = stats.norm.cdf(d1_vals) if option_type == 'call' else stats.norm.cdf(d1_vals) - 1
    
    # Calculate PnL components
    delta_diff = np.diff(delta_vals, prepend=0)
    cashflows = delta_diff * option_pos * S
    comp_factors = np.exp(r * T_remaining) - 1
    interests = cashflows * comp_factors
    
    # Calculate cumulative PnL path
    pnl_path = np.cumsum(interests + cashflows)
    
    # Adjust final PnL for option payoff
    if option_type == 'call':
        pnl_path[-1] -= np.where(S[-1] > K, K * option_pos, 0)
    else:  # put
        pnl_path[-1] += np.where(S[-1] < K, K * option_pos, 0)
    
    # Apply discount factor and subtract initial option premium
    pnl_path = pnl_path * np.exp(-r * T) - option_price * option_pos
    
    return pnl_path