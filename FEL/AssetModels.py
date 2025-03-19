import numpy as np

def GBM(S0, mu, sigma, T, dt):
    N = int(T / dt)
    t = np.array([i * dt for i in range(N + 1)])
    W = np.random.normal(0, 1, N)
    W = np.insert(W, 0, 0)
    W = np.cumsum(W) * np.sqrt(dt)
    S = S0 * np.exp((mu - 0.5 * sigma ** 2) * t + sigma * W)
    return S
    
def BM(S0, mu, sigma, T, dt):
    N = int(T / dt)
    t = np.linspace(0, T, N + 1)
    W = np.random.normal(0, 1, N)
    W = np.insert(W, 0, 0)
    W = np.cumsum(W) * np.sqrt(dt)
    return S0 + mu * t + sigma * W

def OrnsteinUhlenbeck(S0, S_bar, lambda_, sigma, T, dt):
    N = int(T / dt)
    t = np.linspace(0, T, N + 1)
    S = np.zeros(N + 1)
    S[0] = S0
    
    exp_neg_lambda_dt = np.exp(-lambda_ * dt)
    mean_factor = S_bar * (1 - exp_neg_lambda_dt)
    std_dev = sigma * np.sqrt((1 - np.exp(-2 * lambda_ * dt)) / (2 * lambda_))
    
    for i in range(1, N + 1):
        S[i] = S[i-1] * exp_neg_lambda_dt + mean_factor + std_dev * np.random.normal()
    
    return S

def CIR(r0, lambda_, r_bar, sigma, T, dt):
    N = int(T / dt)
    t = np.linspace(0, T, N + 1)
    r = np.zeros(N + 1)
    r[0] = r0
    
    exp_neg_lambda_dt = np.exp(-lambda_ * dt)
    mean_factor = r_bar * (1 - exp_neg_lambda_dt)
    std_dev = sigma * np.sqrt((1 - np.exp(-2 * lambda_ * dt)) / (2 * lambda_))
    
    for i in range(1, N + 1):
        r[i] = r[i-1] * exp_neg_lambda_dt + mean_factor + std_dev * np.sqrt(r[i-1]) * np.random.normal()
    
    return r

def Vasicek(r0, alpha, r_bar, sigma, T, dt):
    N = int(T / dt)
    t = np.linspace(0, T, N + 1)
    r = np.zeros(N + 1)
    r[0] = r0
    
    exp_neg_alpha_dt = np.exp(-alpha * dt)
    mean_factor = r_bar * (1 - exp_neg_alpha_dt)
    std_dev = sigma * np.sqrt((1 - np.exp(-2 * alpha * dt)) / (2 * alpha))
    
    for i in range(1, N + 1):
        r[i] = r[i-1] * exp_neg_alpha_dt + mean_factor + std_dev * np.random.normal()
    
    return r