import numpy as np

def GARCH_model(data):
    from arch import arch_model
    if data is None or len(data) == 0:
        raise ValueError("Input data cannot be None or empty.")
    
    data_wip = np.asarray(data).astype(np.float64)
    
    if np.any(np.isnan(data_wip)):
        raise ValueError("Input data contains NaNs. Please clean the data before passing it to the model.")
    
    try:
        garch_model = arch_model(data_wip, mean='AR', lags=1, vol='Garch', p=1, q=1, rescale=False).fit(update_freq=5)
    except Exception as e:
        raise RuntimeError(f"An error occurred while fitting the GARCH model: {e}")
    
    return garch_model

def GARCH_forecast(data, prices = False, ascending = False, steps = 1):
    
    garch_model = GARCH_model(data, prices, ascending)
    forecast = garch_model.forecast(horizon=steps)
    return forecast.variance[-1:]