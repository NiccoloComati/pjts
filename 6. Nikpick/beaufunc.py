from typing import Union, Optional
import pandas as pd

def seasonality_plot(data: Union[pd.DataFrame, pd.Series], 
                    date_column: str = None,
                    value_column: str = None,
                    freq1: str = 'month',
                    freq2: Optional[str] = None,
                    figsize: tuple = (12, 6),
                    title: str = None,
                    color_palette: str = 'husl',
                    grid: bool = True,
                    errorbar: bool = False,
                    errorbar_ci: float = 'sd') -> None:
    """
    Create a seasonality plot for time series data with flexible frequency options.
    
    Parameters:
    -----------
    data : Union[pd.DataFrame, pd.Series]
        Input data. If DataFrame, must specify date_column and value_column.
        If Series, must have datetime index.
    date_column : str, optional
        Name of the date column if data is DataFrame
    value_column : str, optional
        Name of the value column if data is DataFrame
    freq1 : str
        Primary frequency to plot on x-axis
        Options: 'year', 'month', 'day', 'weekday', 'week', 'quarter', 'hour'
    freq2 : str, optional
        Secondary frequency to create separate lines
        Same options as freq1
    figsize : tuple
        Figure size as (width, height)
    title : str, optional
        Plot title. If None, will be auto-generated
    color_palette : str
        Seaborn color palette name
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Convert to DataFrame if Series
    if isinstance(data, pd.Series):
        df = data.to_frame(name='value')
        df.index.name = 'date'
        df = df.reset_index()
        date_column = 'date'
        value_column = 'value'
    else:
        df = data.copy()
    
    # Ensure date column is datetime
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Dictionary of datetime attributes and their ranges
    freq_ranges = {
        'year': None,
        'month': 12,
        'day': 31,
        'weekday': 7,
        'week': 53,
        'quarter': 4,
        'hour': 24
    }
    
    # Validate frequencies
    if freq1 not in freq_ranges:
        raise ValueError(f"freq1 must be one of {list(freq_ranges.keys())}")
    if freq2 and freq2 not in freq_ranges:
        raise ValueError(f"freq2 must be one of {list(freq_ranges.keys())}")
    
    # Extract datetime components
    df[freq1] = getattr(df[date_column].dt, freq1)
    if freq2:
        df[freq2] = getattr(df[date_column].dt, freq2)
    
    # Group by frequencies
    if freq2:
        grouped = df.groupby([freq1, freq2])[value_column].mean().reset_index()
    else:
        grouped = df.groupby([freq1])[value_column].mean().reset_index()
    
    # Create plot
    plt.figure(figsize=figsize)
    if freq2:
        sns.lineplot(
            data=grouped,
            x=freq1,
            y=value_column,
            hue=freq2,
            marker='o',
            palette=color_palette,
            ci=None if errorbar else errorbar_ci
        )
    else:
        sns.lineplot(
            data=grouped,
            x=freq1,
            y=value_column,
            marker='o',
            ci=None if errorbar else errorbar_ci
        )
    
    # Set title
    if title is None:
        if freq2:
            title = f'Average {value_column} by {freq1} and {freq2}'
        else:
            title = f'Average {value_column} by {freq1}'
    plt.title(title)
    
    # Customize x-axis ticks if applicable
    if freq_ranges[freq1]:
        plt.xticks(range(0, freq_ranges[freq1]))
    
    if grid:
        plt.grid(True, alpha=0.3)
    else:
        plt.grid(False)
    plt.xlabel(freq1.capitalize())
    plt.ylabel(f'Average {value_column}')
    
    if freq2:
        plt.legend(title=freq2.capitalize())
    
    plt.show()


def periodogram_plot(ts, freq='D', min_freq=0.5, detrend='linear', figsize=(12, 6)):
    """
    Compute and plot periodogram for time series data with customizable frequency.
    
    Parameters:
    -----------
    ts : array-like
        Time series data to analyze
    freq : str, default='D'
        Frequency of the data. Common options:
        - 'D': Daily
        - 'W': Weekly
        - 'M': Monthly
        - 'Q': Quarterly
        - 'Y': Yearly
    min_freq : float, default=0.5
        Minimum frequency to display in the plot
    detrend : str, default='linear'
        The type of detrending: 'linear', 'constant', or False
    figsize : tuple, default=(12, 6)
        Figure size for the plot
    
    Returns:
    --------
    tuple: (frequencies, spectrum, axis)
        frequencies: array of frequencies
        spectrum: array of spectrum values
        axis: matplotlib axis object
    """
    from scipy.signal import periodogram
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Define frequency scaling based on input frequency
    freq_dict = {
        'D': 365,                                          # Daily data
        'FD': 252,                                         # Financial daily data
        'W': 52,                                           # Weekly data
        'M': 12,                                           # Monthly data
        'Q': 4,                                            # Quarterly data
        'Y': 1                                             # Yearly data
    }
    
    if freq not in freq_dict:
        raise ValueError(f"Frequency '{freq}' not supported. Use one of: {list(freq_dict.keys())}")
    
    fs = freq_dict[freq]
    
    # Compute periodogram
    frequencies, spectrum = periodogram(ts, 
                                     fs=fs,
                                     detrend=detrend,
                                     window="boxcar",
                                     scaling='spectrum')
    
    # Create DataFrame and filter frequencies
    periodogram_df = pd.DataFrame({'Frequency': frequencies, 'Spectrum': spectrum})
    periodogram_df = periodogram_df[periodogram_df['Frequency'] > min_freq]
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the spectrum
    sns.lineplot(data=periodogram_df, x='Frequency', y='Spectrum', color='purple', ax=ax)
    
    # Customize plot
    ax.set_xscale('log')
    
    # Define tick positions and labels based on data frequency
    tick_positions = [1, 2, 4, 6, 12, 26, 52, 104]
    tick_labels = [
        "Annual (1)", 
        "Semiannual (2)", 
        "Quarterly (4)", 
        "Bimonthly (6)", 
        "Monthly (12)", 
        "Biweekly (26)", 
        "Weekly (52)", 
        "Semiweekly (104)"
    ]
    
    # Filter ticks based on data frequency
    max_tick = max(frequencies)
    valid_ticks = [(pos, label) for pos, label in zip(tick_positions, tick_labels) 
                   if pos <= max_tick]
    valid_positions, valid_labels = zip(*valid_ticks) if valid_ticks else ([], [])
    
    ax.set_xticks(valid_positions)
    ax.set_xticklabels(valid_labels, rotation=30)
    
    # Set labels and title
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Variance')
    ax.set_title('Periodogram')
    
    # Add grid for better readability
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    
    # Format y-axis to use scientific notation
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    
    plt.tight_layout()
    
    return frequencies, spectrum, ax