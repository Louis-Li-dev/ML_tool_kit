import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, Tuple


def _plot_data(
        time=None, 
        data=None, 
        splits=None, 
        figsize=(12, 6), 
        title="title"
    ):
    """
    Plots the generated data with optional split regions colored differently.

    Parameters:
    - time (np.ndarray, optional): Time indices.
    - data (np.ndarray): Sequential data.
    - splits (dict, optional): Dictionary containing split indices.
        Example:
            splits = {
                'Train': (start_index, end_index),
                'Validation': (start_index, end_index),
                'Test': (start_index, end_index)
            }
    - figsize (tuple): Size of the plot.
    - title (str): Title of plot, default to 'title'
    """
    if time is None and data is not None:
        time = np.arange(len(data))
    elif data is None or time is None:
        raise ValueError("Data not generated yet.")
    
    plt.figure(figsize=figsize)
    
    # Plot the entire data
    plt.plot(time, data, color='gray', label='All Data')
    
    # Define colors for splits
    colors = {
        'Train': 'blue',
        'Validation': 'orange',
        'Test': 'green'
    }
    
    # Plot each split
    if splits:
        for split_name, (start, end) in splits.items():
            plt.plot(time[start:end], data[start:end], color=colors.get(split_name, 'black'), label=split_name)
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

class DummySequentialData:
    def __init__(self, length: int = 1000, seasonal_period: int = 50, trend: float = 0.1, noise_level: float = 1.0, random_seed: Optional[int] = None):
        """
        Initializes the DummySequentialData object.

        Parameters:
        - length (int): Number of time steps.
        - seasonal_period (int): The period of the seasonal component.
        - trend (float): The slope of the linear trend.
        - noise_level (float): Standard deviation of the Gaussian noise.
        - random_seed (int, optional): Seed for random number generator for reproducibility.
        """
        self.length = length
        self.seasonal_period = seasonal_period
        self.trend = trend
        self.noise_level = noise_level
        self.random_seed = random_seed
        self.data = None
        self.time = None
        self.generate_data()
    
    def generate_data(self):
        """
        Generates the sequential data with trend, seasonality, and noise.
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        self.time = np.arange(self.length)
        
        # Trend component
        trend_component = self.trend * self.time
        
        # Seasonal component using sine wave
        seasonal_component = 10 * np.sin(2 * np.pi * self.time / self.seasonal_period)
        
        # Noise component
        noise_component = np.random.normal(0, self.noise_level, self.length)
        
        # Combine all components
        self.data = trend_component + seasonal_component + noise_component
    
    def get_data(self) -> np.ndarray:
        """
        Returns the generated data.

        Returns:
        - data (np.ndarray): The generated sequential data.
        """
        return self.data
    
    def plot_data(
        self, 
        val_split: Optional[Union[float, int]] = None, 
        test_split: Optional[Union[float, int]] = None,
        figsize: Tuple[int, int] = (12, 6)
    ):
        """
        Plots the generated data with optional training, validation, and testing splits colored differently.

        Parameters:
        - val_split (Optional[Union[float, int]]): Validation split ratio or index.
        - test_split (Optional[Union[float, int]]): Testing split ratio or index.
        - figsize (tuple): Size of the plot.
        """
        splits = {}
        total_length = self.length
        train_end = total_length  # Initialize to total length
        
        # Determine test split
        if test_split is not None:
            if isinstance(test_split, float):
                if not 0 < test_split < 1:
                    raise ValueError("test_split as float must be between 0 and 1.")
                test_size = int(total_length * test_split)
            elif isinstance(test_split, int):
                if not 0 <= test_split < total_length:
                    raise ValueError("test_split as int must be within the data range.")
                test_size = total_length - test_split
            else:
                raise TypeError("test_split must be either float or int.")
            
            test_start = total_length - test_size
            splits['Test'] = (test_start, total_length)
            train_end = test_start
        else:
            test_start = None
        
        # Determine validation split
        if val_split is not None:
            if isinstance(val_split, float):
                if not 0 < val_split < 1:
                    raise ValueError("val_split as float must be between 0 and 1.")
                val_size = int(train_end * val_split)
            elif isinstance(val_split, int):
                if not 0 <= val_split < train_end:
                    raise ValueError("val_split as int must be within the training data range.")
                val_size = train_end - val_split
            else:
                raise TypeError("val_split must be either float or int.")
            
            val_start = train_end - val_size
            splits['Validation'] = (val_start, train_end)
            train_end = val_start
        else:
            val_start = None
        
        # Define training split
        splits['Train'] = (0, train_end)
        
        # Plot using the helper function
        _plot_data(self.time, self.data, splits=splits, figsize=figsize, title='Dummy Sequential Data with Seasonal Patterns')
