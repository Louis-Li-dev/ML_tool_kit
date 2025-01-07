import numpy as np
class LogTransformer:
    def __init__(self):
        """
        - Note: for training dataset and testing dataset alike, fit is required.
        """
        pass
    
    def fit(self, x):
        """
        Fit the transformer by calculating the offset based on the input data.
        The offset ensures that all values are shifted to be greater than zero for the log transformation.

        Args:
            x (array-like): Input data.
        """
        self.offset = np.min(x) - 1  # Ensure all values > 0
        self.fitted = True

    def __transform(self, x):
        """
        Apply the log transformation to the input data using the fitted offset.

        Args:
            x (array-like): Input data.

        Returns:
            np.ndarray: Log-transformed data.
        """
        if self.fitted:
            return np.log(x - self.offset)
        else:
            raise ValueError("The transformer must be fitted before calling transform.")

    def fit_transform(self, x):
        """
        Fit the transformer and apply the log transformation in one step.

        Args:
            x (array-like): Input data.

        Returns:
            np.ndarray: Log-transformed data.
        """
        self.fit(x)
        return self.__transform(x)

    def inverse_transform(self, y):
        """
        Apply the inverse of the log transformation to the data.

        Args:
            y (array-like): Log-transformed data.

        Returns:
            np.ndarray: Original data before log transformation.
        """
        if self.fitted:
            return np.exp(y) + self.offset
        else:
            raise ValueError("The transformer must be fitted before calling inverse_transform.")
    
import numpy as np

class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        """
        Initializes the MinMaxScaler with the specified feature range.
        
        Args:
            feature_range (tuple): Desired range of transformed data (min, max).
        """
        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None
        self.range_ = None
        self.fitted = False

    def fit(self, x):
        """
        Compute the min and max to be used for later scaling.

        Args:
            x (array-like): Input data to compute the scaling parameters.
        """
        self.min_ = np.min(x)
        self.max_ = np.max(x)
        self.range_ = self.max_ - self.min_
        if self.range_ == 0:
            raise ValueError("All values in the input data are identical. Cannot scale.")
        self.fitted = True

    def transform(self, x):
        """
        Scale the input data to the specified feature range.

        Args:
            x (array-like): Input data to scale.

        Returns:
            np.ndarray: Scaled data.
        """
        if not self.fitted:
            raise ValueError("The scaler must be fitted before calling transform.")
        
        scale_min, scale_max = self.feature_range
        return (x - self.min_) / self.range_ * (scale_max - scale_min) + scale_min

    def fit_transform(self, x):
        """
        Fit the scaler and transform the input data in one step.

        Args:
            x (array-like): Input data to fit and scale.

        Returns:
            np.ndarray: Scaled data.
        """
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x_scaled):
        """
        Inverse transform the scaled data back to the original range.

        Args:
            x_scaled (array-like): Scaled data.

        Returns:
            np.ndarray: Original data.
        """
        if not self.fitted:
            raise ValueError("The scaler must be fitted before calling inverse_transform.")
        
        scale_min, scale_max = self.feature_range
        return (x_scaled - scale_min) / (scale_max - scale_min) * self.range_ + self.min_
