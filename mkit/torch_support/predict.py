import numpy as np
import torch
from typing import Union, Callable
from tqdm import tqdm

def autoregressive(
    x: Union[np.ndarray, torch.Tensor],  # Initial input sequence
    model: Union[Callable, torch.nn.Module],  # Model to predict with
    steps_ahead: int = 100,  # Number of steps to predict ahead
    look_back: int = 100,
    device: torch.device = torch.device("cpu"),  # Device for computation (default: CPU)
) -> Union[np.ndarray, torch.Tensor]:
    """
    Generates autoregressive predictions using the provided model.
    
    Parameters:
    - x (Union[np.ndarray, torch.Tensor]): The initial input sequence.
    - model (Union[Callable, torch.nn.Module]): The model for prediction.
        - Uses `predict` if available, otherwise `forward`.
    - steps_ahead (int, optional): Number of steps to predict ahead. Default is 100.
    - look_back (int, optional): Number of steps to refer back. Default is 100.
    - device (torch.device, optional): The device to use for computation. Default is "cpu".
    
    Returns:
    - Union[np.ndarray, torch.Tensor]: The extended sequence with autoregressive predictions.
    """
    # Ensure input is a NumPy array or Torch Tensor
    if not isinstance(x, (np.ndarray, torch.Tensor)):
        raise TypeError("Input x must be a NumPy array or PyTorch Tensor.")

    # Convert NumPy input to Torch Tensor if necessary
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32, device=device)
    else:
        x = x.to(device)
    

    # Move model to the specified device if it's a PyTorch model
    if isinstance(model, torch.nn.Module):
        model.to(device)

    # Prepare the sequence for autoregressive predictions
    sequence = x.tolist()  # Convert initial input to a list for easy manipulation
    result = []
    # Determine the prediction method
    if hasattr(model, 'predict') and callable(getattr(model, 'predict')):
        predict_fn = model.predict
    elif hasattr(model, 'forward') and callable(getattr(model, 'forward')):
        predict_fn = model.forward
    else:
        raise AttributeError("Model must have either a 'predict' or 'forward' method.")

    # Autoregressive prediction loop
    for _ in tqdm(range(steps_ahead)):
        # Prepare the last input window (as a batch)
        input_data = torch.tensor(
            sequence[-look_back:],
            dtype=torch.float32, device=device
        ).unsqueeze(0)
        # Predict the next value
        if callable(predict_fn):
            next_value = predict_fn(input_data.unsqueeze(-1))
        else:
            raise ValueError("Unable to determine the prediction method (predict or forward).")
        
        # Ensure next_value is scalar and convert to CPU for appending
        if isinstance(next_value, torch.Tensor):
            next_value = next_value.squeeze(-1).detach().cpu()
        elif isinstance(next_value, np.ndarray):
            next_value = next_value.squeeze()
        
        # Append the predicted value to the sequence
        sequence.append(next_value)
        sequence = sequence[-look_back:]
        result.append(next_value)
        del next_value
    # Convert back to the original format (NumPy array or Torch Tensor)
    
    tqdm.write("Autoregressive Prediction Completed🚀")
    if isinstance(x, torch.Tensor):
        return torch.tensor(result, dtype=torch.float32, device=device)
    else:
        return np.array(result)