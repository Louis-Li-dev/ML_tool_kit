import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch_geometric.data import Data
from typing import Union, Tuple, List, Callable, Any, Optional
import numpy as np
from tqdm import tqdm
from torch.utils.data import Subset
from sklearn.model_selection import KFold

class MaskedGraphDataset:
    def __init__(self, data, mask_type):
        """
        A dataset wrapper for PyTorch Geometric Data object, using train, val, or test masks.

        Args:
            data (Data): PyTorch Geometric Data object.
            mask_type (str): One of 'train', 'val', or 'test', indicating the mask to use.
        """
        assert mask_type in ['train', 'val', 'test'], "mask_type must be 'train', 'val', or 'test'"
        self.data = data
        self.mask = getattr(data, f"{mask_type}_mask")  # Boolean mask for nodes

    def __len__(self):
        return self.mask.sum().item()  # Number of nodes in the mask

    def __getitem__(self, idx):
        """
        Returns a subgraph centered on a single node with its features and label.
        
        Args:
            idx (int): Index within the subset (filtered by the mask).

        Returns:
            x (Tensor): Node features.
            y (Tensor): Node label.
            node_idx (int): Original node index in the graph.
        """
        indices = self.mask.nonzero(as_tuple=False).squeeze()
        node_idx = indices[idx]  # Map dataset index to graph node index
        return self.data.x[node_idx], self.data.y[node_idx], node_idx
    


def x_y_sequences(
    data: Union[np.ndarray, torch.Tensor], 
    seq_len: int, 
    forecast_horizon: int = 0,
    return_torch: bool = True,
    output_dtype = torch.float32
    ):
    """
    Creates input and target sequences from the long time series data.
    This is often used in transformer structure.
    
    Args:
      - data (Data): 1D array containing time-series data.
      - seq_len: Length of the sequence window used for training.
      - forecast_horizon: Offset for target sequences. If 0, the target is identical to the input sequence.
      - return_torch (bool) default = True: return data type.
      - output_dtype = torch.float32: output data type.
    
    Returns:
      - x: Tensor of shape (num_sequences, seq_len, 1)
      - y: Tensor of shape (num_sequences, seq_len, 1)
      
    For example, with forecast_horizon > 0:
      x[i] = data[i : i + seq_len]
      y[i] = data[i + forecast_horizon : i + forecast_horizon + seq_len]
    """
    x_seqs = []
    y_seqs = []
    total_steps = len(data)
    # Create windows such that we don't go out-of-bound.
    for i in range(total_steps - seq_len - forecast_horizon + 1):
        x_window = data[i: i + seq_len]
        y_window = data[i + forecast_horizon: i + forecast_horizon + seq_len]
        x_seqs.append(x_window)
        y_seqs.append(y_window)
    
    x = np.array(x_seqs).reshape(-1, seq_len, 1)
    y = np.array(y_seqs).reshape(-1, seq_len, 1)
    if return_torch: return torch.tensor(x, dtype=output_dtype), torch.tensor(y, dtype=output_dtype)
    else: return x.astype(output_dtype), x.astype(output_dtype)



def graph_x_y_split(data, train_ratio=0.6, val_ratio=0.2):
    """
    Splits the data into train, validation, and test sets with masks.
    
    Args:
        data (Data): PyTorch Geometric Data object with x, edge_index, and y attributes.
        train_ratio (float): Proportion of nodes to use for training.
        val_ratio (float): Proportion of nodes to use for validation.
        
    Returns:
        Data: Updated data object with train_mask, val_mask, and test_mask.
    """
    num_nodes = data.num_nodes
    indices = np.arange(num_nodes)
    np.random.shuffle(indices)

    # Define split sizes
    train_size = int(train_ratio * num_nodes)
    val_size = int(val_ratio * num_nodes)
    test_size = num_nodes - train_size - val_size

    # Get indices for each split
    train_indices = torch.tensor(indices[:train_size], dtype=torch.long)
    val_indices = torch.tensor(indices[train_size:train_size + val_size], dtype=torch.long)
    test_indices = torch.tensor(indices[train_size + val_size:], dtype=torch.long)

    # Initialize masks
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # Assign masks
    data.train_mask[train_indices] = True
    data.val_mask[val_indices] = True
    data.test_mask[test_indices] = True

    return data




def k_fold_validation(
        dataset: Union[torch.utils.data.Dataset, Tuple] = None,
        n_splits: int = 5,
        procedure: callable = None,
        shuffle: bool = True,
        index_only: bool = False,
        **kwargs
    ) -> List[Any]:
    """
    Performs K-Fold Cross Validation on a given dataset.

    Parameters:
    - dataset (torch.utils.data.Dataset): The dataset to split.
    - n_splits (int): Number of folds.
    - procedure (callable): 
        - Function to execute on each fold. Should accept (train_subset, test_subset, **kwargs). 
        - If index_only = True, it should accept (train_ids, test_ids, **kwargs)
    - shuffle (bool): Shuffle the dataset for K-fold.
    - index_only (bool): only pass in indices instead of the indexed and processed datasets. 
    - **kwargs: Additional keyword arguments to pass to the procedure.

    Return:
    - Return a list of results returned from your self-defined procedure.

    Examples:
    - Deep Learning Examples
        >>> N_SPLITS = 5
        >>> dataset = MNIST(root='./data', train=False, download=True, transform=transform)
        >>> def procedure(train_subset, test_subset, **kwargs):
        >>>     ...
        >>> k_fold_validation(dataset, procedure=procedure)

    """
    if dataset is None:
        raise ValueError("Dataset must be provided.")
    results_from_fold = []
    kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=42 if shuffle == True else None)
    flag = 0

    if isinstance(dataset, Tuple):
        dataset = dataset
        flag = 1
    else:
        dataset = [dataset]
    

    for fold, (train_ids, test_ids) in enumerate(kfold.split(*dataset)):
        tqdm.write(f"Current Fold: [{fold + 1}/{n_splits}]")
        tqdm.write(f"Training Data Size: {len(train_ids)}; Testing Data Size: {len(test_ids)}")
        if flag == 1:
            train_subset = tuple([ele[train_ids] for ele in dataset])
            test_subset = tuple([ele[test_ids] for ele in dataset])
        elif flag == 0:
            train_subset = Subset(dataset, train_ids)
            test_subset = Subset(dataset, test_ids)
        if index_only:
            result = procedure(train_ids, test_ids)
        else:
            result = procedure(train_subset, test_subset)
        results_from_fold.append(result)
        tqdm.write('\n')
    return results_from_fold

def sequential_x_y_split(
        data: Union[np.ndarray, List[float]],  # The input sequential data (array or list of floats/integers).
        look_back: int = 10,  # The number of time steps to include in each input sequence (window size).
        stride: int = 1,  # The step size for moving the window through the data.
        to_numpy: bool = True,  # Whether to return the results as NumPy arrays (True) or as lists (False).
) -> Tuple[Union[np.ndarray, List[List[float]]], Union[np.ndarray, List[float]]]:
    """
    Splits sequential data into input (x) and target (y) pairs for supervised learning.

    Parameters:
    - data (Union[np.ndarray, List[float]]): The sequential dataset to be split.
    - look_back (int, optional): The size of the input sequence (window). Default is 10.
    - stride (int, optional): The step size for moving the window. Default is 1.
    - to_numpy (bool, optional): If True, the outputs are converted to NumPy arrays. Default is True.
    
    Returns:
    - Tuple[Union[np.ndarray, List[List[float]]], Union[np.ndarray, List[float]]]:
        - x: The input sequences (shape: [num_samples, look_back]).
        - y: The corresponding target values (shape: [num_samples]).
    
    Example:
    --------
    >>> import numpy as np
    >>> data = np.arange(1, 21)  # Sequential data: [1, 2, ..., 20]
    >>> x, y = sequential_x_y_split(data, look_back=3, stride=1, to_numpy=True)
    >>> print("X:", x)
    >>> print("Y:", y)
    
    Output:
    --------
    X: [[ 1  2  3]
        [ 2  3  4]
        [ 3  4  5]
         ...
        [16 17 18]
        [17 18 19]]
    Y: [ 4  5  6  ... 19 20]
    """
    x = []  # List to store input sequences.
    y = []  # List to store target values.

    # Slide the window through the data
    for i in range(look_back, len(data), stride):
        # Append the input sequence (x) and target value (y)
        x.append(data[i - look_back: i])
        y.append(data[i])

    # Return as NumPy arrays or lists
    if to_numpy:
        return np.array(x), np.array(y)
    else:
        return x, y



def one_cut_split(
    data: Union[np.ndarray, list],
    val_split: Optional[Union[float, int]] = None, 
    test_split: Optional[Union[float, int]] = None
) -> Union[
    Tuple[np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray]
]:
    """
    Splits the data into training, validation, and testing sets based on provided split parameters.

    Parameters:
    - data (np.ndarray or list): The sequential data to be split.
    - val_split (Optional[Union[float, int]]): Validation split ratio or starting index.
    - test_split (Optional[Union[float, int]]): Testing split ratio or starting index.

    Returns:
    - Tuple containing training, validation, and/or testing splits.
    """
    if data is None:
        raise ValueError("Data must be provided for splitting.")
    if isinstance(data, list):
        data = np.array(data)
    elif not isinstance(data, np.ndarray):
        raise TypeError("Data must be a NumPy ndarray or list.")
    
    total_length = len(data)
    if total_length == 0:
        raise ValueError("Data is empty.")
    
    # Initialize split indices
    train_end = total_length  # Default end of training data
    val_start = None
    test_start = None

    # Handle test split first
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
        train_end = test_start  # Training ends before the test set starts

    # Handle validation split
    if val_split is not None:
        if isinstance(val_split, float):
            if not 0 < val_split < 1:
                raise ValueError("val_split as float must be between 0 and 1.")
            val_size = int((train_end) * val_split)  # Use the remaining portion for validation
        elif isinstance(val_split, int):
            if not 0 <= val_split < train_end:
                raise ValueError("val_split as int must be within the training data range.")
            val_size = train_end - val_split
        else:
            raise TypeError("val_split must be either float or int.")

        val_start = train_end - val_size
        train_end = val_start  # Training ends before the validation set starts

    # Perform slicing
    train_data = data[:train_end]

    if val_split is not None:
        val_data = data[val_start:val_start + val_size]
    else:
        val_data = None

    if test_split is not None:
        test_data = data[test_start:]
    else:
        test_data = None

    # Prepare return tuple
    if val_data is not None and test_data is not None:
        return train_data, val_data, test_data
    elif val_data is not None:
        return train_data, val_data
    elif test_data is not None:
        return train_data, test_data
    else:
        return (train_data,)



def xy_to_tensordataset(
    X,
    y,
    val_ratio: float = 0.0,
    test_ratio: float = 0.0,
    data_object: Any = TensorDataset,
    shuffle: bool = True,
    random_seed: int = 42,
    # NEW PARAMETERS FOR DATALOADER
    return_loader: bool = False,
    batch_size: int = 32,
    # You can add more DataLoader params if desired (e.g., num_workers, pin_memory, drop_last)
    drop_last: bool = False,
    unsqueeze_last: bool = False,
    input_dtype: torch.dtype = None,
    output_dtype: torch.dtype = None
) -> Union[
    TensorDataset,
    Tuple[TensorDataset, TensorDataset],
    Tuple[TensorDataset, TensorDataset, TensorDataset],
    DataLoader,
    Tuple[DataLoader, DataLoader],
    Tuple[DataLoader, DataLoader, DataLoader]
]:
    """
    Converts X and y into a TensorDataset, optionally splitting into 
    (train), (train, val), or (train, val, test). If 'return_loader=True', 
    returns the corresponding DataLoader(s) instead of the dataset(s).

    Parameters
    ----------
    X : 
        Feature array (list, tuple, np.ndarray, or torch.Tensor).
        Must have shape [n_samples, ...].
    y : 
        Target array (list, tuple, np.ndarray, or torch.Tensor).
        Must have shape [n_samples, ...].

    val_ratio : float, optional
        Fraction of data to go into the validation set. Default is 0.0 (no validation set).

    test_ratio : float, optional
        Fraction of data to go into the test set. Default is 0.0 (no test set).

    data_object : Any, optional
        The data type the returned dataset would have. Default is TensorDataset
    
    shuffle : bool, optional
        Whether to shuffle the entire dataset before splitting (for train/val/test).
        By default True.

    random_seed : int, optional
        Random seed used for shuffling. Default is 42.

    return_loader : bool, optional
        If True, returns DataLoader(s) instead of TensorDataset(s). Default False.

    batch_size : int, optional
        Batch size for the DataLoader (if return_loader=True). Default 32.

    drop_last : bool, optional
        Whether to drop the last incomplete batch (if return_loader=True). Default False.

    input_dtype : torch.dtype, optional
        Whether to set the dtype for the input tensors

    output_dtype : torch.dtype, optional
        Whether to set the dtype for the output tensors
    Returns
    -------
    If return_loader=False:
        - If val_ratio == 0 and test_ratio == 0:
            TensorDataset
        - If val_ratio > 0 and test_ratio == 0:
            (train_dataset, val_dataset)
        - If val_ratio == 0 and test_ratio > 0:
            (train_dataset, test_dataset)
        - If val_ratio > 0 and test_ratio > 0:
            (train_dataset, val_dataset, test_dataset)

    If return_loader=True:
        - If val_ratio == 0 and test_ratio == 0:
            DataLoader
        - If val_ratio > 0 and test_ratio == 0:
            (train_loader, val_loader)
        - If val_ratio == 0 and test_ratio > 0:
            (train_loader, test_loader)
        - If val_ratio > 0 and test_ratio > 0:
            (train_loader, val_loader, test_loader)

    Raises
    ------
    TypeError
        If X or y is not a torch.Tensor, np.ndarray, list, or tuple.

    ValueError
        If X and y do not share the same first-dimension length,
        or if val_ratio + test_ratio >= 1,
        or if computed subset sizes are invalid.
    """

    # --- 1) Convert X, y to torch.Tensor ---
    def to_tensor(arr, name: str):
        if isinstance(arr, torch.Tensor):
            return arr
        elif isinstance(arr, np.ndarray):
            return torch.from_numpy(arr)
        elif isinstance(arr, (list, tuple)):
            return torch.tensor(arr)
        else:
            raise TypeError(
                f"Unsupported data type for {name}. "
                f"Expected torch.Tensor, np.ndarray, list, or tuple, but got {type(arr)}."
            )

    X_t = to_tensor(X, "X")
    y_t = to_tensor(y, "y")
    if input_dtype is not None: X_t = X_t.to(input_dtype)
    if output_dtype is not None: y_t = y_t.to(output_dtype)

    if unsqueeze_last:
        X_t = X_t.unsqueeze(-1)
        y_t = y_t.unsqueeze(-1)
    # --- 2) Check consistent lengths (first dimension) ---
    if X_t.shape[0] != y_t.shape[0]:
        raise ValueError(
            f"X and y must have the same number of samples. "
            f"Got X: {X_t.shape[0]}, y: {y_t.shape[0]}."
        )

    n_samples = X_t.shape[0]

    # --- 3) Validate ratio sums ---
    if val_ratio + test_ratio >= 1.0:
        raise ValueError(
            f"val_ratio + test_ratio must be < 1.0, but got {val_ratio} + {test_ratio}."
        )

    # --- 4) Shuffle if requested ---
    indices = torch.arange(n_samples)
    if shuffle:
        g = torch.Generator()
        g.manual_seed(random_seed)
        indices = indices[torch.randperm(n_samples, generator=g)]

    # --- 5) Compute sizes for train/val/test ---
    val_size = int(val_ratio * n_samples)
    test_size = int(test_ratio * n_samples)
    train_size = n_samples - val_size - test_size

    if train_size < 0:
        raise ValueError(
            f"Computed train_size < 0 (train_size={train_size}). "
            f"Check val_ratio={val_ratio} and test_ratio={test_ratio}."
        )

    # Partition indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]  # could be empty
    test_indices = indices[train_size + val_size :]            # could be empty

    # Helper to slice X, y and return a TensorDataset
    def slice_xy(idxs, data_object):
        return data_object(X_t[idxs], y_t[idxs])

    # --- 6) Build the subsets (TensorDataset) ---
    train_ds = slice_xy(train_indices, data_object)

    # Decide how many subsets we'll have
    has_val = (val_size > 0)
    has_test = (test_size > 0)

    # If no validation or test
    if not has_val and not has_test:
        # 6A) Return single
        if not return_loader:
            return train_ds
        else:
            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=shuffle,   # Typically True for training
                drop_last=drop_last
            )
            return train_loader

    elif has_val and not has_test:
        # 6B) Return (train, val)
        val_ds = slice_xy(val_indices, data_object)
        if not return_loader:
            return (train_ds, val_ds)
        else:
            # Typically, we shuffle only the training DataLoader
            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last
            )
            # Validation is typically not shuffled
            val_loader = DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False
            )
            return (train_loader, val_loader)

    elif not has_val and has_test:
        # 6C) Return (train, test)
        test_ds = slice_xy(test_indices, data_object)
        if not return_loader:
            return (train_ds, test_ds)
        else:
            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last
            )
            test_loader = DataLoader(
                test_ds,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False
            )
            return (train_loader, test_loader)

    else:
        # 6D) Return (train, val, test)
        val_ds = slice_xy(val_indices, data_object)
        test_ds = slice_xy(test_indices, data_object)
        if not return_loader:
            return (train_ds, val_ds, test_ds)
        else:
            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False
            )
            test_loader = DataLoader(
                test_ds,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False
            )
            return (train_loader, val_loader, test_loader)


def _default_getitem(dataset, index):
    data, label = dataset[index]
    return data, label

class UnsqueezeDataset(Dataset):
    """
    A wrapper for a dataset to unsqueeze the labels along a specified dimension.
    """
    def __init__(self, dataset, funct_implemented: Callable = _default_getitem):
        """
        Args:
            dataset (Dataset): The original dataset to wrap.
            funct_implemented (Callable): The custom function to get items from a dataset
        """
        self.dataset = dataset
        self.funct_implemented = funct_implemented

    def __getitem__(self, index):
        """
        Returns a tuple of (data, label) with the label unsqueezed.
        """
        return self.funct_implemented(self.dataset, index)

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.dataset)
    
    