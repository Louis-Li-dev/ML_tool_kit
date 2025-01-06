import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from typing import Union, Tuple, Optional

def xy_to_tensordataset(
    X,
    y,
    val_ratio: float = 0.0,
    test_ratio: float = 0.0,
    shuffle: bool = True,
    random_seed: int = 42,
    # NEW PARAMETERS FOR DATALOADER
    return_loader: bool = False,
    batch_size: int = 32,
    # You can add more DataLoader params if desired (e.g., num_workers, pin_memory, drop_last)
    drop_last: bool = False
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
    def slice_xy(idxs):
        return TensorDataset(X_t[idxs], y_t[idxs])

    # --- 6) Build the subsets (TensorDataset) ---
    train_ds = slice_xy(train_indices)

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
        val_ds = slice_xy(val_indices)
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
        test_ds = slice_xy(test_indices)
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
        val_ds = slice_xy(val_indices)
        test_ds = slice_xy(test_indices)
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
