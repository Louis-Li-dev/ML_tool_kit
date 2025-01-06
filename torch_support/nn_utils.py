import torch
import torch.nn as nn
from typing import Union, Tuple, Any, List, Callable
from tqdm import tqdm
def __extract_loader(loader: Any = None) -> Tuple[Any, Any]:
    """
    Extracts (inputs, labels) from a batch.

    Parameters
    -----------
    loader : Any

    Returns
    -------
    inputs, labels : Tuple[Any, Any]
    """
    if loader is None:
        raise ValueError("Loader can't be None")
    inputs, labels = loader
    return inputs, labels

def default_train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    loader: Any,  # The entire batch (whatever the DataLoader yields)
    device: torch.device
) -> torch.Tensor:
    """
    Default train step using the entire 'loader' object.

    1) Parse the loader into (inputs, targets).
    2) Check shape/dtype (for example, for CrossEntropyLoss).
    3) Move to device.
    4) Forward pass, loss, backward pass, optimizer step.

    Returns
    -------
    torch.Tensor
        The computed loss (scalar) for this batch.
    """

    # 1) Extract (inputs, targets) from the loader
    inputs, targets = __extract_loader(loader)

    # 2) (Optional) Error detection / validation
    #    If using CrossEntropyLoss, confirm targets are long
    if isinstance(criterion, nn.CrossEntropyLoss) and targets.dtype not in (torch.long, torch.int64):
        raise TypeError(
            f"Targets must be torch.long for CrossEntropyLoss. Found: {targets.dtype}."
        )
    # Check batch size
    if inputs.shape[0] != targets.shape[0]:
        raise ValueError(
            f"Mismatch in batch size: inputs has {inputs.shape[0]} samples, "
            f"but targets has {targets.shape[0]} samples."
        )
    # Check that inputs are floating point (common for typical models)
    if not torch.is_floating_point(inputs):
        raise TypeError(
            f"Inputs should be a floating-point tensor (e.g., float32). Found: {inputs.dtype}."
        )

    # 3) Move data to device
    inputs, targets = inputs.to(device), targets.to(device)

    # 4) Forward pass, loss, backward pass, optimizer step
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    return loss


def training_loop(
    model: nn.Module,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    epochs: int = 5,
    print_every: Union[int, None] = None,
    val_loader: torch.utils.data.DataLoader = None,
    keep_losses: bool = False,
    train_step_func: Callable = None
) -> Union[nn.Module, Tuple[nn.Module, List[float]], Tuple[nn.Module, List[float], List[float]]]:
    """
    Trains a PyTorch model over multiple epochs, optionally validating after each epoch.
    The entire 'loader' (batch) is provided to the train_step_func, enabling custom logic.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to be trained.
    device : torch.device
        The device on which to perform training (e.g., 'cuda' or 'cpu').
    train_loader : torch.utils.data.DataLoader
        DataLoader for the training dataset.
    optimizer : torch.optim.Optimizer
        The optimizer for updating model parameters.
    criterion : nn.Module
        The loss function (e.g., nn.CrossEntropyLoss).
    epochs : int, optional
        Number of training epochs, by default 5.
    print_every : Union[int, None], optional
        Print training status every N batches. If None, no per-batch printing is done.
        Defaults to None.
    val_loader : torch.utils.data.DataLoader, optional
        DataLoader for the validation dataset. If None, no validation is performed.
    keep_losses : bool, optional
        If True, store training/validation losses in lists and return them.
    train_step_func : callable, optional
        A function with signature:
           (model, optimizer, criterion, loader, device) -> torch.Tensor (loss).
        Defaults to `default_train_step`, which expects (inputs, targets).

    Returns
    -------
    model : nn.Module
        The trained model (in-place updated).
    (Optional) List[float] training_losses
    (Optional) List[float] val_losses
    """

    # If user doesn't specify a custom train step, use the default
    if train_step_func is None:
        train_step_func = default_train_step

    model.to(device)  # Move model parameters to the specified device

    training_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        # Iterate over training data
        for batch_idx, loader in enumerate(
            tqdm(train_loader, desc=f"EPOCH {epoch}/{epochs}", leave=True), start=1
        ):
            # -- Use the train_step_func which handles everything for this batch --
            loss = train_step_func(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                loader=loader,
                device=device
            )

            running_loss += loss.item()

            # Print batch-by-batch info if print_every is set
            if print_every is not None and batch_idx % print_every == 0:
                avg_loss = running_loss / print_every
                tqdm.write(
                    f"Epoch [{epoch}/{epochs}] "
                    f"Batch [{batch_idx}/{len(train_loader)}] "
                    f"Train Loss: {avg_loss:.4f}"
                )
                running_loss = 0.0

            # Keep track of each batch's loss
            training_losses.append(loss.item())

        # Validate (if val_loader is provided)
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_targets = __extract_loader(val_data)
                    val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)

                    val_outputs = model(val_inputs)
                    loss = criterion(val_outputs, val_targets)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            tqdm.write(f"Epoch [{epoch}/{epochs}] Validation Loss: {val_loss:.4f}\n")

    tqdm.write("Training complete.")

    # Return based on keep_losses
    if keep_losses:
        if val_loader is not None:
            return model, training_losses, val_losses
        else:
            return model, training_losses
    else:
        return model
