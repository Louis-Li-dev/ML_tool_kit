import torch
import torch.nn as nn
from typing import Union, Tuple, Any, List
from tqdm import tqdm
import warnings
class IterStep(nn.Module):
    def __init__(
            self,
        ):
        super(IterStep, self).__init__()
        pass
    def __extract_batch(self, batch: Any = None) -> Tuple[Any, Any]:
        """
        Extracts (inputs, labels) from a batch.

        Parameters
        -----------
        batch : Any

        Returns
        -------
        inputs, labels : Tuple[Any, Any]
        """
        if batch is None:
            raise ValueError("Loader can't be None")
        inputs, labels = batch
        return inputs, labels
    def forward(
        self,        
        model: nn.Module,
        criterion: nn.Module,
        batch: Any,  # The entire batch (whatever the DataLoader yields)
        device: torch.device,
        optimizer: torch.optim.Optimizer = None,
        train=True
    ):
        """
        Default train step using the entire batch object.

        1) Parse the batch into (inputs, targets).
        2) Check shape/dtype (for example, for CrossEntropyLoss).
        3) Move to device.
        4) Forward pass, loss, backward pass, optimizer step.
        Parameters
        ----------
        model : nn.Module
            The PyTorch model to be trained.
        criterion : nn.Module
            The loss function (e.g., nn.CrossEntropyLoss).
        batch : Any
            One batch being processed.
        device : torch.device
            The device on which to perform training (e.g., 'cuda' or 'cpu').
        optimizer : torch.optim.Optimizer, optional.
            The optimizer for updating model parameters.
        train : bool, optional
            Whether to train the model (True) or evaluate it (False), by default True.
        Returns
        -------
        torch.Tensor
            The computed loss (scalar) for this batch.
        """

        inputs, targets = self.__extract_batch(batch)
        inputs, targets = inputs.to(device), targets.to(device)
        if train:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():    
                outputs = model(inputs)
                loss = criterion(outputs, targets)
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
    train_step_module: nn.Module = IterStep(),
    early_stopping: bool = False,
    patience: int = 5,
) -> Union[nn.Module, Tuple[nn.Module, List[float]], Tuple[nn.Module, List[float], List[float]]]:
    """
    Trains a PyTorch model over multiple epochs, optionally validating after each epoch.
    The entire batch is provided to the train_step_func, enabling custom logic.

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
    train_step_module : nn.Module, optional
        A module with signature:
           (model, criterion, batch, device, optimizer, train) -> torch.Tensor (loss).
    early_stopping : bool, optional, by default False
        If the validation dataset is passed in, the model will be the one with the best validation score.
    patience : int, optional, by default 5
        If the early_stopping is set, every time the validation score doesn't improve, the counter increments until hitting patience to abort the training. 
    Returns
    -------
    model : nn.Module
        The trained model (in-place updated).
    (Optional) List[float] training_losses
    (Optional) List[float] val_losses

    Example
    -------
    ```python
    # Example model: a simple feed-forward network for classification
    class SimpleNet(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.layer1 = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.layer2 = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            # x should have shape [batch_size, input_dim]
            out = self.layer1(x)
            out = self.relu(out)
            out = self.layer2(out)
            return out

    # Example usage of the training_loop
    if __name__ == "__main__":
        # 1. Create some synthetic data (for demonstration)
        #    Suppose we have 1000 samples, each with 20 features, and we want 3-class classification.
        np.random.seed(42)
        X = np.random.randn(10000, 20).astype(np.float32)
        y = np.random.randint(0, 3, size=(10000,))

        train_loader, val_loader = xy_to_tensordataset(X, y.astype(np.int64), val_ratio=.2, return_loader=True)

        # 3. Initialize the model, loss, optimizer
        model = SimpleNet(input_dim=20, hidden_dim=50, output_dim=3)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # 4. Define or import your training_loop function
        # (Assuming you've already defined it in your code as shown previously.)
        # from your_script_name import training_loop

        # 5. Train the model
        trained_model = training_loop(
            model=model,
            device=device,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            epochs=5,              # Number of epochs
            val_loader=val_loader  # Use the validation loader to check val loss
        )

        # 6. After training, you can do further evaluation or inference
        model.eval()
        test_input = torch.randn(1, 20, device=device)  # A single sample
        with torch.no_grad():
            logits = trained_model(test_input)
            predicted_label = torch.argmax(logits, dim=1)
        print("Example inference on a single test sample:", predicted_label.item())
    ```
    """

    model.to(device)  # Move model parameters to the specified device

    training_losses = []
    val_losses = []

    best_model_state = None
    best_val_loss = float('inf')
    count = 0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        train_loss = 0.0
        # Iterate over training data
        for batch_idx, batch in enumerate(
            tqdm(train_loader, desc=f"EPOCH {epoch}/{epochs}", leave=True), start=1
        ):
            # -- Use the train_step_func which handles everything for this batch --
            loss = train_step_module(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                batch=batch,
                device=device
            )

            running_loss += loss.item()
            train_loss += loss.item()
            # Print batch-by-batch info if print_every is set
            if print_every is not None and batch_idx % print_every == 0:
                avg_loss = running_loss / print_every
                tqdm.write(
                    f"Epoch [{epoch}/{epochs}] "
                    f"Batch [{batch_idx}/{len(train_loader)}] "
                    f"Training Loss: {avg_loss:.4f}"
                )
                running_loss = 0.0

            # Keep track of each batch's loss
            training_losses.append(loss.item())
        tqdm.write(
            f"Epoch [{epoch}/{epochs}] "
            f"Training Loss: {train_loss/len(train_loader):.4f}", end=" "
        )
        # Validate (if val_loader is provided)
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_data in val_loader:
                    # val_inputs, val_targets = __extract_batch(val_data)
                    # val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)

                    # val_outputs = model(val_inputs)
                    # loss = criterion(val_outputs, val_targets)
                    loss = train_step_module(
                        model=model,
                        criterion=criterion,
                        batch=val_data,
                        device=device,
                        train=False
                    )

                    val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            tqdm.write(f" Validation Loss: {val_loss:.4f}\n")
            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict()
                    count = 0
                else:
                    count += 1
                if count > patience:
                    tqdm.write(f"{count} times not improving the score. Training stops.")
                    break

        elif val_loader is None and early_stopping == True:
            warnings.warn("Early stopping is turned off as there is no validation dataset.")

    tqdm.write("Training complete.")
    if early_stopping and best_model_state is not None:
        tqdm.write(f"Best validation loss: {best_val_loss}, model loaded with the selected parameters")
        model.load_state_dict(best_model_state)
    model.eval()
    # Return based on keep_losses
    if keep_losses:
        if val_loader is not None:
            return model, training_losses, val_losses
        else:
            return model, training_losses
    else:
        return model
