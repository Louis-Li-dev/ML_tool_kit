import torch
from torch import nn
from typing import List, Optional


class ResidualIndRNNBlock(nn.Module):
    """
    A single Residual IndRNN (Independently Recurrent Neural Network) block.

    Each time step computation has a residual connection and two LayerNorm steps
    to stabilize training. If `input_dim != hidden_dim`, an input projection layer
    is employed to match dimensions.

    Attributes:
        hidden_dim (int): Dimensionality of the hidden state.
        seq_len (int): Sequence length for the recurrent operation.
        input_projection (Optional[nn.Linear]): Linear layer to project input
            dimension to hidden dimension if they differ.
        input_weights (nn.Parameter): Weight matrix for the input contribution.
        recurrent_weights (nn.Parameter): Weight vector for the recurrent contribution.
        bias (nn.Parameter): Bias term added during recurrent computation.
        bn1 (nn.LayerNorm): LayerNorm used after the first recurrent step.
        bn2 (nn.LayerNorm): LayerNorm used after the second recurrent step.
        relu (nn.ReLU): ReLU activation function.
        device (torch.device): The device on which this module’s tensors should be stored.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        seq_len: int,
        device: torch.device
    ) -> None:
        """
        Initializes the ResidualIndRNNBlock.

        Args:
            input_dim (int): The dimensionality of the input at each time step.
            hidden_dim (int): The dimensionality of the hidden state within the block.
            seq_len (int): The length of the input sequence.
            device (torch.device): The device on which parameters and operations are placed.
        """
        super(ResidualIndRNNBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.device = device

        # Define learnable parameters
        self.input_weights = nn.Parameter(torch.randn(hidden_dim, hidden_dim, device=self.device))
        self.recurrent_weights = nn.Parameter(torch.randn(hidden_dim, device=self.device))
        self.bias = nn.Parameter(torch.zeros(hidden_dim, device=self.device))

        # Optional projection if input dimension doesn't match hidden dimension
        if input_dim != hidden_dim:
            self.input_projection = nn.Linear(input_dim, hidden_dim).to(self.device)
        else:
            self.input_projection = None

        # Normalization and activation layers
        self.bn1 = nn.LayerNorm(hidden_dim).to(self.device)
        self.bn2 = nn.LayerNorm(hidden_dim).to(self.device)
        self.relu = nn.ReLU().to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the ResidualIndRNNBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_dim).
        """
        batch_size, seq_len, _ = x.size()

        device = x.device

        self.input_weights = self.input_weights.to(device)
        self.recurrent_weights = self.recurrent_weights.to(device)
        self.bias = self.bias.to(device)

        # Initialize hidden state on the correct device
        hidden = torch.zeros(batch_size, self.hidden_dim, device=device)
        
        outputs = []
        for t in range(seq_len):
            # Project input if necessary
            x_t = self.input_projection(x[:, t, :]) if self.input_projection else x[:, t, :]

            # First recurrent pass
            input_contribution = torch.matmul(x_t, self.input_weights)
            recurrent_contribution = hidden * self.recurrent_weights
            hidden = self.bn1(input_contribution + recurrent_contribution + self.bias)
            hidden = self.relu(hidden)

            # Second recurrent pass
            recurrent_contribution2 = hidden * self.recurrent_weights
            hidden = self.bn2(recurrent_contribution2 + input_contribution)
            hidden = self.relu(hidden)

            # Residual connection
            hidden = hidden + x_t

            outputs.append(hidden.unsqueeze(1))

        return torch.cat(outputs, dim=1)


class StackedResidualIndRNN(nn.Module):
    """
    A stacked architecture of ResidualIndRNNBlock layers,
    followed by a fully connected output layer.

    Attributes:
    ---
        layers (nn.ModuleList): List of ResidualIndRNNBlock layers.
        fc (nn.Linear): Final linear layer mapping the last hidden state
            to the desired output dimension.

    How to use:
    ---
    >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    >>> model = StackedResidualIndRNN(
    >>>     input_dim=32,
    >>>     hidden_dims=[64, 64],
    >>>     seq_len=100,
    >>>     output_dim=10,
    >>>     device=device
    >>> )
    >>> model.to(device)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        seq_len: int,
        output_dim: int,
        device: torch.device
    ) -> None:
        """
        Initializes a stacked ResidualIndRNN model.

        Args:
            input_dim (int): Dimensionality of the input at each time step.
            hidden_dims (List[int]): List containing the hidden dimension sizes for each layer.
            seq_len (int): The length of the input sequences.
            output_dim (int): The dimensionality of the final output.
            device (torch.device): The device on which parameters and operations are placed.
        """
        super(StackedResidualIndRNN, self).__init__()

        self.layers = nn.ModuleList()
        for i, hidden_dim in enumerate(hidden_dims):
            self.layers.append(
                ResidualIndRNNBlock(
                    input_dim=input_dim if i == 0 else hidden_dims[i - 1],
                    hidden_dim=hidden_dim,
                    seq_len=seq_len,
                    device=device
                )
            )

        self.fc = nn.Linear(hidden_dims[-1], output_dim).to(device)
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass through the stacked Residual IndRNN layers,
        ending with a fully connected projection.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: Output logits or features of shape (batch_size, output_dim).
        """
        
        if x.dtype != torch.float32:
            x = x.float()

        # Check the dimensionality of the input tensor
        if x.dim() == 2:
            # If input is [batch_size, input_dim], add a dummy sequence length
            x = x.unsqueeze(-1)  # Add a sequence length of 1
        elif x.dim() != 3:
            # Raise an exception if the input tensor doesn't have the expected shape
            raise ValueError(
                f"Input tensor has invalid shape {x.shape}. Expected (batch_size, seq_len, input_dim)."
            )

        # Process through the stacked layers
        for layer in self.layers:
            x = layer(x)

        # Use the last time step from the final layer for classification/regression
        x = x[:, -1, :]
        x = self.fc(x)

        return x

