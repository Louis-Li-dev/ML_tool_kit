import torch
from torch import nn

class Discriminator(nn.Module):
    """
    Discriminator:
    Discriminates between real and generated data using convolutional layers.
    
    Args:
        hidden_dims (list of int): A list specifying the number of output channels for each layer.
    """
    def __init__(self, hidden_dims):
        super(Discriminator, self).__init__()
        assert len(hidden_dims) >= 2, "hidden_dims must have at least two dimensions for meaningful discrimination."

        self.hidden_dims = hidden_dims

        # Dynamically build convolutional layers
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(self.__block(hidden_dims[i], hidden_dims[i + 1], (3, 3)))
        self.layers = nn.Sequential(*layers)

        # Fully connected layer for output
        last_dim = hidden_dims[-1]
        self.flatten = nn.Flatten()
        self.out = nn.Linear(last_dim * 23 * 49, 1)  # Adjust based on input size

    def __block(self, input_dim, output_dim, kernel):
        return nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel, padding=1, stride=1),
            nn.InstanceNorm2d(output_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25)
        )

    def forward(self, condi, path):
        """
        Forward Pass:
        Args:
            condi (torch.Tensor): Conditional input (e.g., labels or auxiliary data).
            path (torch.Tensor): Image or data path to discriminate.
        """
        x = torch.cat((condi, path), dim=1)  # Combine condition and path inputs
        x = self.layers(x)
        x = self.flatten(x)
        return nn.Sigmoid()(self.out(x))
