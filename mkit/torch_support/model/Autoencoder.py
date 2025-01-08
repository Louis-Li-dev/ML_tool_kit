import torch
import torch.nn as nn


class GANEncoder(nn.Module):
    """
    GAN Encoder:
    Encodes input data into a latent representation using convolutional layers.
    
    Args:
        hidden_dims (list of int): A list specifying the number of output channels for each layer.
    """
    def __init__(self, hidden_dims):
        super(GANEncoder, self).__init__()
        assert len(hidden_dims) >= 2, "hidden_dims must have at least two dimensions for meaningful encoding."

        self.hidden_dims = hidden_dims

        # Dynamically build convolutional layers
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(self.__block(hidden_dims[i], hidden_dims[i + 1], (3, 3)))
        self.layers = nn.Sequential(*layers)

        # Residual connections
        self.residuals = nn.ModuleList(
            [nn.Conv2d(hidden_dims[0], hidden_dims[-1], kernel_size=1)]
        )
        self.gaussian_noise = nn.Parameter(
            torch.zeros(1, hidden_dims[0], 1, 1)
        )

    def __block(self, input_dim, output_dim, kernel):
        return nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel, padding=1, stride=1),
            nn.InstanceNorm2d(output_dim),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        res = self.residuals[0](x)
        for layer in self.layers:
            x = layer(x)
        x += self.gaussian_noise * torch.randn_like(x)
        return x + res


class GANDecoder(nn.Module):
    """
    GAN Decoder:
    Decodes latent representations back to input-like data using transpose convolutional layers.
    
    Args:
        hidden_dims (list of int): A list specifying the number of input channels for each layer.
    """
    def __init__(self, hidden_dims):
        super(GANDecoder, self).__init__()
        assert len(hidden_dims) >= 2, "hidden_dims must have at least two dimensions for meaningful decoding."

        self.hidden_dims = hidden_dims

        # Dynamically build transpose convolutional layers
        layers = []
        for i in range(len(hidden_dims) - 1, 0, -1):
            layers.append(self.__block(hidden_dims[i], hidden_dims[i - 1], (3, 3)))
        self.layers = nn.Sequential(*layers)

        # Residual connections
        self.residuals = nn.ModuleList(
            [nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[0], kernel_size=1)]
        )

    def __block(self, input_dim, output_dim, kernel):
        return nn.Sequential(
            nn.ConvTranspose2d(input_dim, output_dim, kernel, stride=1, padding=1),
            nn.InstanceNorm2d(output_dim),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        res = self.residuals[0](x)
        for layer in self.layers:
            x = layer(x)
        return x + res


class Encoder(nn.Module):
    """
    Encoder:
    Encodes input data into a lower-dimensional latent representation.

    Args:
        input_size (int): Size of the input feature vector.
        hidden_dims (list of int): A list specifying the number of units in each hidden layer.
        activation_fn (nn.Module): Activation function to use (default: nn.Tanh).
    """
    def __init__(self, input_size, hidden_dims, activation_fn=nn.Tanh):
        super(Encoder, self).__init__()
        assert len(hidden_dims) >= 1, "hidden_dims must have at least one hidden layer."

        self.hidden_dims = hidden_dims
        self.activation_fn = activation_fn

        # Dynamically build fully connected layers
        layers = []
        in_dim = input_size
        for out_dim in hidden_dims:
            layers.append(self.__block(in_dim, out_dim))
            in_dim = out_dim
        self.layers = nn.Sequential(*layers)

    def __block(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            self.activation_fn()
        )

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    """
    Decoder:
    Decodes latent representations back to higher-dimensional input-like data.

    Args:
        output_size (int): Size of the output feature vector.
        hidden_dims (list of int): A list specifying the number of units in each hidden layer.
        activation_fn (nn.Module): Activation function to use (default: nn.Tanh).
    """
    def __init__(self, output_size, hidden_dims, activation_fn=nn.Tanh):
        super(Decoder, self).__init__()
        assert len(hidden_dims) >= 1, "hidden_dims must have at least one hidden layer."

        self.hidden_dims = hidden_dims
        self.activation_fn = activation_fn

        # Dynamically build fully connected layers
        layers = []
        in_dim = hidden_dims[0]
        for out_dim in hidden_dims[1:]:
            layers.append(self.__block(in_dim, out_dim))
            in_dim = out_dim
        self.fc_out = nn.Linear(in_dim, output_size)  # Final output layer
        self.layers = nn.Sequential(*layers)

    def __block(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            self.activation_fn()
        )

    def forward(self, x):
        x = self.layers(x)
        return self.fc_out(x)


