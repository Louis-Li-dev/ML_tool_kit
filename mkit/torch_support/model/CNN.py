import torch
import torch.nn as nn

class AdjustableCNN(nn.Module):
    def __init__(
        self,
        cnn_type="2d",              # "1d" or "2d" CNN
        input_channels=3,           # Number of input channels
        num_classes=10,             # Number of output classes
        num_filters=[32, 64],       # List of filters for each convolutional layer
        use_maxpool=True,           # Add pooling layers
        normalization="batch",      # "batch", "layer", "instance", or None
        activation="relu",          # Activation function: "relu", "leaky_relu", etc.
        width=0,                    # For 2D input: width of the input image
        height=0,                   # For 2D input: height of the input image
        seq_length=0,               # For 1D input: length of the input sequence
    ):
        super(AdjustableCNN, self).__init__()
        
        assert cnn_type in ["1d", "2d"], "cnn_type must be '1d' or '2d'"
        self.cnn_type = cnn_type
        
        layers = []
        in_channels = input_channels

        # Select layer types based on cnn_type
        if cnn_type == "1d":
            ConvLayer = nn.Conv1d
            NormBatch = nn.BatchNorm1d
            NormInstance = nn.InstanceNorm1d
            PoolLayer = nn.MaxPool1d
        else:  # "2d"
            ConvLayer = nn.Conv2d
            NormBatch = nn.BatchNorm2d
            NormInstance = nn.InstanceNorm2d
            PoolLayer = nn.MaxPool2d

        for out_channels in num_filters:
            # Convolution
            layers.append(ConvLayer(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            
            # Normalization
            if normalization == "batch":
                layers.append(NormBatch(out_channels))
            elif normalization == "layer":
                # For 1D: nn.LayerNorm([out_channels, length])
                # For 2D: nn.LayerNorm([out_channels, height, width])
                # We'll just apply a LayerNorm with a 1D shape for 1D CNN
                # or a 2D shape for 2D CNN. However, it's often simpler to do
                # it with the dummy input approach, or just skip layer norm
                # entirely if the shapes are not known up front.
                if cnn_type == "1d":
                    # A placeholder: might need dynamic shape
                    layers.append(nn.LayerNorm([out_channels, seq_length]))
                else:
                    layers.append(nn.LayerNorm([out_channels, height, width]))
            elif normalization == "instance":
                layers.append(NormInstance(out_channels))

            # Activation
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.1))

            # Pooling
            if use_maxpool:
                layers.append(PoolLayer(kernel_size=2, stride=2))

            in_channels = out_channels

        self.features = nn.Sequential(*layers)

        # ------------------------------------------------------------
        #  Dynamic calculation of the final feature dimension
        # ------------------------------------------------------------
        # We create a dummy input tensor to pass through 'self.features'
        # to see what the flattened size will be.

        # Switch to a CPU device for safe shape calculation
        device = torch.device("cpu")
        self.features.to(device)

        if self.cnn_type == "1d":
            if seq_length <= 0:
                raise ValueError("For a 1D CNN, please provide a valid seq_length.")
            dummy_input = torch.zeros(1, input_channels, seq_length, device=device)
        else:  # 2D
            if width <= 0 or height <= 0:
                raise ValueError("For a 2D CNN, please provide valid width and height.")
            dummy_input = torch.zeros(1, input_channels, height, width, device=device)

        with torch.no_grad():
            dummy_output = self.features(dummy_input)
        flattened_size = dummy_output.view(dummy_output.size(0), -1).shape[1]

        # Finally, create the classifier based on the computed size
        self.classifier = nn.Linear(flattened_size, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x
