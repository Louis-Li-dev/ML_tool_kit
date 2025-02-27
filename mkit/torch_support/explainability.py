import matplotlib.pyplot as plt
import seaborn as sns
import torch
from typing import Union
def collect_parameters(
        model: torch.nn.Module, 
        param_type: str = "weight"):
    """
    Collects weights or biases from a PyTorch model.
    
    Args:
        model (torch.nn.Module): The model to extract parameters from.
        param_type (str): Either "weight" or "bias" to specify which parameters to collect.

    Returns:
        dict: A dictionary mapping layer names to their respective parameters.
    """
    param_dict = {}
    
    for name, params in model.named_parameters():
        if param_type in name:  # Filter weights or biases
            param_dict[name] = params.cpu().detach().numpy()

    if not param_dict:
        raise ValueError(f"No {param_type} parameters found in the model.")
    
    return param_dict

def visualize_weights_or_biases(
        model: torch.nn.Module, 
        layer_index: Union[str, int] = None, 
        param_type="weight"):
    """
    Visualizes the weights or biases of a PyTorch model.
    
    Args:
        model (torch.nn.Module): The model containing parameters.
        layer_index (int or str, optional): Specifies which layer to visualize.
            - If an integer (0, 1, 2), visualizes the first, second, or third layer.
            - If "all", visualizes all layers in subplots.
            - If None, defaults to the first layer.
        param_type (str): Either "weight" or "bias" to specify what to visualize.
    """
    param_dict = collect_parameters(model, param_type=param_type)
    layer_names = list(param_dict.keys())

    if isinstance(layer_index, int):
        # Ensure index is within range
        if layer_index < 0 or layer_index >= len(layer_names):
            raise IndexError(f"Layer index {layer_index} is out of range (0 to {len(layer_names)-1}).")
        layer_names = [layer_names[layer_index]]  # Select the specific layer
    elif layer_index == "all":
        pass  # Keep all layers

    else:  # Default to first layer if unspecified
        layer_names = [layer_names[0]]

    # Plot single or multiple layers
    num_layers = len(layer_names)
    fig, axes = plt.subplots(1, num_layers\
        , figsize=(5 * num_layers, 4) if num_layers != 1 else (param_dict[layer_names[0]].shape[1] // 6 + 1,
         param_dict[layer_names[0]].shape[0] // 6 + 1))

    if num_layers == 1:
        axes = [axes]  # Ensure axes is iterable for single-layer cases

    for ax, layer_name in zip(axes, layer_names):
        params = param_dict[layer_name]
        sns.heatmap(params, cmap="coolwarm", center=0, annot=False, cbar=True, ax=ax)

        # Set font and labels consistently
        font_config = {"fontname": "Times New Roman", "fontsize": 12, "fontweight": "bold"}
        ax.set_title(f"{param_type.capitalize()} Visualization: {layer_name}", **font_config)
        ax.set_xlabel("Neurons in Next Layer", fontname="Times New Roman", fontsize=10)
        ax.set_ylabel("Neurons in Previous Layer", fontname="Times New Roman", fontsize=10)

        # Tick label font
        ax.tick_params(axis='both', labelsize=9)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname("Times New Roman")

        # Colorbar font
        cbar = ax.collections[0].colorbar
        cbar.ax.set_ylabel(f"{param_type.capitalize()} Magnitude", fontname="Times New Roman", fontsize=10)
        for label in cbar.ax.get_yticklabels():
            label.set_fontname("Times New Roman")

    plt.tight_layout()
    plt.show()