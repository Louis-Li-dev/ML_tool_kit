from torch import nn
import torch
import math
from torch.nn import functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads=1):
        super().__init__()
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        
        # Define linear layers for queries, keys, and values
        self.linear_q = nn.Linear(model_dim, model_dim, bias=False)
        self.linear_k = nn.Linear(model_dim, model_dim, bias=False)
        self.linear_v = nn.Linear(model_dim, model_dim, bias=False)
        
        # Output linear layer
        self.linear_out = nn.Linear(model_dim, model_dim, bias=False)
        
    def forward(self, x):
        # x: (batch_size, seq_len, model_dim)
        batch_size, seq_len, _ = x.size()
        
        # Linear projections
        queries = self.linear_q(x)  # (batch_size, seq_len, model_dim)
        keys    = self.linear_k(x)  # (batch_size, seq_len, model_dim)
        values  = self.linear_v(x)  # (batch_size, seq_len, model_dim)
        
        # Reshape for multi-head attention
        # New shape: (batch_size, num_heads, seq_len, head_dim)
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys    = keys.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values  = values.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(queries, keys.transpose(-2, -1))  # (batch_size, num_heads, seq_len, seq_len)
        scores = scores / math.sqrt(self.head_dim)
        
        attention_weights = F.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        attention_output = torch.matmul(attention_weights, values)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous()  # (batch_size, seq_len, num_heads, head_dim)
        attention_output = attention_output.view(batch_size, seq_len, self.model_dim)  # (batch_size, seq_len, model_dim)
        
        # Final linear layer
        output = self.linear_out(attention_output)  # (batch_size, seq_len, model_dim)
        
        return output

class FeedForward(nn.Module):
    def __init__(self, model_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = model_dim
        self.fc1 = nn.Linear(model_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, model_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
# --------------------------------------
# Model Components: Transformer for Time Series
# --------------------------------------
class SelfAttention(nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        self.model_dim = model_dim
        self.linear_q = nn.Linear(model_dim, model_dim, bias=False)
        self.linear_k = nn.Linear(model_dim, model_dim, bias=False)
        self.linear_v = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, x):
        # x: (batch_size, seq_len, model_dim)
        batch_size, seq_len, _ = x.size()
        queries = self.linear_q(x)
        keys    = self.linear_k(x)
        values  = self.linear_v(x)
        
        scores = torch.matmul(queries, keys.transpose(-2, -1))  # (batch_size, seq_len, seq_len)
        scores = scores / math.sqrt(self.model_dim)
        
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, values)
        return attention_output

class FeedForward(nn.Module):
    def __init__(self, model_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = model_dim
        self.fc1 = nn.Linear(model_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, model_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, output_dim, num_layers=1, num_heads=1):
        """
        input_dim: Number of features per time step.
        model_dim: Internal model dimension.
        output_dim: Number of predictions per time step.
        num_layers: Number of transformer layers.
        """
        super().__init__()
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': MultiHeadAttention(model_dim, num_heads),
                'ff': FeedForward(model_dim),
            })
            for _ in range(num_layers)
        ])
        self.output_projection = nn.Linear(model_dim, output_dim)
    def layer_norm(self, x):
        with torch.no_grad():
            std = x.std(dim=1, keepdim=True)
            mean = x.mean(dim=1, keepdim=True)
            normed_x = (x - mean) / std
        return normed_x
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.input_projection(x)  # (batch_size, seq_len, model_dim)
        for layer in self.layers:
            attn_out = layer['attention'](x)
            x = self.layer_norm(x + attn_out)
            ff_out = layer['ff'](x)
            x = self.layer_norm(x + ff_out)
        output = self.output_projection(x)  # (batch_size, seq_len, output_dim)
        return output

