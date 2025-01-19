import torch
import torch.nn as nn

# create a class wrapper from PyTorch nn.Module for exponential activation
class ExpAct(nn.Module):
    '''
    Applies a safe Exponential activation function element-wise:
        ExpAct(x) = exp(clamp(x, max=88.0))
    Shape:
        - Input: (N, *) where * means any number of additional dimensions
        - Output: (N, *), same shape as the input
    '''
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__() # init the base class

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        clamp = 5 *torch.tanh(input/5)
        return torch.exp(clamp)#consider reducing to 20


class LinearBlock(nn.Module):
    """
    Linear layer with ReLU and BatchNorm
    """
    def __init__(self, input_prev, embed_dim, dropout_rate):
        super(LinearBlock, self).__init__()
        self.fc = nn.Linear(input_prev, embed_dim)
        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(embed_dim)
        #self.bn = nn.BatchNorm1d(embed_dim)
        #self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        x = inputs
        x = self.fc(x)
        x = self.gelu(x)
        #x = self.bn(x)
        x = self.ln(x)
        #x = self.dropout(x)
        return x

class ResidualBlock(nn.Module):
    """
    Residual block with two linear layers and sinusoidal position embedding for time step t.
    """
    def __init__(self, embed_dim, time_embedding_dim, dropout_rate):
        super(ResidualBlock, self).__init__()
        self.layers = nn.ModuleList([
            LinearBlock(embed_dim, embed_dim, dropout_rate),
            LinearBlock(embed_dim, embed_dim, dropout_rate)
        ])
        self.time_proj = nn.Linear(time_embedding_dim, embed_dim*2)

    def forward(self, inputs, t):
        x = inputs
        t_emb = sinusoidal_position_embedding(t, self.time_proj.in_features)
        t_emb = self.time_proj(t_emb)  # Project to same dimension as x
        t_emb = t_emb.chunk(2, dim=-1)
        scale, shift = t_emb
        x = x * (scale + 1) + shift

        
        for layer in self.layers:
            x = layer(x)  # Now dimensions match for addition
        x += inputs  # skip-connection
        return x

class ResidualBlock_no_time(nn.Module):
    """
    Residual block with two linear layers
    """
    def __init__(self, embed_dim, dropout_rate):
        super(ResidualBlock_no_time, self).__init__()
        self.layers = nn.ModuleList([
            LinearBlock(embed_dim, embed_dim, dropout_rate),
            LinearBlock(embed_dim, embed_dim, dropout_rate)
        ])
    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        x += inputs
        return x

import math

def sinusoidal_position_embedding(t, dim):
    """
    Generate sinusoidal temporal embeddings for the given time step t.
    """
    half_dim = dim // 2
    emb = math.log(1000) / (half_dim - 1)#reduced from 10000 as time steps are small
    emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float) * -emb)
    emb = t.unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat((emb.sin(), emb.cos()), dim=1)
    return emb

class Model(nn.Module):
    """
    Architecture suitable for score matching objective.
    """
    def __init__(self, input_dim=324, output_dim=1, time_embedding_dim=128, width=1000, num_classes=6, dropout_rate=0,n_residual_blocks=4):
        super(Model, self).__init__()
        self.model_type = "FF_residual_model"
        self.input_dim = input_dim
        self.time_embedding_dim = time_embedding_dim
        self.num_classes = num_classes
        self.n_residual_blocks = n_residual_blocks

        # Generator embedding
        # DOESN'T DO ANYTHING!
        #self.generator_embedding = nn.Embedding(len(env.moves), time_embedding_dim)

        # Main network
        self.embedding = LinearBlock(input_dim, 5*width, dropout_rate)  # Removed time_embedding_dim from input
        self.layers = nn.ModuleList([
            LinearBlock(5*width, width, dropout_rate),
            LinearBlock(width, width, dropout_rate),
            *[ResidualBlock(width, time_embedding_dim, dropout_rate) 
               for _ in range(self.n_residual_blocks)],  # Default n_residual_blocks=4
        ])
        self.output = nn.Sequential(
            nn.Linear(width, output_dim),
            ExpAct()
            #nn.Softplus()
        )  # Output dimension matches input for score matching, with Softplus activation

    def forward(self, inputs, t):
        # Convert inputs to one-hot vectors and reshape
        x = nn.functional.one_hot((inputs+2).long(), num_classes=self.num_classes).to(torch.float)
        x = x.reshape(-1, self.input_dim)
        #print(self.embedding)
        # Main network - now time embedding is handled in ResidualBlocks
        x = self.embedding(x)
        x = self.layers[0](x)
        x = self.layers[1](x)

        for layer in self.layers[2:]:
            x = layer(x, t)
        
        # Output score
        score = self.output(x)
        
        return score



class TransformerModel(nn.Module):
    """
    Transformer-based architecture for score matching on AC presentations.
    The input is a sequence of integers representing generator indices and their signs.
    """
    def __init__(self, input_dim=324, output_dim=12, time_embedding_dim=128, 
                 d_model=256, nhead=8, num_layers=4, max_seq_len=200, num_classes=5, dropout_rate=0):
        super(TransformerModel, self).__init__()
        self.model_type = "TransformerEncoderModel"
        self.input_dim = input_dim
        self.time_embedding_dim = time_embedding_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_classes = num_classes

        # Token embedding - maps each integer (-2,-1,0,1,2) to a vector
        self.token_embedding = nn.Embedding(num_classes, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        # Time embedding projection - takes scalar time values
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embedding_dim),  # Input is scalar time value
            nn.GELU(),
            nn.Linear(time_embedding_dim, d_model)
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.output = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, output_dim),
            ExpAct()
        )

    def forward(self, inputs, t):
        # inputs shape: (batch_size, seq_len)
        # t shape: (batch_size,)
        
        # Create attention mask for padding tokens (0s)
        padding_mask = (inputs == 0)
        
        # Embed tokens
        x = self.token_embedding(inputs + 2)  # Shift by 2 to make indices non-negative
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :]
        
        # Project scalar time values to embedding space
        t_emb = self.time_mlp(t.unsqueeze(-1))  # Shape: (batch, d_model)
        x = x + t_emb.unsqueeze(1)  # Add time embedding to each position
        
        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        
        # Global average pooling over sequence length (excluding padding)
        mask = ~padding_mask
        x = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        
        # Output score
        score = self.output(x)
        
        return score

class TransformerModel_RC(nn.Module):
    """
    Transformer-based architecture for score matching on AC presentations.
    The input is a sequence of integers representing generator indices and their signs.
    """
    def __init__(self, output_dim=12, time_embedding_dim=128, 
                 d_model=256, nhead=8, num_layers=4, max_seq_len=200, num_classes=5, dropout_rate=TrainConfig.dropout,dim_feedforward_transformer=None,dim_MLP_end = None):
        super(TransformerModel_RC, self).__init__()
        if dim_feedforward_transformer is None:
            self.dim_feedforward_transformer = 4*d_model
        else:
            self.dim_feedforward_transformer = dim_feedforward_transformer
        self.model_type = "TransformerEncoderModel"
        #self.input_dim = input_dim
        self.time_embedding_dim = time_embedding_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_classes = num_classes

        # Token embedding - maps each integer (-2,-1,0,1,2) to a vector in R^d_model
        self.token_embedding = nn.Embedding(num_classes, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        # Time embedding projection - takes scalar time values
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embedding_dim),  # Input is scalar time value
            nn.GELU(),
            nn.Linear(time_embedding_dim, d_model)
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=self.dim_feedforward_transformer,
            dropout=dropout_rate,
            batch_first=True,
            activation= "gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.MLP_end = nn.Sequential(
            nn.Linear(d_model,dim_MLP_end),
            ResidualBlock_no_time(dim_MLP_end,dropout_rate),
            ResidualBlock_no_time(dim_MLP_end,dropout_rate),
            ResidualBlock_no_time(dim_MLP_end,dropout_rate),
            ResidualBlock_no_time(dim_MLP_end,dropout_rate),
            ResidualBlock_no_time(dim_MLP_end,dropout_rate),
            ResidualBlock_no_time(dim_MLP_end,dropout_rate),
            ResidualBlock_no_time(dim_MLP_end,dropout_rate),
            ResidualBlock_no_time(dim_MLP_end,dropout_rate),
            #ResidualBlock_no_time(dim_MLP_end,dropout_rate),
            nn.Linear(dim_MLP_end,d_model),
        )
        # Output layers
        self.output = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, output_dim),
            ExpAct()
        )

    def forward(self, inputs, t):
        # inputs shape: (batch_size, seq_len)
        # t shape: (batch_size,)
        
        # Create attention mask for padding tokens (0s)
        #padding_mask = (inputs == 0)
        
        # Embed tokens
        x = self.token_embedding(inputs) #shape of x is (batch, seq_len, d_model)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :]
        
        # Project scalar time values to embedding space
        t_emb = self.time_mlp(t.unsqueeze(-1)).unsqueeze(1)  # Shape: (batch, d_model,1)
        x = x + t_emb  # Add time embedding to each position uniformly
        # Apply transformer
        x = self.transformer(x)
        
        # Global average pooling over sequence length (excluding padding)
        #mask = ~padding_mask
        #x = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)# global average pool over the entire sequence length, ok.
        x = x.mean(dim=1)# global average pool over entire state

        x = self.MLP_end(x)
        
        # Output score
        score = self.output(x)
        
        return score
