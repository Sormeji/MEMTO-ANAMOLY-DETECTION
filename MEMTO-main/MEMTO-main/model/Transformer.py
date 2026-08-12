import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn_layer import AttentionLayer
from .embedding import TokenEmbedding, InputEmbedding
from .ours_memory_module import MemoryModule

class EncoderLayer(nn.Module):
    def __init__(self, attn, d_model, d_ff=None, dropout=0.1, activation='relu'):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff if d_ff is not None else 4 * d_model
        self.attn_layer = attn
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x):
        out = self.attn_layer(x)
        x = x + self.dropout(out)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)

class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x):
        for attn_layer in self.attn_layers:
            x = attn_layer(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

class Decoder(nn.Module):
    def __init__(self, d_model, c_out, d_ff=None, activation='relu', dropout=0.1):
        super(Decoder, self).__init__()
        # d_model here is the 1024 coming from the memory module
        self.out_linear = nn.Linear(d_model, c_out)

    def forward(self, x):
        # Force the projection from 1024 back to 38
        return self.out_linear(x)

class TransformerVar(nn.Module):
    def __init__(self, win_size, enc_in, c_out, n_memory, shrink_thres=0, \
                 d_model=512, n_heads=8, e_layers=3, d_ff=512, dropout=0.0, activation='gelu', \
                 device=None, memory_init_embedding=None, memory_initial=False, phase_type=None, dataset_name=None):
        super(TransformerVar, self).__init__()

        self.memory_initial = memory_initial

        # Step 1: Input Projection (38 -> 512)
        self.projection = nn.Linear(enc_in, d_model)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(win_size, d_model, n_heads, dropout=dropout), 
                    d_model, d_ff, dropout=dropout, activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer = nn.LayerNorm(d_model)
        )

        self.mem_module = MemoryModule(n_memory=n_memory, fea_dim=d_model, shrink_thres=shrink_thres, 
                                       device=device, memory_init_embedding=memory_init_embedding, 
                                       phase_type=phase_type, dataset_name=dataset_name)

        # Step 2: Decoder Projection (1024 -> 38)
        # Note: 2 * d_model is used because MemoryModule typically concats query + memory
        self.weak_decoder = Decoder(2 * d_model, c_out)

    def forward(self, x):
        # x: [Batch, Seq_Len, 38]
        x = self.projection(x) 
        
        queries = self.encoder(x) 
        
        outputs = self.mem_module(queries)
        out_mem = outputs['output'] # Likely [Batch, Seq_Len, 1024]
        attn = outputs.get('attn', None)
        memory_item_embedding = outputs.get('memory_init_embedding', None)
        mem = self.mem_module.mem
        
        if self.memory_initial:
            # We must still project back to 38 so the Loss function doesn't crash
            out_final = self.weak_decoder(out_mem)
            return {"out": out_final, "memory_item_embedding": None, "queries": queries, "mem": mem}
        else:
            out_final = self.weak_decoder(out_mem)
            return {
                "out": out_final, 
                "memory_item_embedding": memory_item_embedding, 
                "queries": queries, 
                "mem": mem, 
                "attn": attn
            }