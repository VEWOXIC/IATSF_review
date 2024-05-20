__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np


from layers.RevIN import RevIN
from layers.TGTSF_torch import text_encoder, text_temp_cross_block, positional_encoding, TS_encoder


class Model(nn.Module):
    def __init__(self, configs):
        
        super().__init__()
        
        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model

        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        
        individual = configs.individual
    
        patch_len = configs.patch_len
        self.patch_len = patch_len
        stride = configs.stride
        self.stride = stride
        padding_patch = configs.padding_patch
        
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last

        self.out_attn_weights = configs.out_attn_weights
        
        self.TS_encoder = TS_encoder(embedding_dim=d_model, layers=n_layers, num_heads=n_heads, dropout=dropout, patch_len=patch_len, stride=stride, causal=True, input_len=configs.seq_len)

        cross_encoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim,
                                                    nhead=num_heads,
                                                    dropout=dropout,
                                                    dim_feedforward=embedding_dim*4,
                                                    activation='gelu',
                                                    batch_first=True,
                                                    norm_first=True)
        
        encoder_norm = nn.LayerNorm(embedding_dim, eps=1e-5)
        self.text_encoder = nn.TransformerDecoder(cross_encoder_layer, cross_layer, norm=encoder_norm)
        for p in self.text_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # self.text_encoder = text_encoder(cross_layer=configs.cross_layers, self_layer=configs.self_layers, embedding_dim=configs.text_dim, num_heads=configs.n_heads, dropout=configs.dropout)

        # position embedding for text
        # self.dropout = nn.Dropout(dropout)

        self.dropout = dropout

        q_len = context_window

        self.mixer = text_temp_cross_block(text_embedding_dim=configs.text_dim, temp_embedding_dim=d_model, num_heads=configs.n_heads, dropout=0.0, self_layer=configs.mixer_self_layers)

        patch_num = int((context_window - patch_len)/stride + 1)
        self.patch_num = patch_num
        # if padding_patch == 'end': # can be modified to general case
        #     self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
        #     patch_num += 1

        # Head
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.individual = individual

        assert stride == patch_len, 'stride should be equal to patch_len for token_decoder head'
        self.head = nn.Linear(d_model, patch_len)
    
    
    def forward(self, x, news, description, news_mask):           # x: [Batch, Input length, Channel] news: [Batch, l, news_num, text_dim] description: [b, l, c, d]

        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var=torch.var(x, dim=1, keepdim=True)+ 1e-5
        # print(x_var)
        x = x / torch.sqrt(x_var)

        x = self.TS_encoder(x)    # x: [bs x nvars x d_model x patch_num]

        # t = self.text_encoder(news, description, news_mask) # t: [bs, l, nvars, d_model] # 添加positional embedding!
        t = self.text_encoder(tgt=text_emb, memory=news_emb, memory_key_padding_mask=news_mask)

        x, mix_weights = self.mixer(t, x) # x: [bs, patch_num, nvars, d_model]

        x = self.head(x) # x: [bs, patch_num, nvars, patch_len]
        x = x.permute(0,2,1,3)
        x = x.reshape(x.shape[0], x.shape[1], -1) # x: [bs, nvars, patch_num*patch_len]
        x = x.permute(0,2,1) # x: [bs, patch_num*patch_len, nvars]


        
        # denorm
        x= x * torch.sqrt(x_var) + x_mean

        if self.out_attn_weights:
            return x, mix_weights
        else:
            return x