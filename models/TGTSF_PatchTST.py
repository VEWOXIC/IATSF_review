__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.PatchTST_backbone import TSTiEncoder, Flatten_Head
from layers.PatchTST_layers import series_decomp
from layers.RevIN import RevIN
from layers.TGTSF import text_encoder, text_temp_cross_block, positional_encoding

class TGTSF_PatchTST_backbone(nn.Module):
    def __init__(self, c_in:int, context_window:int, target_window:int, patch_len:int, stride:int, max_seq_len:Optional[int]=1024, 
                 n_layers:int=3, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
                 d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
                 padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout = 0, padding_patch = None,
                 pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine = True, subtract_last = False,
                 verbose:bool=False, notrans=False, **kwargs):
        
        super().__init__()
        
        
        
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len)/stride + 1)
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            patch_num += 1
        
        # Backbone 
        self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose,notrans=notrans, **kwargs)


        
    
    def forward(self, z):                                                                   # z: [bs x nvars x seq_len]
        # print(z.shape, self.stride)
        # do patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # z: [bs x nvars x patch_num x patch_len]
        # print('unfold'+str(z.shape))
        z = z.permute(0,1,3,2)                                                              # z: [bs x nvars x patch_len x patch_num]
        
        # model
        z = self.backbone(z)                                                                # z: [bs x nvars x d_model x patch_num]
        # z = self.head(z)                                                                    # z: [bs x nvars x target_window]  # head is removed
        return z
    
    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                    nn.Conv1d(head_nf, vars, 1)
                    )


class Model(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        
        individual = configs.individual
    
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size

        notrans = configs.notrans

        self.out_attn_weights = configs.out_attn_weights
        self.mixer_type = configs.mixer_type

        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        
        
        self.PatchTST = TGTSF_PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, notrans=notrans, **kwargs)
        
        self.text_encoder = text_encoder(cross_layer=configs.cross_layers, self_layer=configs.self_layers, embedding_dim=configs.text_dim, num_heads=configs.n_heads, dropout=0.0)

        # position embedding for text
        # self.dropout = nn.Dropout(dropout)

        self.dropout = dropout

        q_len = context_window
        self.W_pos = positional_encoding(pe, learn_pe, q_len, configs.text_dim)

        if self.mixer_type == 'cross':

            self.mixer = text_temp_cross_block(text_embedding_dim=configs.text_dim, temp_embedding_dim=d_model, num_heads=configs.n_heads, dropout=0.0, self_layer=configs.mixer_self_layers)

        elif self.mixer_type == 'append':
            pass
            self.mixer = text_temp_cross_block(text_embedding_dim=configs.text_dim, temp_embedding_dim=d_model, num_heads=configs.n_heads, dropout=0.0, self_layer=configs.mixer_self_layers, append=True)

        elif self.mixer_type == 'adaLN':
            pass

        patch_num = int((context_window - patch_len)/stride + 1)
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            patch_num += 1

        # Head
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head: 
            self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout) # custom head passed as a partial func with all its kwargs
        elif head_type == 'flatten': 
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)
        elif head_type == 'token_decoder':
            assert stride == patch_len, 'stride should be equal to patch_len for token_decoder head'
            self.head = nn.Linear(d_model, patch_len)
    
    
    def forward(self, x, news, description, news_mask):           # x: [Batch, Input length, Channel] news: [Batch, l, news_num, text_dim] description: [b, l, c, d]

        if self.revin: 
            x = self.revin_layer(x, 'norm')


        x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]

        x = self.PatchTST(x)    # x: [bs x nvars x d_model x patch_num]

        t = self.text_encoder(news, description, news_mask) # t: [bs, l, nvars, d_model] # 添加positional embedding!

        # print(f"\n{t.shape}--{self.W_pos.shape}\n")
        # t = self.dropout(t + self.W_pos)  # seems to be weird
        
        if self.training:
            # create a mask tensor with p=0.5 with shape [bs, l, nvars]
            mask = torch.empty(t.shape[:-1]).uniform_(0, 1) < self.dropout
            mask = mask.to(t.device)

            t = t * mask.unsqueeze(-1) # t: [bs, l, nvars, d_model]
        t = t + self.W_pos


        # print(f"\n{t.shape}--{self.W_pos.shape}\n")

        x = x.permute(0, 3, 1, 2) # x: [bs, patch_num, nvars, d_model]

        x, mix_weights = self.mixer(t, x) # x: [bs, patch_num, nvars, d_model]

        if self.head_type == 'token_decoder':
            x = self.head(x) # x: [bs, patch_num, nvars, patch_len]
            x = x.permute(0,2,1,3)
            x = x.reshape(x.shape[0], x.shape[1], -1) # x: [bs, nvars, patch_num*patch_len]
            x = x.permute(0,2,1) # x: [bs, patch_num*patch_len, nvars]
        else:

            x = x.permute(0, 2, 3, 1) # x: [bs, nvars, d_model, patch_num]
            x = self.head(x)  # z: [bs x nvars x target_window]
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]

        
        # denorm
        if self.revin: 

            x = self.revin_layer(x, 'denorm')

        if self.out_attn_weights:
            return x, mix_weights
        else:
            return x