import torch
import torch.nn as nn
from sublayers import MultiHeadAttention, PositionwiseFeedForward


class EncoderLayer(nn.Module):
    '''compose with two sublayers'''
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(n_head,
                                            d_model,
                                            d_k,
                                            d_v,
                                            dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model,
                                               d_inner,
                                               dropout=dropout)

    def forward(self, enc_input, self_attn_mask=None):
        enc_output, enc_self_attn = self.self_attn(enc_input,
                                                   enc_input,
                                                   enc_input,
                                                   mask=self_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_self_attn


class DecoderLayer(nn.Module):
    '''compose with three layers'''
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(n_head,
                                            d_model,
                                            d_k,
                                            d_v,
                                            dropout=dropout)
        self.enc_dec_attn = MultiHeadAttention(n_head,
                                               d_model,
                                               d_k,
                                               d_v,
                                               dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model,
                                               d_inner,
                                               dropout=dropout)

    def forward(self,
                dec_input,
                enc_output,
                self_attn_mask=None,
                enc_dec_attn_mask=None):
        dec_output, dec_self_attn = self.self_attn(dec_input,
                                                   dec_input,
                                                   dec_input,
                                                   mask=self_attn_mask)
        dec_output, enc_dec_attn = self.enc_dec_attn(dec_output,
                                                     enc_output,
                                                     mask=enc_dec_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_self_attn, enc_dec_attn
