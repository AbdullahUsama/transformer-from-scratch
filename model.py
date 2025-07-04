import torch 
import torch.nn as nn
import math

import InputEmbeddings
import PositionalEncoding
from attention import MutliHeadAttentionBlock
from encoder_decoder_block import EncoderBlock, DecoderBlock
from encoder_decoder_block import Encoder, Decoder, ProjectionLayer
from norm import FeedForwardBlock
    
class Transformer(nn.Module):
    def __init__(self, encoder:Encoder, decoder:Decoder, src_embed: InputEmbeddings, tgt_embed:InputEmbeddings, src_pos:PositionalEncoding,tgt_pos:PositionalEncoding,  projection_layer:ProjectionLayer)->None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output,  src_mask, tgt,tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size:int, tgt_vocab_size:int, src_seq_len:int, d_model:int = 512, N:int=6, h:int=8, dropout:float=0.1, d_ff:int=2048)->Transformer:
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, src_seq_len, dropout)

    encoder_block = []
    for _ in range(N):
        encoder_self_attention_block = MutliHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)        
        encoder_block.append(encoder_block)

    decoder_block = []
    for _ in range(N):
        decoder_self_attention_block = MutliHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MutliHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block.append(DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout))

    encoder = Encoder(nn.ModuleList(encoder_block))
    decoder = Decoder(nn.ModuleList(decoder_block))

    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
