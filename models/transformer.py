import torch
import torch.nn as nn

from blocks.embedding_layer import EmbeddingLayer
from blocks.encoder_layer import EncoderLayer

class CustomTransformer(nn.Module):
    def __init__(self, model_cfg):
        super(CustomTransformer, self).__init__()
        self.model_cfg = model_cfg
        N_CLASSES = model_cfg['TRANSFORMER']['N_CLASSES']
        ENCODER_LAYERS = model_cfg['TRANSFORMER']['ENCODER_LAYERS']
        DECODER_LAYERS = model_cfg['TRANSFORMER']['DECODER_LAYERS']
        INPUT_DIM       = model_cfg['TRANSFORMER']['INPUT_DIM']

        # self.embedding  = EmbeddingLayer(model_cfg)
        # self.encoder    = EncoderLayer(model_cfg)
        self.transformer = nn.Transformer(d_model=INPUT_DIM, nhead=N_CLASSES, num_encoder_layers=ENCODER_LAYERS, 
                            num_decoder_layers=DECODER_LAYERS, custom_encoder=EmbeddingLayer(model_cfg), custom_decoder=ncoderLayer(model_cfg))

        # embedding_layer = EmbeddingLayer(model_cfg)
        # encoder_layer = EncoderLayer(model_cfg)
        
    def forward(self, x):
        # embedded = self.embedding(x)
        output = self.transformer(x)
        return output