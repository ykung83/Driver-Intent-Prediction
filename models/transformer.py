import torch
from torch import Tensor
import torch.nn as nn

from .blocks.encoding import PositionalEncoding
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerModel(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        # N_CLASSES = model_cfg['TRANSFORMER']['N_CLASSES']
        # N_ENC_LAYERS = model_cfg['TRANSFORMER']['ENCODER_LAYERS']
        # DECODER_LAYERS = model_cfg['TRANSFORMER']['DECODER_LAYERS']
        # MODEL_DIM       = model_cfg['TRANSFORMER']['MODEL_DIM']
        # MHA_HEADS       = model_cfg['ENCODER_LAYER']['MHA_HEADS']
        # HIDDEN_DIM      = model_cfg['ENCODER_LAYER']['HIDDEN_DIM']
        # EMB_DIM         = model_cfg['EMBEDDING_LAYER']['EMBEDDING_DIM']
        dropout = 0.5

        INPUT_DIM=32
        EMBEDDING_DIM=128
        HIDDEN_DIM=128
        SEQ_LEN=150
        MHA_HEADS=4
        N_ENC_LAYERS=6
        N_CLASSES=5

        #1
        self.embedding  = nn.Linear(INPUT_DIM, EMBEDDING_DIM)   # NUM_IMAGES x INDIM -> NUM_IMAGES x EMBDIM
        self.cls_token  = nn.Parameter(torch.rand(1, EMBEDDING_DIM))    # 1 x EMBDIM
        self.pos_embed  = PositionalEncoding(EMBEDDING_DIM, SEQ_LEN)
        encoder_layers  = TransformerEncoderLayer(EMBEDDING_DIM, MHA_HEADS, HIDDEN_DIM, dropout) # MODEL_DIM -> HIDDEN_DIM
        self.transformer_encoder = TransformerEncoder(encoder_layers, N_ENC_LAYERS) # MODEL_DIM -> HIDDEN_DIM
        self.mlp_head = nn.Linear(HIDDEN_DIM, N_CLASSES)
        import pdb; pdb.set_trace()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, data):
        #1 Embed input features
        tokens = self.embedding(data)

        num_tokens = tokens.shape[0]
        class_tokens = cls_tokens.repeat(num_tokens, 1)
        tokens = torch.stack((tokens, num_tokens))

        #2 Add positional information to embeddings
        tokens = self.pos_embed(tokens)

        #3 Encode embedding + cls token to hidden features
        hidden_feats = self.transformer_encoder(tokens)

        #4 MLP hidden features to class prediction vector
        output = self.mlp_head(hidden_feats)

        return output