import torch
from torch import Tensor
import torch.nn as nn

from .blocks.encoding import PositionalEncoding
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerModel(nn.Module):
    def __init__(self, model_cfg, device="cuda:0"):
        super().__init__()
        self.model_cfg = model_cfg
        # N_CLASSES = model_cfg['TRANSFORMER']['N_CLASSES']
        # N_ENC_LAYERS = model_cfg['TRANSFORMER']['ENCODER_LAYERS']
        # DECODER_LAYERS = model_cfg['TRANSFORMER']['DECODER_LAYERS']
        # MODEL_DIM       = model_cfg['TRANSFORMER']['MODEL_DIM']
        # MHA_HEADS       = model_cfg['ENCODER_LAYER']['MHA_HEADS']
        # HIDDEN_DIM      = model_cfg['ENCODER_LAYER']['HIDDEN_DIM']
        # EMB_DIM         = model_cfg['EMBEDDING_LAYER']['EMBEDDING_DIM']
        dropout = 0.1

        self.MAX_INPUT_SIZE=512 # Scale input indices to this
        INPUT_DIM=32
        # INPUT_DIM = 4 # Road camera only
        # INPUT_DIM = 12 # Face camera only
        D_MODEL=128
        SEQ_LEN=150
        MHA_HEADS=2
        N_ENC_LAYERS=2
        HIDDEN_DIM=512
        N_CLASSES=5
        self.D_MODEL = torch.tensor(D_MODEL, dtype=torch.float)

        #1
        # self.embedding  = nn.Linear(INPUT_DIM, EMBEDDING_DIM, device=device)   # NUM_IMAGES x INDIM -> NUM_IMAGES x EMBDIM
        self.embedding = nn.Embedding(self.MAX_INPUT_SIZE, D_MODEL, device=device)
        # self.cls_token  = nn.Parameter(torch.rand(1, EMBEDDING_DIM, device=device, requires_grad=True))    # 1 x EMBDIM
        self.pos_embed  = PositionalEncoding(SEQ_LEN, INPUT_DIM, device=device)
        encoder_layers  = TransformerEncoderLayer(
            INPUT_DIM, MHA_HEADS, HIDDEN_DIM, activation='relu', 
            dropout=dropout, device=device, batch_first=True
        ) # MODEL_DIM -> HIDDEN_DIM
        self.transformer_encoder = TransformerEncoder(encoder_layers, N_ENC_LAYERS) # MODEL_DIM -> HIDDEN_DIM
        
        # Classification MLP Head
        self.mlp_head = nn.Sequential(
            nn.Linear((SEQ_LEN*INPUT_DIM), N_CLASSES, device=device),
            nn.ReLU(),
            nn.Softmax(dim=-1)
        )

    def forward(self, data):
        """
        data - B x N x F (batch size, num frames, feature size)
        """
        #0 Scale data to new range for embedding dim
        # scaled_data = data * (self.MAX_INPUT_SIZE-1)
        # scaled_data = scaled_data.long()

        #1 Embed input features
        # x = self.embedding(scaled_data) * torch.sqrt(self.D_MODEL)
        x = data
        # import pdb; pdb.set_trace()

        #2 Add cls token and 1d time based positional encoding
        # num_tokens = x.shape[0]

        # cls_tokens = self.cls_token.expand(num_tokens, -1, -1)

        # x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_embed(x)
        # B, S, F, E = x.shape
        B, S, F = x.shape
        # x = x.view(B, S, -1)

        # import pdb; pdb.set_trace()

        #3 Encode embedding + cls token to hidden features
        hidden_feats = self.transformer_encoder(x)

        # import pdb; pdb.set_trace()
        hidden_feats = hidden_feats.view(B, -1)
        #4 MLP hidden features to class prediction vector
        # out = hidden_feats[:, 0, :] # Get cls token only
        out = self.mlp_head(hidden_feats)

        return out
    
    def print_model_params(self):
        for p in self.parameters():
            if p.name:
                print(p.name, p.data)

    def print_model_grads(self):
        grads = []
        for p in self.parameters():
            print(p.requires_grad, p.grad.view(-1))
            grads.append(p.grad.view(-1))
        grads = torch.cat(grads)
        print(grads.shape)