import torch
import torch.nn as nn

class CustomTransformer(nn.Module):
    def __init__(self): #, embedding, d_model=512, nhead=8, num_encoder_layers=6):
        super(CustomTransformer, self).__init__()
        
        # self.embedding = embedding
        # self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers)
        
    def forward(self, x):
        pass
        # embedded = self.embedding(x)
        # output = self.transformer(embedded)
        # return output