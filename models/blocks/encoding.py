import torch
import torch.nn as nn

import math

class PositionalEncoding(nn.Module):

    def __init__(self, seq_len: int, d_model: int, dropout: float = 0.1, device="cpu"):
        super().__init__()
        self.device = device
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(seq_len, d_model, device=device)
        position = torch.arange(0, seq_len, device=device).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float, device=device) *
                         -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))
        # self.pos_embed = nn.Embedding(
        #     seq_len, d_model, device=device
        # )


    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, input_dim, embedding_dim]``
        """
        # B, seq_len, _ = x.shape
        # import pdb; pdb.set_trace()
        positions_encoding = self.pe

        # all_positions = torch.arange(0, seq_len, device=self.device)
        # positions_encoding = self.pos_embed(all_positions).unsqueeze(0).unsqueeze(2)
        # positions_encoding = self.pos_embed(all_positions).unsqueeze(0)
        # import pdb; pdb.set_trace()
        # print(positions_encoding.shape)
        # import pdb; pdb.set_trace()
        return x + positions_encoding

        # return x + self.pe

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_size = 1
    seq_len = 150
    in_dim= 32
    d_model = 90
    pe = PositionalEncoding(seq_len, in_dim, device="cuda")
    plt.figure(figsize=(15, 5))
    emb = pe(torch.ones( (batch_size, seq_len, in_dim), device=device)) # Batch size, seq len, in dim
    # import pdb; pdb.set_trace()
    plt.plot(np.arange(seq_len), emb.data.detach().cpu().numpy().squeeze())
    # plt.legend(["dim %d"%p for p in [4,5,6,7]])
    plt.savefig('encoding.png')
    # plt.show()