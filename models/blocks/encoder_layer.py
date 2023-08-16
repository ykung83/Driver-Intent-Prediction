from torch.nn import MultiHeadAttention, LayerNormalization, Layer, Sequential, Dense

#encoder layer
class EncoderLayer(Layer):
    def __init__(self, model_cfg):
        super(EncoderLayer, self).__init__()# Multihead attention layer
        self.model_cfg      = model_cfg

        encoder_cfg   = self.model_cfg['ENCODING_LAYER']

        TOTAL_HEADS     = encoder_cfg['N_HEADS']
        EMBEDDING_DIM   = self.model_cfg['EMBEDDING_LAYER']['EMBEDDING_DIM']
        DENSE_UNITS     = encoder_cfg['DENSE_UNITS']

        self.multihead = MultiHeadAttention(num_heads=TOTAL_HEADS, key_dim=EMBEDDING_DIM)# Feed forward network layer
        self.nnw = Sequential([Dense(DENSE_UNITS, activation="relu"), Dense(EMBEDDING_DIM)])# Normalization
        self.normalize_layer = LayerNormalization()

    def call(self, inputs):
        attn_output = self.multihead(inputs, inputs)
        normalize_attn = self.normalize_layer(inputs + attn_output)
        nnw_output = self.nnw(normalize_attn)
        final_output = self.normalize_layer(normalize_attn + nnw_output)
        return final_output