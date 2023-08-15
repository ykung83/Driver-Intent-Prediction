#encoder layer
class EncoderLayer(Layer):
    def __init__(self, total_heads, total_dense_units, embed_dim):
        super(EncoderLayer, self).__init__()# Multihead attention layer
        self.multihead = MultiHeadAttention(num_heads=total_heads, key_dim=embed_dim)# Feed forward network layer
        self.nnw = Sequential([Dense(total_dense_units, activation="relu"),
        Dense(embed_dim)])# Normalization
        self.normalize_layer = LayerNormalization()

    def call(self, inputs):
        attn_output = self.multihead(inputs, inputs)
        normalize_attn = self.normalize_layer(inputs + attn_output)
        nnw_output = self.nnw(normalize_attn)
        final_output = self.normalize_layer(normalize_attn + nnw_output)
        return final_output