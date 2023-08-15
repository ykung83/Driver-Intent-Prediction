# embedding for gaze and time step
class EmbeddingLayer(Layer):
    def __init__(self, sequence_length, input_size, embed_dim):
        super(EmbeddingLayer, self).__init__()
        self.word_embedding = Embedding(input_dim=input_size, output_dim=embed_dim)
        self.position_embedding = Embedding(input_dim=sequence_length, output_dim=embed_dim)

    def call(self, tokens):
        sequence_length = shape(tokens)[-1]
        all_positions = range(start=0, limit=sequence_length, delta=1)
        positions_encoding = self.position_embedding(all_positions)
        words_encoding = self.word_embedding(tokens)
        return positions_encoding + words_encoding