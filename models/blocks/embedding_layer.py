from torch.nn import Embedding, Layer

# embedding for gaze and time step
class EmbeddingLayer(Layer):
    def __init__(self, model_cfg):
        super(EmbeddingLayer, self).__init__()
        self.model_cfg = model_cfg

        embedding_cfg   = self.model_cfg['EMBEDDING_LAYER']

        EMBEDDING_DIM   = embedding_cfg['EMBEDDING_DIM']
        SEQ_LEN         = embedding_cfg['SEQ_LEN']
        INPUT_SIZE      = embedding_cfg['INPUT_SIZE']

        self.word_embedding = Embedding(input_dim=INPUT_SIZE, output_dim=EMBEDDING_DIM)
        self.position_embedding = Embedding(input_dim=SEQ_LEN, output_dim=EMBEDDING_DIM)

    def call(self, tokens):
        sequence_length = tokens.shape[-1]
        all_positions = range(start=0, limit=sequence_length, delta=1)
        positions_encoding = self.position_embedding(all_positions)
        words_encoding = self.word_embedding(tokens)
        return positions_encoding + words_encoding