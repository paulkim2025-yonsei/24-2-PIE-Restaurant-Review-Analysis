# src/embedding_utils.py
from sentence_transformers import SentenceTransformer

class SbertEmbedding:

    def __init__(self, model_name='snunlp/KR-SBERT-V40K-klueNLI-augSTS'):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts, convert_to_tensor=True):
        return self.model.encode(texts, convert_to_tensor=convert_to_tensor)
