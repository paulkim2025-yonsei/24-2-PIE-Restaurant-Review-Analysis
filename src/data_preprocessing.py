# src/data_preprocessing.py

import re
import numpy as np
import pandas as pd
from konlpy.tag import Okt
import pickle

class DataPreprocessor:

    def __init__(self, stopwords_path=None):
        self.okt = Okt()
        self.stopwords = []
        if stopwords_path:
            with open(stopwords_path, 'rb') as f:
                self.stopwords = pickle.load(f)

    def load_dataframes(self, paths):

        dfs = []
        for csv_path in paths.get('csv', []):
            df = pd.read_csv(csv_path, encoding='utf-8')
            dfs.append(df)
        for xlsx_path in paths.get('xlsx', []):
            df = pd.read_excel(xlsx_path)
            dfs.append(df)
        merged_df = pd.concat(dfs, axis=1)
        return merged_df

    def filter_short_texts(self, series, min_length=10):
        return series.dropna().apply(lambda x: x if len(str(x)) > min_length else None).dropna()

    def tokenize_and_remove_stopwords(self, text):
        tokens = self.okt.morphs(str(text))
        filtered = [word for word in tokens if word not in self.stopwords and word.isalpha()]
        return filtered

    def get_embedding_input(self, tokenized_text, ft_model, max_len=800, emb_dim=300):
        vectors = [ft_model.get_word_vector(word) for word in tokenized_text]
        if len(vectors) < max_len:
            pad_len = max_len - len(vectors)
            vectors += [np.zeros(emb_dim)] * pad_len
        else:
            vectors = vectors[:max_len]
        return np.array(vectors)
