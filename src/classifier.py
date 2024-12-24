# src/classifier.py

import torch
from sentence_transformers import util
from .embedding_utils import SbertEmbedding

class ClassifierSBERT:
    def __init__(self, threshold=0.33):

        self.sbert = SbertEmbedding()
        self.class_names = ["클래스1", "클래스2", "클래스3", "클래스4",
                            "클래스5", "클래스6", "클래스7", "클래스8"]
        self.class_descriptions = ["맛있음", "위생", "서비스", "분위기",
                                   "위치 접근성", "대기 시간", "가성비", "가격"]
        self.class_embeddings = self.sbert.encode(self.class_descriptions, convert_to_tensor=True)
        self.threshold = threshold

    def classify_review(self, review):
        """
        단일 리뷰에 대해 SBERT 임베딩 -> 각 클래스와 유사도 계산 -> 임계값 이상인 클래스 반환
        """
        review_embedding = self.sbert.encode(review, convert_to_tensor=True)
        similarities = util.cos_sim(review_embedding, self.class_embeddings)[0]

        assigned_classes = []
        for i, sim in enumerate(similarities):
            if sim >= self.threshold:
                assigned_classes.append(self.class_names[i])
        return assigned_classes

    def classify_reviews(self, reviews):
        """
        여러 리뷰에 대해 클래스 할당
        """
        results = []
        for rv in reviews:
            assigned = self.classify_review(rv)
            results.append({'review': rv, 'assigned_classes': assigned})
        return results
