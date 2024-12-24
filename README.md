# 24-2-PIE-Restaurant-Review-Analysis
Recommendation System for Shinchon Restaurants: Analyzing Reviews Using LSTM-Attention and Sentence-Transformer Models

대학가 식당은 학생들이 가장 자주 이용하는 공간이지만, 식당이 매우 많고 리뷰 이벤트로 인한 광고성 리뷰 등으로 인해 선택에 어려움을 겪는 경우가 많다. 이러한 문제를 해결하고자, 광고성 리뷰를 걸러내고 신뢰할 수 있는 평가를 기반으로 맛, 가격, 위생 등 중요한 요소를 종합적으로 고려한 식당 추천 시스템을 개발하기로 하였다.

![image](https://github.com/user-attachments/assets/c351a056-44e5-4a9b-af29-07d436b3fbeb)

먼저, 네이버 블로그 리뷰 약 800개를 직접 크롤링하였다. 크롤링한 후, "내돈내산", "솔직후기"와 같은 키워드가 포함되면 비광고성 리뷰(0)으로, 그렇지 않으면 광고성 리뷰(1)로 라벨링하였다.

이후 LSTM-Attention 모델에 라벨링 데이터를 학습 데이터로 하여 Train하였다. 이때, CheckAttention으로 Attention되는 키워드를 확인한 후, 핵심 키워드가 아닌 단어를 수동으로 불용어처리하였다. 예를 들어, '갈비'라는 키워드가 있으면 무조건 광고성 리뷰로 분류하는 오류를 막기 위해 '갈비'를 불용어처리하였다. F1-score는 0.9811로 준수한 편이었다.

![image](https://github.com/user-attachments/assets/58c5a694-b143-4e96-8d83-038f39e7e9e4)

그 다음, 앞서 사용한 네이버 블로그 리뷰와 독립적인 네이버 지도 기반 신촌 식당 리뷰를 약 5,000개 크롤링하였다. 이후Hugging Face의 snunlp/KR-SBERT-V40K-klueNLI-augSTS라는 Sentence-Transformer를 통해 문장 단위 임베딩을 진행하였다. 그후, 사전에 키워드와 키워드의 sentiment와 연관된 키워드를 정의하였다. 사전 정의된 핵심 키워드와의 유사도가 일정 수준 이상이면 해당 라벨의 Sentiment로 할당하였다. 마지막으로, 긍정도는 긍정리뷰수/(긍정리뷰수+부정리뷰수)로 측정하여, 해당 식당의 해당 라벨에 대한 긍정도로 하였다.

최종적으로, 사용자가 입력한 라벨 가중치를, 긍정도와 곱하여 가장 점수가 높은 식당을 추천하는 추천 시스템을 완성하였다.

[구현 링크]
https://two024-2-pie-ds.onrender.com/ 


References
- 유동관, 임한길, & 채동규. (2021). LSTM-Attention 모델 기반의 광고성 리뷰 탐지 및 핵심 단어 추출 연구. 한국소프트웨어종합학술대회 논문집, 1505-1506.
- 조민서, & 박수현. (2024). AD Finder: 머신러닝을 활용한 실시간 광고 블로그 게시글 탐지. 한국컴퓨팅컨퍼런스, 42(1), 56-70
- 최현준, & 임규빈. (2022). 머신러닝 기반 음식점 추천 시스템 설계 및 구현. 디지털콘텐츠학회논문지, 18(6), 45-60.
