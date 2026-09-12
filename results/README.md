# Results

Git에는 재현에 필요한 작은 지표와 표만 저장합니다. 영상, 전체 CSV, 모델 바이너리, 학습 캐시는 제외합니다.

`tcn_v1/`에는 현재 classifier 학습 이력과 held-out test confusion matrix가 있습니다.

- test windows: 421
- correct: 418
- accuracy: 약 99.3%
- SWIMMING: 161/161
- FLOATING: 225/227
- ACTIVE: 32/33

이 값은 현재 로컬 데이터 분할 결과입니다. 실제 현장 일반화 성능과 안전 성능은 별도 시험으로 평가해야 합니다.
