# Results

Git에는 재현과 의사결정 근거에 필요한 작은 지표와 표만 저장합니다. 영상, 전체 sequence CSV와 학습 캐시는 제외하고 모델 바이너리는 `ai/models/`와 `deployment/pi5_hailo/`에서 Git LFS로 관리합니다.

`tcn_v1/`에는 현재 classifier 학습 이력과 held-out test confusion matrix가 있습니다.

- test windows: 421
- correct: 418
- accuracy: 약 99.3%
- SWIMMING: 161/161
- FLOATING: 225/227
- ACTIVE: 32/33

이 값은 현재 로컬 데이터 분할 결과입니다. 실제 현장 일반화 성능과 안전 성능은 별도 시험으로 평가해야 합니다.

추가 결과 폴더:

- `data/`: 4영법·ACTIVE augmentation·Blender·SwimXYZ aggregate
- `head_detector/`: hard-negative 파인튜닝 학습 이력
- `reid/`: Pool ReID 학습, SAME/DIFFERENT similarity와 threshold 결과
- `tracking/`: Task 90 GT 평가와 geometry/ReID threshold sweep

Task 90의 IoU 0.5 결과에서 Raw ByteTrack은 IDF1 `0.8821`, ID switch `63`이었고 V12 GeoMax 0.45 실험은 IDF1 `0.8442`, ID switch `53`이었습니다. 즉 V12는 이 시험에서 ID 전환을 줄였지만 IDF1 전체 성능이 항상 더 높지는 않았습니다. 현장 파라미터를 확정하기 전에 다른 영상으로 교차 검증해야 합니다.
