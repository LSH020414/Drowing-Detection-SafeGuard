# Data pipeline

대용량 원본 영상과 생성 CSV는 저장소에 포함하지 않습니다. 이 폴더에는 재현에 필요한 코드와 작은 스키마만 둡니다.

## 구성

- `extraction/extract_swim2_real_normal.py`: Swim2 실제 정상 수영 시계열 추출
- `extraction/extract_blender_image_sequences_v12.py`: Blender ACTIVE/FLOATING 시계열 추출
- `validation/test_swim2_three_strokes_cascade_v2.py`: 3영법 person→head cascade 검증
- `validation/test_swim2_person_head_cascade_0039.py`: 단일 영상/Breaststroke 포함 기준 검증
- `processing/build_final_classifier_dataset_v1.py`: source window 단위로 train/validation/test 구성
- `schema/FEATURE_SCHEMA.csv`: classifier 입력 특징 정의

## 최종 시계열 규격

- sampling: 10Hz
- window: 5초
- length: 50 timestep
- features: 14
- classifier label: `SWIMMING`, `FLOATING`, `ACTIVE`
- `PASSIVE_DROWNING`: classifier label이 아니라 장시간 LOST override

원본 데이터는 개인 식별 가능성, 라이선스, 용량을 확인한 뒤 승인된 외부 저장소에 둡니다. 저장소에는 다운로드 위치·버전·해시만 기록하는 방식을 권장합니다.
