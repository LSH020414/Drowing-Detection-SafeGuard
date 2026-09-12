# Data pipeline

대용량 원본 영상과 생성 CSV는 저장소에 포함하지 않습니다. 이 폴더에는 재현에 필요한 코드, 특징 스키마, 데이터 출처와 배포 주의사항만 둡니다.

## 상태 분류 데이터 구성

| 최종 상태 | 프로젝트 기록상 원천 | 구성 |
|---|---|---|
| `SWIMMING` | SwimXYZ + 프로젝트 내부 Swim2 4영법 데이터 | SwimXYZ 정상 수영과 자유형·배영·평영·접영 영상을 Head 검수·시계열화하고 train 증강 적용 |
| `FLOATING` | 실제 YouTube 부유 장면 + Blender 합성 | 실제 구간을 추출하고 부족한 움직임을 합성 sequence로 보강 |
| `ACTIVE_DROWNING` | 실제 YouTube 익수 장면 + Blender 합성 | 실제 활동적 익수 구간, ACTIVE 합성 sequence, train 증강 |
| `PASSIVE_DROWNING` | 별도 classifier 학습 데이터 없음 | 동일 Track의 연속 LOST 시간이 임계값을 넘으면 CPU rule로 override |

최종 builder는 이미 정리된 `SwimXYZ_NORMAL_CLEAN`, `NORMAL_4STROKES_AUGMENTED_V1`, `FLOATING_FINAL`, `ACTIVE_AUGMENTED_V1`을 읽습니다. 따라서 저장소 코드만으로는 각 통합 CSV 안에서 실제 영상과 Blender가 차지하는 세부 비율까지 역추적할 수 없습니다. 재학습 시에는 source manifest와 원본 window 목록을 함께 보관해야 합니다.

## 최종 classifier 수량

| 클래스 | Train | Validation | Test | 합계 |
|---|---:|---:|---:|---:|
| `SWIMMING` | 1,253 | 127 | 161 | 1,541 |
| `FLOATING` | 675 | 140 | 227 | 1,042 |
| `ACTIVE` | 1,000 | 32 | 33 | 1,065 |
| 전체 | 2,928 | 299 | 421 | 3,648 |

전체 sequence row는 `3,648 windows × 50 timestep = 182,400 rows`입니다. 신경망은 3-class이고 시스템의 최종 출력은 PASSIVE override를 포함한 4-state입니다.

## 데이터 출처

### CrowdHuman

- 용도: Head Detector 초기 학습
- 제공 annotation: head box(`hbox`), visible region box(`vbox`), full-body box(`fbox`)
- 공식 사이트: [CrowdHuman](https://www.crowdhuman.org/)
- annotation·다운로드·이용 조건: [CrowdHuman download](https://www.crowdhuman.org/download.html)

CrowdHuman 이용 조건은 비상업적 연구·교육 목적 사용과 이미지 재배포 금지를 포함합니다. 따라서 원본 이미지나 재패키징 데이터는 이 Git 저장소에 넣지 않습니다.

당시 CrowdHuman 기반 detector 결과:

```text
Precision : 0.9010
Recall    : 0.8672
mAP50     : 0.8162
mAP50-95  : 0.5224
```

CrowdHuman으로 일반적인 머리 검출 능력을 확보한 뒤 실제 수영 환경 Head label로 보완·검증했습니다.

### SwimXYZ

- 용도: 정상 수영 `SWIMMING`
- 특징: 합성 수영 동작·영상과 2D/3D joint annotation을 제공하는 연구 데이터셋
- 논문: [SwimXYZ: A large-scale dataset of synthetic swimming motions and videos](https://arxiv.org/abs/2310.04360)
- 프로젝트 자료: [SwimXYZ research page](https://g-fiche.github.io/research-pages/swimxyz/)

### Swim2

- 용도: 실제 수영 환경의 자유형·배영·평영·접영 `SWIMMING` 데이터
- 처리: 10fps frame 변환, CVAT Head 검수, 잠김 구간 `detected=0`, sequence 생성

`Swim2`는 현재 프로젝트에서 사용한 내부 명칭만 기록되어 있습니다. 공개 데이터셋의 정확한 공식 명칭, 배포 페이지, 라이선스는 확인 전이므로 외부 배포 시 반드시 원본 출처를 추가해야 합니다.

### 실제 YouTube 영상

- 용도: `FLOATING` 실제 부유 장면과 `ACTIVE_DROWNING` 실제 익수 장면
- 기록된 채널: [프로젝트 영상 출처 채널](https://www.youtube.com/channel/UCnERyC7dwJwTvEyzYz6uxHw)
- 처리: 필요한 시간 구간 추출 → 10Hz sampling → 5초/50-step sequence

저장소에는 영상 자체를 포함하지 않습니다. 학습·발표·재배포 전에 개별 영상의 라이선스, 저작권자 허가, 인물 개인정보 및 플랫폼 약관을 확인해야 합니다. 링크는 출처 기록이며 재사용 허가를 의미하지 않습니다.

### Blender/Omniverse

- 용도: 부족한 `FLOATING`, `ACTIVE_DROWNING` 움직임 패턴 보조
- 형태: video가 아닌 image sequence
- 확인된 초기 생성량: ACTIVE 10 windows/500 rows, FLOATING 4 windows/200 rows

합성 데이터는 주 데이터가 아니라 train augmentation을 위한 보조 데이터로 사용했습니다.

## Head annotation 규칙

CVAT에서 Head bounding box와 Track ID를 관리했습니다.

```text
head visible
→ bbox + 동일 Track ID

head submerged
→ 새 객체를 만들지 않음
→ 기존 Track ID 유지
→ outside=True / detected=0

head reappears
→ 기존 Track ID 계속 사용
```

이 규칙으로 LOST, 재검출, 연속 미검출 시간이 실제 시계열 특징으로 남습니다. CVAT 기능은 [공식 annotation editor 문서](https://docs.cvat.ai/docs/annotation/annotation-editor/)를 참고합니다.

## 최종 시계열 규격

- sampling: 10Hz
- window: 5초
- length: 50 timestep
- features: 14
- classifier label: `SWIMMING`, `FLOATING`, `ACTIVE`
- `PASSIVE_DROWNING`: classifier label이 아니라 장시간 LOST override

```text
detected, confidence, cx_norm, cy_norm, head_scale,
dx, dy, speed, scale_change,
lost_event, redetect_event, consecutive_missing,
visible_ratio, missing_ratio
```

50행이 아닌 window는 제외합니다. 좌표와 scale의 누락값은 보간하지만 `detected`, LOST, 재검출 정보는 별도 feature로 보존합니다.

## Train/validation/test 분리

- seed: 42
- 기본 비율: video 단위 70%/15%/15%
- SwimXYZ와 FLOATING: `video_id` 단위 분리
- 4영법 SWIMMING: 증강 train + 실제 validation/test
- ACTIVE: 증강 train + 실제 validation/test
- 데이터 출처 prefix를 `window_id`, `video_id`에 붙여 ID 충돌 방지

인접 window가 서로 다른 split에 들어가는 frame-level 누수를 피하기 위해 video 단위로 나눴습니다.

## 코드 구성

- `extraction/extract_swim2_real_normal.py`: Swim2 실제 정상 수영 시계열 추출
- `extraction/extract_blender_image_sequences_v12.py`: Blender ACTIVE/FLOATING 시계열 추출
- `validation/test_swim2_three_strokes_cascade_v2.py`: 3영법 person→head cascade 검증
- `validation/test_swim2_person_head_cascade_0039.py`: 단일 영상/Breaststroke 포함 기준 검증
- `processing/build_final_classifier_dataset_v1.py`: source/video window 단위 최종 split 구성
- `schema/FEATURE_SCHEMA.csv`: classifier 입력 특징 정의

## 데이터 보관 원칙

- Git: 코드, 스키마, 작은 지표, 출처·해시·라이선스 기록
- 외부 저장소: 원본 영상, frame, 전체 CSV, annotation export
- 공개 금지: CrowdHuman 원본 이미지 재배포, 사용 허가를 확인하지 않은 YouTube 영상
- 재현성 기록: 원본 URL/버전, 다운로드 날짜, 파일 SHA-256, split manifest, augmentation seed
