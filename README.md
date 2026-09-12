# Drowing Detection SafeGuard

수영장 영상에서 익수 위험자를 탐지·추적하고, 실제 수영장 좌표를 계산해 구조 튜브 발사 장치의 조준·제어까지 연결하는 임베디드 안전 시스템입니다.

> 원격 저장소 이름의 `Drowing` 철자는 기존 URL 호환을 위해 유지합니다.

## 문제 정의

일반 사람 검출은 수면 아래에 가려지는 신체와 반사광 때문에 수영장 환경에서 불안정합니다. 이 프로젝트는 수면 위에서 상대적으로 관찰 가능한 **머리**를 중심으로 검출하고, 머리의 이동뿐 아니라 `LOST`·재검출 패턴을 5초 시계열로 분석합니다. 단일 프레임의 모양만으로 익수를 판단하지 않고, 최근 행동 변화와 장시간 소실을 함께 사용합니다.

## 전체 시스템

```text
Camera Module 3 × 2
        ↓
Raspberry Pi 5 + Hailo-8
        ↓
Head Detector (pool_head_best.pt)
        ↓
ByteTrack + V12 Stable ID
        ↓
10 Hz × 5 s = 50 timestep
        ↓
Temporal CNN: SWIMMING / FLOATING / ACTIVE
        +
장시간 LOST rule: PASSIVE_DROWNING override
        ↓
Homography → 수영장 실제 X, Y 좌표
        ↓
USB Serial
        ↓
Arduino Nano
        ├─ BLD-50 × 2 → 상·하부 BLDC 발사 휠
        ├─ DMD-150    → 수평 DC 웜기어모터
        └─ A4988      → 수직 NEMA17 스테퍼
```

상세 설계는 [시스템 아키텍처](docs/ARCHITECTURE.md), 하드웨어 신호 구조는 [하드웨어 문서](hardware/README.md)를 참고합니다.

## 소프트웨어 파이프라인

1. `pool_head_best.pt`가 프레임별 머리 박스와 confidence를 검출합니다.
2. `bytetrack_pool.yaml`과 V12 ID 복구 로직이 동일 인물을 이어 붙입니다.
3. Track별 특징을 10Hz로 샘플링해 5초 길이의 50 timestep window를 만듭니다.
4. 14개 특징을 받는 Temporal CNN이 `SWIMMING`, `FLOATING`, `ACTIVE`를 분류합니다.
5. 연속 LOST 조건이 충족되면 분류기 결과보다 우선해 `PASSIVE_DROWNING`으로 전환합니다.
6. 보정 웹 도구의 homography로 픽셀을 미터 좌표로 바꾸고 제어 명령을 MCU에 전달합니다.

현재 V12 추적/규칙 엔진과 Temporal CNN 학습·내보내기 코드는 각각 확보되어 있습니다. **두 모듈을 Raspberry Pi/Hailo 실시간 루프로 완전히 결합하는 작업은 아직 남아 있습니다.**

## 저장소 구조

```text
.
├─ ai/
│  ├─ configs/        # ByteTrack 및 V12 rule 설정
│  ├─ pipeline/       # 최종 V12 머리 추적/규칙 파이프라인
│  ├─ training/       # 50×14 Temporal CNN 학습
│  ├─ export/         # ONNX/Hailo 변환 및 calibration 입력 생성
│  └─ models/         # 모델 배치 안내(바이너리는 기본 미포함)
├─ data/
│  ├─ extraction/     # Swim2·Blender 시계열 추출
│  ├─ processing/     # 최종 classifier dataset 구성
│  ├─ schema/         # 특징 스키마
│  └─ validation/     # Swim2 cascade 검증 코드
├─ demo/              # 발표/시연용 tracking video 생성
├─ hardware/          # Pi5 → Nano → motor driver 제어 구조
├─ pool-calibrator/   # 2-camera homography 보정 웹 도구
├─ docs/              # 아키텍처, 상태, 원본 이력
└─ results/           # 작은 지표·표만 버전 관리
```

## 주요 파일

| 파일 | 역할 |
|---|---|
| [`ai/pipeline/drowning_main_v12.py`](ai/pipeline/drowning_main_v12.py) | Head Detector + ByteTrack + Stable ID + V12 rule 기준 구현 |
| [`ai/configs/bytetrack_pool.yaml`](ai/configs/bytetrack_pool.yaml) | 최종 ByteTrack 설정 |
| [`ai/training/train_classifier_hailo_v1.py`](ai/training/train_classifier_hailo_v1.py) | 50 timestep × 14 feature Temporal CNN 학습 |
| [`data/processing/build_final_classifier_dataset_v1.py`](data/processing/build_final_classifier_dataset_v1.py) | train/validation/test classifier CSV 구성 |
| [`data/extraction/extract_blender_image_sequences_v12.py`](data/extraction/extract_blender_image_sequences_v12.py) | Blender ACTIVE/FLOATING 시계열 추출 |
| [`data/extraction/extract_swim2_real_normal.py`](data/extraction/extract_swim2_real_normal.py) | Swim2 실제 정상 수영 시계열 추출 |
| [`data/validation/test_swim2_three_strokes_cascade_v2.py`](data/validation/test_swim2_three_strokes_cascade_v2.py) | Swim2 3영법 cascade 검증 |
| [`data/validation/test_swim2_person_head_cascade_0039.py`](data/validation/test_swim2_person_head_cascade_0039.py) | Breaststroke 포함 단일 영상 cascade 기준 |
| [`demo/make_demo_tracking_video.py`](demo/make_demo_tracking_video.py) | 10Hz 표시 갱신·5px 양자화·시간별 상태 시연 영상 |
| [`pool-calibrator/`](pool-calibrator/) | 두 카메라 수영장 좌표 보정 웹 도구 |

원본 파일 경로와 SHA-256은 [SOURCE_MANIFEST.md](docs/SOURCE_MANIFEST.md)에 기록했습니다.

## 실행 방법

### 1. 환경

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-ai.txt
```

### 2. 모델 배치

`pool_head_best.pt`를 `ai/models/pool_head_best.pt`에 둡니다. 모델·데이터 관리 규칙과 검증 해시는 [ai/models/README.md](ai/models/README.md)에 있습니다.

### 3. V12 추적 실행

```powershell
python ai/pipeline/drowning_main_v12.py `
  --source "C:\path\input.mp4" `
  --model "ai/models/pool_head_best.pt" `
  --tracker "ai/configs/bytetrack_pool.yaml" `
  --reid-model "C:\path\reid_best.pt" `
  --output-dir "outputs"
```

`--reid-model`은 현재 V12 stable-ID 복구에 사용하는 별도 모델 경로입니다. 우선 입출력만 검사하려면 `--dry-run`을 추가합니다.

### 4. 시연 영상

`demo/make_demo_tracking_video.py` 상단의 `VIDEO_NAME`, `STATE_TIMELINE`, 모델/트래커 경로를 환경에 맞게 바꾼 뒤 실행합니다.

```powershell
python demo/make_demo_tracking_video.py
```

### 5. 데이터셋·TCN

추출·학습 스크립트는 확정 당시의 절대 경로를 보존합니다. 먼저 각 파일 상단의 `ROOT`, `*_PATH`, `*_DIR` 값을 로컬 데이터 위치로 바꿉니다. 권장 순서는 다음과 같습니다.

```text
Swim2/Blender extractor
→ build_final_classifier_dataset_v1.py
→ train_classifier_hailo_v1.py
→ export_classifier_onnx_hailo_v2.py
→ make_classifier_calib_npy.py
```

## 현재 성능과 상태

- Head detector 최종 기준 파일: `pool_head_best.pt`
- Tracking 최종 기준: ByteTrack + V12 stable-ID
- 시계열 입력: 10Hz × 5초 = 50 timestep, 14 feature
- 분류기: SWIMMING / FLOATING / ACTIVE 3-class Temporal CNN
- PASSIVE: 장시간 LOST rule override
- classifier dataset: train 2,928 / validation 299 / test 421 windows
- 로컬 held-out test confusion matrix: 418/421, 약 **99.3%**
- 카메라 보정 웹 도구: 자동 테스트 12개 통과 기록

99.3%는 현재 구성된 로컬 분할 결과이며 실제 수영장 일반화 성능을 의미하지 않습니다. 현장 영상, 다른 수영모·조명·카메라 시점에서 별도 검증이 필요합니다.

## 남은 작업

- Swim2 추가 라벨로 Head Detector 재파인튜닝 후 최종 모델 교체 여부 결정
- Temporal CNN ONNX를 Hailo HEF로 변환하고 Raspberry Pi 5 실시간 추론에 통합
- V12 output → 50×14 feature buffer → classifier → PASSIVE override 단일 런타임 연결
- 두 Camera Module 3의 현장 homography 오차 및 Global ID 검증
- Pi5 → Arduino Nano serial protocol과 Nano firmware 확정
- BLD-50, DMD-150, A4988 실제 장비별 방향·속도·limit/home/E-stop 시험
- 실제 익수 상황을 모사한 안전 시험과 false alarm/누락률 측정

자세한 완료/보류 상태는 [PROJECT_STATUS.md](docs/PROJECT_STATUS.md)를 참고합니다.
