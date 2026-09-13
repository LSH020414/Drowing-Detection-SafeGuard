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
        ├─ Arduino Nano/UNO → 수평 회전 모터(IN1/PWM/IN2)
        └─ ESP32            → BLDC 발사 휠 2개(PWM/DIR)
```

상세 설계는 [시스템 아키텍처](docs/ARCHITECTURE.md), 하드웨어 신호 구조는 [하드웨어 문서](hardware/README.md)를 참고합니다.

## 개발 발표서

2026년 9월에 작성한 20쪽 분량의 개발 발표서 원본을 [`docs/presentation/AI_학습_기반_수영장_익수_감지_및_구조지원_시스템.pdf`](docs/presentation/AI_학습_기반_수영장_익수_감지_및_구조지원_시스템.pdf)에 보존했습니다.

| 페이지 | 주요 내용 |
|---|---|
| 1~4 | AquaGuard 개요, 개발 배경, 수면 위 카메라와 머리 중심 판단 근거 |
| 5~7 | Detector → Tracker → Classifier 3단계 소프트웨어와 5초 행동 시계열 |
| 8~10 | 실제 영상·Swim2, Omniverse/Blender, SwimXYZ를 이용한 데이터 구성 |
| 11~16 | 피칭 머신식 발사대, 장전 장치, 구조 캡슐, 회로와 AI-하드웨어 연결 |
| 17 | Hailo-8 변환·성능 저하 원인 분석과 detector/classifier 최적화 방향 |
| 18~19 | 감지부터 구조 지원까지의 차별성, 활용처와 기대 효과 |
| 20 | 6~9월 개발 일정과 팀 업무 분장 |

발표서는 당시 설계와 설명을 보존한 자료입니다. 슬라이드의 ESP32/Arduino 병행 구성처럼 이후 변경된 내용이 있으므로, **현재 구현 기준은 이 README와 [PROJECT_STATUS.md](docs/PROJECT_STATUS.md), [hardware/README.md](hardware/README.md)를 우선합니다.**

## 개발 과정

프로젝트는 전체 구조 설계에서 시작해 Head Detector, CVAT 라벨링, Tracker 비교, 50 timestep 시계열 모델, Hailo 이식, 발사대 제어, 시연 영상 순으로 발전했습니다.

| 시기 | 핵심 작업 | 결정·변경 |
|---|---|---|
| 7월 말~8월 초 | 전체 시스템과 수영장 좌표계 설계 | Camera → Pi5 → AI → Homography → 발사 구조 확정 |
| 8월 초~중순 | Head Detector와 CVAT 환경 구축 | 전신 대신 머리 중심 검출, 잠김 구간은 같은 ID의 `outside=True` |
| 8월 중순~말 | Tracker·ReID 비교 | ByteTrack + V12 Stable ID를 기준으로 선정 |
| 8월 말 | 발사대와 모터 드라이버 구체화 | 수평축을 NEMA17에서 DC 웜기어모터 방향으로 변경 |
| 9월 초 | Swim2·SwimXYZ·Blender 데이터 구축 | 10Hz × 5초 = 50 timestep, 4개 상태 체계 확정 |
| 9월 초 | Temporal CNN·PASSIVE rule 설계 | 3-class CNN + 장시간 LOST override로 역할 분리 |
| 9월 초 | Raspberry Pi/Hailo 및 MCU 시험 | 640→512 변환 이슈 확인, ESP32/Uno 시험 후 Nano 목표 구조로 정리 |
| 최근 | 발표용 추적 영상·MCU 소스 확정 | 10Hz·5px 박스 표시와 Pi 연계용 Nano/UNO·ESP32 텍스트 프로토콜 적용 |

32단계의 상세 작업 이력과 당시 결과는 [개발 타임라인](docs/DEVELOPMENT_TIMELINE.md)에 정리했습니다.

## 데이터와 개발 도구

상태 분류 데이터는 SwimXYZ, 프로젝트 내부 Swim2 4영법 데이터, 실제 부유·익수 영상, Blender/Omniverse 합성 sequence를 정리해 사용했습니다. Head Detector 초기 학습에는 CrowdHuman의 head annotation을 사용했습니다.

- 데이터 출처·수량·split·라벨링·배포 주의사항: [data/README.md](data/README.md)
- YOLO11·ByteTrack·PyTorch·CVAT·Docker·Hailo 변환 구성: [ai/README.md](ai/README.md)
- 모델 바이너리·SHA-256·Git LFS: [ai/models/README.md](ai/models/README.md)

외부 원본 영상과 대용량 학습 데이터는 저장소에 포함하지 않으며, 출처별 라이선스와 재배포 조건을 별도로 확인합니다.

## 소프트웨어 파이프라인

1. `pool_head_best.pt`가 프레임별 머리 박스와 confidence를 검출합니다.
2. `bytetrack_pool.yaml`과 V12 ID 복구 로직이 동일 인물을 이어 붙입니다.
3. Track별 특징을 10Hz로 샘플링해 5초 길이의 50 timestep window를 만듭니다.
4. 14개 특징을 받는 Temporal CNN이 `SWIMMING`, `FLOATING`, `ACTIVE`를 분류합니다.
5. 연속 LOST 조건이 충족되면 분류기 결과보다 우선해 `PASSIVE_DROWNING`으로 전환합니다.
6. 보정 웹 도구의 homography로 픽셀을 미터 좌표로 바꾸고 제어 명령을 MCU에 전달합니다.

PC용 V12 추적/규칙 엔진과 Temporal CNN 학습·내보내기 코드에 더해, Hailo detector·classifier·ByteTrack·PASSIVE LOST rule을 한 루프로 실행하는 Raspberry Pi 5용 최종 배포 코드와 HEF 모델도 확보했습니다. 다만 **두 카메라 Global ID, homography 좌표, Nano 발사 제어까지 하나의 현장 런타임으로 연결하는 작업은 아직 남아 있습니다.**

## 저장소 구조

```text
.
├─ ai/
│  ├─ configs/        # ByteTrack 및 V12 rule 설정
│  ├─ pipeline/       # 최종 V12 머리 추적/규칙 파이프라인
│  ├─ training/       # 50×14 Temporal CNN 학습
│  ├─ export/         # ONNX/Hailo 변환 및 calibration 입력 생성
│  ├─ reid/           # Pool ReID 추출·학습·평가
│  └─ models/         # PC용 PT/ONNX 모델(Git LFS)
├─ data/
│  ├─ extraction/     # Swim2·Blender 시계열 추출
│  ├─ processing/     # 최종 classifier dataset 구성
│  ├─ cvat/           # Swim2 CVAT 작업 준비
│  ├─ preparation/    # CrowdHuman→YOLO 변환
│  ├─ schema/         # 특징 스키마
│  └─ validation/     # Swim2 cascade 검증 코드
├─ demo/              # 발표/시연용 tracking video 생성
├─ experiments/
│  └─ tracking/       # V10/V11/V12 GT 평가와 threshold sweep
├─ deployment/
│  └─ pi5_hailo/      # Pi5 + Hailo-8 최종 실시간 실행 묶음
├─ hardware/
│  └─ firmware/       # Nano/UNO 수평축 및 ESP32 BLDC 최종 MCU 소스
├─ pool-calibrator/   # 2-camera homography 보정 웹 도구
├─ docs/              # 아키텍처, 상태, 원본 이력
│  └─ presentation/   # 개발 발표서 PDF 원본
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
| [`deployment/pi5_hailo/drowning_full-pipeline_final.py`](deployment/pi5_hailo/drowning_full-pipeline_final.py) | Pi Camera + Hailo detector/classifier + ByteTrack + PASSIVE rule 최종 실행 |
| [`hardware/firmware/arduino_nano_uno/arduino_launcher_motor.ino`](hardware/firmware/arduino_nano_uno/arduino_launcher_motor.ino) | Nano/UNO 수평 회전 제어, 각도 상태·제한, ACK/ERR/STATUS 시리얼 프로토콜 |
| [`hardware/firmware/esp32/esp32_bldc_controller.ino`](hardware/firmware/esp32/esp32_bldc_controller.ino) | ESP32 BLDC 2축 속도 제어, 5% step, 80% 제한, kick-start, 시리얼 응답 |
| [`ai/reid/`](ai/reid/) | Pool ReID용 CVAT crop 추출·학습·유사도 평가 |
| [`experiments/tracking/`](experiments/tracking/) | Task 90 tracking GT 평가와 geometry/ReID sweep |
| [`pool-calibrator/`](pool-calibrator/) | 두 카메라 수영장 좌표 보정 웹 도구 |
| [`docs/presentation/AI_학습_기반_수영장_익수_감지_및_구조지원_시스템.pdf`](docs/presentation/AI_학습_기반_수영장_익수_감지_및_구조지원_시스템.pdf) | 20쪽 개발 발표서 원본 |

원본 파일 경로와 SHA-256은 [SOURCE_MANIFEST.md](docs/SOURCE_MANIFEST.md)에 기록했습니다.

## 실행 방법

### 1. 환경

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-ai.txt
```

### 2. 모델 배치

핵심 PT/ONNX/HEF는 Git LFS로 포함돼 있습니다. clone 후 실제 바이너리를 받고 해시를 확인합니다.

```powershell
git lfs install
git lfs pull
```

모델 목록과 검증 해시는 [ai/models/README.md](ai/models/README.md)에 있습니다.

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

### 6. Raspberry Pi 5 + Hailo-8 최종 실행

`deployment/pi5_hailo/`를 Raspberry Pi로 복사한 뒤 해당 폴더에서 실행합니다.

```bash
cd deployment/pi5_hailo
python3 drowning_full-pipeline_final.py
```

화면 없이 상태 로그만 실행하려면 `--no-display`를 추가합니다. HailoRT, Picamera2, `hailo_platform`, Hailo Apps ByteTrack 경로 등 장비별 준비 사항은 [배포 README](deployment/pi5_hailo/README.md)에 정리했습니다.

### 7. MCU 펌웨어

Arduino IDE에서 각 `.ino` 파일을 같은 이름의 sketch 폴더째 열어 업로드합니다. Nano/UNO는 9600 baud, ESP32는 115200 baud이며, 두 장치는 newline으로 끝나는 `0`, `1`, `2`, `STATUS` 명령과 `ACK`/`ERR`/`STATUS` 응답을 사용합니다. 정확한 핀과 명령 의미는 [하드웨어 문서](hardware/README.md)에 있습니다.

## 현재 성능과 상태

- Head detector 최종 기준 파일: `pool_head_best.pt`
- Tracking 최종 기준: ByteTrack + V12 stable-ID
- 시계열 입력: 10Hz × 5초 = 50 timestep, 14 feature
- 분류기: SWIMMING / FLOATING / ACTIVE 3-class Temporal CNN
- PASSIVE: 장시간 LOST rule override
- Pi5/Hailo 배포 묶음: detector HEF + classifier HEF + ByteTrack + 통합 실행 코드 확보
- Pi 연계 MCU 소스: Nano/UNO 수평 회전 제어 + ESP32 BLDC 2개 속도 제어 확정
- classifier dataset: train 2,928 / validation 299 / test 421 windows
- 로컬 held-out test confusion matrix: 418/421, 약 **99.3%**
- 카메라 보정 웹 도구: 자동 테스트 12개 통과 기록
- Task 90 tracking 실험: V12 GeoMax 0.45가 Raw ByteTrack 대비 ID switch 63→53, IDF1 0.8821→0.8442

99.3%는 현재 구성된 로컬 분할 결과이며 실제 수영장 일반화 성능을 의미하지 않습니다. 현장 영상, 다른 수영모·조명·카메라 시점에서 별도 검증이 필요합니다.

Tracking 수치는 V12가 ID 전환을 줄인 대신 해당 단일 시험의 IDF1은 낮았다는 뜻입니다. 파라미터 선택은 여러 현장 영상에서 다시 검증해야 합니다.

## 남은 작업

- Swim2 추가 라벨로 Head Detector 재파인튜닝 후 최종 모델 교체 여부 결정
- Pi5/Hailo 최종 묶음의 실제 Camera Module 3 장시간 안정성·FPS·온도 측정
- 배포 코드의 `ACTIVE` 출력명을 프로젝트 표준 `ACTIVE_DROWNING`과 통일
- 두 Camera Module 3의 현장 homography 오차 및 Global ID 검증
- Pi5 실시간 AI 결과 → homography → 발사 제어 런타임 연결
- Pi5에서 Nano/UNO와 ESP32 두 serial 장치를 함께 운용하는 발사 시퀀스 통합
- 수평축 10° 이동시간 보정과 실제 각도 센서/limit/home/E-stop 검증
- BLDC 실제 회전 방향·RPM·80% 출력과 kick-start 시험
- 수직 A4988, 장전·발사 액추에이터 및 하드웨어 interlock 구현
- 실제 익수 상황을 모사한 안전 시험과 false alarm/누락률 측정

자세한 완료/보류 상태는 [PROJECT_STATUS.md](docs/PROJECT_STATUS.md), 작업 순서는 [DEVELOPMENT_TIMELINE.md](docs/DEVELOPMENT_TIMELINE.md)를 참고합니다.
