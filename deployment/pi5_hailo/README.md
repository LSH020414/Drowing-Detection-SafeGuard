# Raspberry Pi 5 + Hailo-8 deployment

사용자가 제공한 `2026ESWContest_--main.zip`의 최종 실행 묶음을 Raspberry Pi에서 코드 기본 인자 그대로 실행할 수 있도록 정리한 폴더입니다.

## 파일

| 저장소 파일 | 압축 원본 이름 | 역할 |
|---|---|---|
| `drowning_full-pipeline_final.py` | 동일 | Picamera2 → Hailo detector → ByteTrack → 50×14 classifier → PASSIVE rule |
| `pool_head_best_512.hef` | `head_detector.hef` | 512 입력 Head Detector |
| `drowning_classifier_50x14_hailo.hef` | `drowning_classifier.hef` | 50 timestep × 14 feature classifier |
| `bytetrack_pool.yaml` | `pool_ID_Tracker.yaml` | 배포용 ByteTrack 설정 |

압축 안 실제 이름과 Python 기본 인자 이름이 달랐기 때문에, **코드는 수정하지 않고 세 리소스의 저장소 파일명만 코드가 기대하는 이름으로 바꿨습니다.**

## 동작 규격

| 항목 | 기본값 |
|---|---:|
| Camera frame | 1280×720, 30fps |
| Temporal sampling | 10Hz |
| Window | 50 timestep |
| Temporal history | 5초 |
| PASSIVE continuous LOST | 3초 |
| Track state TTL | 8초 |
| Detection confidence | 0.05 |
| NMS IoU | 0.70 |

분류기 출력은 `SWIMMING`, `FLOATING`, `ACTIVE`이고 연속 LOST가 기본 30 sample 이상이면 `PASSIVE_DROWNING`으로 override합니다.

## Raspberry Pi 준비

- Raspberry Pi 5 64-bit OS
- Camera Module 3와 Picamera2
- Hailo-8 및 HEF와 호환되는 HailoRT
- Python에서 import 가능한 `hailo_platform`
- OpenCV, NumPy, PyYAML
- `~/hailo-apps/hailo_apps/python/core/tracker/byte_tracker.py`

HailoRT와 Hailo Apps는 장치에 설치된 버전에 맞춰 준비합니다. HEF가 다른 DFC/HailoRT 버전으로 컴파일됐다면 로딩이 실패할 수 있습니다.

## 실행

```bash
cd deployment/pi5_hailo
python3 drowning_full-pipeline_final.py
```

GUI 없이 실행:

```bash
python3 drowning_full-pipeline_final.py --no-display
```

주요 인자를 바꾸는 예:

```bash
python3 drowning_full-pipeline_final.py \
  --camera-fps 30 \
  --sample-hz 10 \
  --passive-seconds 3 \
  --state-ttl 8 \
  --conf 0.05 \
  --iou 0.70
```

다른 위치의 모델을 사용할 때는 `--detector`, `--classifier`, `--tracker`로 지정합니다.

## 현재 범위

이 코드는 단일 Picamera2 입력에서 Head detection, ByteTrack, 50×14 feature 생성, Hailo classifier, PASSIVE rule, 화면·콘솔 표시까지 수행합니다.

다음 항목은 이 파일에 포함되어 있지 않습니다.

- 두 카메라 Global ID 통합
- homography 기반 수영장 X,Y 계산
- Raspberry Pi → Arduino Nano 명령
- BLD-50/DMD-150/A4988 조준·발사·재장전

따라서 이 배포 묶음은 **AI 상태 판별 최종 런타임**이며 전체 구조 장치 제어 런타임은 아닙니다.

## 확인할 이름 차이

코드의 3번 class 표시는 `ACTIVE`입니다. 프로젝트 문서의 정식 상태명은 `ACTIVE_DROWNING`이므로 외부 API나 하드웨어 연동 전에 이름을 통일해야 합니다.

## 원본 무결성

| 압축 원본 | Bytes | SHA-256 |
|---|---:|---|
| `drowning_full-pipeline_final.py` | 16,352 | `E469B03A121BB89F443A7041F1F60C43277719109FCAAECCFBFF109AC3FB4D79` |
| `head_detector.hef` | 17,908,563 | `41563FE40DAE86F24BB07253A2844F724ECE0482DFA6F9B234611CA9E9983FF8` |
| `drowning_classifier.hef` | 1,221,442 | `C027C241ED97B6F125124612917C3BCD6EBEB014D7F8DD92CB6B8B4832AB38BD` |
| `pool_ID_Tracker.yaml` | 147 | `45C3D0F41557B1C6BA908ACF9EEEC3DB1D6A5D1645D45AEB0C0180A3B311C002` |

HEF 파일은 Git LFS로 관리합니다.
