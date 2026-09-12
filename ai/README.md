# AI

최종 기준 AI 코드를 목적별로 나눈 폴더입니다.

- `pipeline/`: `pool_head_best.pt`, ByteTrack, V12 stable-ID와 rule engine
- `configs/`: 확정 tracker 설정과 V12 규칙 파라미터 기록
- `training/`: 50 timestep × 14 feature Temporal CNN 학습
- `export/`: ONNX/Hailo용 모델 및 calibration 입력 생성
- `models/`: 모델 파일 배치·무결성 안내

V12 파이프라인은 CLI 인자를 지원하지만, 학습·내보내기 코드는 확정 당시 로컬 절대 경로를 보존합니다. 재실행 전 각 파일 상단의 경로를 현재 환경에 맞게 바꿔야 합니다.

`v12_rule_config.json`은 실험 V12의 기준값 기록입니다. 현재 `drowning_main_v12.py` 상수 일부와 값이 다르므로 자동 로딩되는 실행 설정으로 오해하면 안 됩니다. 재현 실험에서는 어느 값을 사용했는지 결과와 함께 기록해야 합니다.

## 모델과 도구

| 도구 | 프로젝트 용도 |
|---|---|
| [Ultralytics YOLO](https://docs.ultralytics.com/) | YOLO11 Head Detector 학습, validation, inference, PT→ONNX export |
| [ByteTrack](https://docs.ultralytics.com/modes/track/) | Head detection을 Track ID와 시간축으로 연결 |
| [PyTorch](https://pytorch.org/docs/stable/index.html) | 50×14 Conv2D temporal classifier 학습과 checkpoint 저장 |
| [PyTorch ONNX exporter](https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html) | classifier PT→ONNX 변환 |
| [OpenCV](https://docs.opencv.org/) | 영상 I/O, frame 추출, resize/letterbox, 시각화 |
| [NumPy](https://numpy.org/doc/) / [Pandas](https://pandas.pydata.org/docs/) | feature 계산, 50-step sequence, CSV/split/augmentation |
| [CVAT](https://docs.cvat.ai/) | Head box, Track ID, outside 구간 검수와 dataset export |
| [Docker Compose](https://docs.docker.com/compose/) | CVAT과 Hailo 변환 컨테이너 실행·volume mount |

## 최종 모델 역할

```text
pool_head_best_512.hef
→ head bounding box + confidence
→ ByteTrack
→ Track별 50 timestep
→ 14 temporal features
→ drowning_classifier_50x14_hailo.hef
→ SWIMMING / FLOATING / ACTIVE
→ long LOST rule
→ PASSIVE_DROWNING override
```

Classifier는 입력을 image-like `[1, 50, 14]` 형태로 구성한 Conv2D 기반 temporal classifier입니다. 프로젝트 문서에서는 역할을 간단히 Temporal CNN/TCN으로 부르지만, 보존된 구현 클래스명은 `HailoTemporalCNN`이며 2D convolution을 시간축 방향으로 적용합니다.

## Hailo 변환 환경

프로젝트 기록에 남은 변환 환경:

```text
Hailo Model Zoo   2.19.0
HailoRT           4.24.0
Dataflow Compiler 3.34.0
```

변환 과정:

```text
PyTorch checkpoint
→ ONNX
→ HAR parse
→ representative calibration data
→ INT8 optimization
→ Hailo compile
→ HEF
```

최종 HEF는 `deployment/pi5_hailo/pool_head_best_512.hef`와 `deployment/pi5_hailo/drowning_classifier_50x14_hailo.hef`입니다. HEF는 컴파일에 사용한 DFC와 Raspberry Pi의 HailoRT 호환성을 확인해야 합니다.
