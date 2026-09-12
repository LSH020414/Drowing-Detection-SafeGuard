# AI

최종 기준 AI 코드를 목적별로 나눈 폴더입니다.

- `pipeline/`: `pool_head_best.pt`, ByteTrack, V12 stable-ID와 rule engine
- `configs/`: 확정 tracker 설정과 V12 규칙 파라미터 기록
- `training/`: 50 timestep × 14 feature Temporal CNN 학습
- `export/`: ONNX/Hailo용 모델 및 calibration 입력 생성
- `models/`: 모델 파일 배치·무결성 안내

V12 파이프라인은 CLI 인자를 지원하지만, 학습·내보내기 코드는 확정 당시 로컬 절대 경로를 보존합니다. 재실행 전 각 파일 상단의 경로를 현재 환경에 맞게 바꿔야 합니다.

`v12_rule_config.json`은 실험 V12의 기준값 기록입니다. 현재 `drowning_main_v12.py` 상수 일부와 값이 다르므로 자동 로딩되는 실행 설정으로 오해하면 안 됩니다. 재현 실험에서는 어느 값을 사용했는지 결과와 함께 기록해야 합니다.
