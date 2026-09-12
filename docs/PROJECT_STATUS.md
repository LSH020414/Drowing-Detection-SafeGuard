# Project status

기준일: 2026-09-12

시간순 개발 과정과 설계 변경 이유는 [DEVELOPMENT_TIMELINE.md](DEVELOPMENT_TIMELINE.md)를 참고합니다.

## 확정·보존된 항목

- `pool_head_best.pt` 기준 모델 파일과 SHA-256 확인
- `bytetrack_pool.yaml` 최종 설정 보존
- `drowning_main_v12.py` V12 추적/규칙 파이프라인 보존
- 10Hz × 5초 = 50 timestep 규격 확정
- 14 feature Temporal CNN 학습/ONNX 내보내기 코드 보존
- PASSIVE_DROWNING을 장시간 LOST rule로 override하는 구조 확정
- Swim2 정상 수영 및 Blender ACTIVE/FLOATING 추출 코드 보존
- Swim2 cascade 검증 기준 코드 2종 보존
- 시간별 상태를 표시하는 발표용 demo 코드 보존
- 두 카메라 보정 웹 도구와 인계 문서 보존

## 확인된 결과

| 항목 | 결과 |
|---|---|
| classifier train windows | 2,928 |
| classifier validation windows | 299 |
| classifier test windows | 421 |
| held-out test confusion matrix | 418/421 correct (약 99.3%) |
| calibrator test 기록 | 12 tests passed |

## 아직 완료로 표시하지 않는 항목

- 새 Swim2 파인튜닝 모델을 `pool_head_best.pt` 대체본으로 확정
- Detector/V12/Temporal CNN/PASSIVE rule의 단일 실시간 실행
- Hailo HEF 생성 및 Raspberry Pi 5 실기 성능 측정
- 두 카메라 Global ID 실시간 통합
- Nano 최종 firmware와 실제 BLD-50/DMD-150/A4988 wiring 확정
- 현장 false positive/false negative, 지연시간, 조준 오차 측정

## 주의

- demo의 상태 라벨과 고정 ID는 발표용이며 AI 성능 결과가 아닙니다.
- 로컬 test accuracy는 동일 출처·증강 데이터가 포함된 분할의 영향을 받을 수 있습니다.
- `ai/configs/v12_rule_config.json`과 V12 코드 상수에 일부 차이가 있으므로 실험 재현 시 사용 설정을 명시해야 합니다.
- `pool-calibrator`의 진행 중 ESP32 firmware는 Nano 목표 하드웨어와 별개입니다.
