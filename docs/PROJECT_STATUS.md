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
- Pi5/Hailo 실시간 통합 코드, detector HEF, 50×14 classifier HEF, tracker 설정 보존
- 데이터 출처, 클래스별 split 수량, annotation 규칙, AI/Hailo 도구 버전 문서화
- 개발 배경부터 AI·발사 구조·Hailo 최적화·기대 효과까지 담은 20쪽 개발 발표서 원본 보존
- 누락됐던 4영법·ACTIVE·FLOATING·SwimXYZ 전처리/증강 코드와 Detector Hailo 변환 코드 보존
- Pool ReID 데이터 추출·학습·평가 코드와 Task 90 tracking 평가/sweep 결과 보존
- PC용 detector, classifier, ReID와 ONNX 모델을 Git LFS로 보존

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
- 제공된 Hailo HEF와 통합 코드의 Raspberry Pi 5 실기 성능·장시간 안정성 측정
- 두 카메라 Global ID 실시간 통합
- Pi5 AI 상태 출력과 homography/발사 제어 연결
- Nano 최종 firmware와 실제 BLD-50/DMD-150/A4988 wiring 확정
- 현장 false positive/false negative, 지연시간, 조준 오차 측정
- 프로젝트 내부 명칭 `Swim2`의 정확한 공식 출처·라이선스 기록 보완
- YouTube 원천 영상별 사용·학습·발표·재배포 권한 확인 및 source manifest 작성

## 주의

- demo의 상태 라벨과 고정 ID는 발표용이며 AI 성능 결과가 아닙니다.
- 로컬 test accuracy는 동일 출처·증강 데이터가 포함된 분할의 영향을 받을 수 있습니다.
- `ai/configs/v12_rule_config.json`과 V12 코드 상수에 일부 차이가 있으므로 실험 재현 시 사용 설정을 명시해야 합니다.
- `pool-calibrator`의 진행 중 ESP32 firmware는 Nano 목표 하드웨어와 별개입니다.
- 최종 Pi5 코드의 classifier 라벨은 `ACTIVE`이며 프로젝트 표준 문서의 `ACTIVE_DROWNING`과 이름을 통일해야 합니다.
- CrowdHuman 원본 이미지는 공식 이용 조건상 이 저장소에 재배포하지 않습니다.
- 개발 발표서의 ESP32/Arduino 병행 회로는 발표 당시 설계입니다. 현재 목표 제어 구조는 `hardware/README.md`의 Pi5 → Arduino Nano 구성을 우선합니다.
- 발표서의 배경 통계와 비교표는 발표자료에 수록된 설명이며, 이 저장소에서 별도의 원문 출처 검증을 완료한 수치는 아닙니다.
- Task 90 한 건에서는 V12가 Raw ByteTrack보다 ID switch는 적었지만 IDF1은 낮았습니다. Stable-ID 개선을 전체 tracking 성능 개선으로 단정하지 않습니다.
