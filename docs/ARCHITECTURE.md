# System architecture

## 역할 분리

| 단계 | 입력 | 출력 | 책임 |
|---|---|---|---|
| Head detection | camera frame | head box, confidence | 머리 위치 검출 |
| ByteTrack/V12 | detections | stable person ID | 짧은 LOST와 재검출 연결 |
| Feature buffer | stable track | 50×14 tensor | 10Hz, 최근 5초 정규화 |
| Temporal CNN | 50×14 tensor | 3-class score | SWIMMING/FLOATING/ACTIVE 분류 |
| PASSIVE rule | LOST history | override state | 장시간 머리 소실 안전 규칙 |
| Homography | image point | pool X,Y | 실제 좌표 변환 |
| Launcher control | target X,Y | two serial command streams | Nano/UNO 수평 회전과 ESP32 BLDC 속도 제어 순서 |

## 50×14 classifier input

```text
detected
confidence
cx_norm
cy_norm
head_scale
dx
dy
speed
scale_change
lost_event
redetect_event
consecutive_missing
visible_ratio
missing_ratio
```

모든 feature는 시간 순서대로 50개가 있어야 하며, 부족한 window는 학습 코드에서 제외됩니다.

## 상태 결정

```text
Temporal CNN
  ├─ SWIMMING
  ├─ FLOATING
  └─ ACTIVE

연속 LOST가 안전 임계값을 넘음
  └─ PASSIVE_DROWNING (override)
```

PASSIVE를 3-class classifier에 억지로 포함하지 않아 데이터 부족 상황에서도 명시적인 안전 조건을 적용할 수 있습니다. 임계값은 실제 현장 검증을 통해 확정해야 합니다.

## 카메라와 좌표

두 카메라는 각각 4개 수영장 모서리를 기준으로 homography를 계산합니다. 픽셀 좌표를 공통 수영장 미터 좌표로 변환한 뒤 local track을 global ID로 통합하는 것이 목표입니다. 현재 보정 웹 도구는 구현되어 있으나, 실시간 두 카메라 Global ID와 AI 파이프라인의 완전 통합은 남은 작업입니다.

## MCU 제어 경계

Raspberry Pi는 판단·좌표 계산·안전 조건과 발사 순서를 담당하고, USB serial로 두 MCU에 newline 기반 텍스트 명령을 보냅니다. Nano/UNO는 -47°~+87° 소프트웨어 범위 안에서 수평축을 10° 단위로 움직이고, ESP32는 반대 방향으로 회전하는 BLDC 2개의 PWM을 5% 단위·최대 80%로 제어합니다. 각 MCU는 `ACK`, `ERR`, `STATUS` 응답을 돌려줍니다.

현재 펌웨어는 수평축 encoder와 BLDC RPM feedback을 사용하지 않는 open-loop 제어입니다. 수직축·장전 액추에이터·limit/home/E-stop 및 Raspberry Pi의 두 포트 통합 orchestration은 별도 구현·실기 검증 범위입니다.
