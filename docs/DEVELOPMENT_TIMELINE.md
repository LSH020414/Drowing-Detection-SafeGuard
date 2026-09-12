# Development timeline

이 문서는 대화와 프로젝트 기록을 바탕으로 개발 흐름을 시간순으로 정리한 것입니다. 날짜가 정확히 기록되지 않은 작업은 월 초·중순·말 단위로 표시했습니다. 수치와 상태는 당시 확인 결과이며, 현재 최종 상태는 [PROJECT_STATUS.md](PROJECT_STATUS.md)를 우선합니다.

## 한눈에 보기

```text
시스템 방향 설정
→ Homography 좌표계
→ Head Detector
→ CVAT 라벨링
→ ByteTrack / Stable ID
→ 10Hz × 5초 시계열
→ Temporal CNN + PASSIVE LOST rule
→ Hailo-8 이식
→ Pi5 / MCU / 모터 제어
→ 발표용 시연 영상
```

## 7월 말~8월 초: 프로젝트 방향과 좌표계

### 1. 전체 시스템 방향 확정

수영장 카메라에서 익수 위험자를 찾아 실제 좌표를 계산하고, 구조용 튜브 발사대를 자동 조준하는 것을 목표로 정했습니다.

```text
Camera
→ Raspberry Pi 5
→ AI detection / tracking
→ 익수 상태 판별
→ 실제 위치 계산
→ 모터 제어
→ 구조용 발사체 발사
```

카메라는 Raspberry Pi Camera Module 3 Standard 2대, 주 연산 장치는 Raspberry Pi 5 8GB와 Hailo-8 조합으로 정리했습니다.

### 2. 수영장 실제 좌표 변환 방식 결정

각 카메라 영상에서 수영장 네 모서리를 지정하고 실제 길이·폭을 입력한 뒤 homography로 픽셀을 미터 좌표로 변환하는 구조를 선택했습니다. 설계 검토에서는 주로 25m × 10m 수영장을 기준으로 사용했습니다.

```text
image pixel (u, v)
→ 4-point homography
→ pool coordinate (x, y)
→ launcher yaw / pitch calculation
```

## 8월 초~중순: Head Detector와 라벨링

### 3. 머리 중심 Detector 구축

수영 환경에서는 신체 대부분이 물에 가려지므로 person 전체보다 머리 검출이 적합하다고 판단했습니다. CrowdHuman 기반 Head Detector 학습의 당시 대표 결과는 다음과 같습니다.

| 지표 | 값 |
|---|---:|
| Precision | 0.9010 |
| Recall | 0.8672 |
| mAP50 | 0.8162 |
| mAP50-95 | 0.5224 |

CroHD/HT21도 검토했지만 CroHD를 detector에 직접 섞었을 때 수영장 영상 오탐이 증가해, detector보다는 tracking/ReID 보조 데이터로 활용하는 방향을 검토했습니다.

### 4. CVAT 공동 라벨링 환경 구축

로컬 CVAT, 동일 Wi-Fi 접속, Tailscale 외부 접속을 구성해 여러 명이 수영장 Head annotation을 수행할 수 있게 했습니다. 머리가 물속에 들어가도 새 객체로 만들지 않고 기존 Track ID를 유지하며, 보이지 않는 구간은 `outside=True`로 기록하는 규칙을 정했습니다.

```text
head visible → 같은 track ID
head submerged → outside=True
head reappears → 기존 track ID 계속 사용
```

이 규칙은 이후 LOST와 재검출 패턴을 시계열 특징으로 사용하는 기반이 됐습니다.

## 8월 중순~말: Tracking과 발사대 설계

### 5. Tracker 비교

ByteTrack, BoT-SORT, FairMOT, ReID 기반 방식을 검토하고 detector-only, detector+tracker, ReID 적용 결과를 비교했습니다. 최종 기준은 ByteTrack에 짧은 LOST와 재검출을 보정하는 Stable ID 로직을 결합하는 방향으로 정했습니다.

### 6. 발사대 초기 구조 설계

초기 발사대는 NEMA17 수평·수직축, BLDC 발사 휠 2개, 장전 액추에이터로 구성했습니다. 검토한 발사체는 직경 약 60mm, 질량 약 700g 수준이며 발사 휠은 약 Ø130mm급이었습니다. 이 값들은 실기 시험 전 설계 검토값입니다.

### 7. 모터와 드라이버 구체화

최종 목표 구동 구성은 다음과 같이 정리했습니다.

```text
BLD-50 × 2 → upper/lower BLDC launch wheels
DMD-150     → horizontal DC worm gear motor
A4988       → vertical NEMA17 stepper
```

### 8. 수평축 토크 문제와 설계 변경

발사대 전체 질량이 증가하면서 NEMA17만으로 수평축을 돌리기에는 토크 여유가 부족할 수 있다고 판단했습니다. 수평축을 24V WGM58 계열 DC 웜기어모터로 변경하고 DMD-150을 사용하는 방향을 검토했습니다.

### 9. 8월 31일 발표자료 하드웨어 흐름 정리

발표자료 후반부를 전체 하드웨어, 발사·장전 구조, 수영장 설치와 homography, 전체 시스템 동작 순으로 구성했습니다.

```text
시스템 시작
→ 카메라 입력
→ 익수자 탐지
→ 위치 계산
→ 자동 조준
→ 구조요원 확인
→ 발사
→ 재장전
```

## 9월 초: 시계열 익수 판별과 데이터 구축

### 10. 입력 규격 통일

모든 데이터 출처에 동일한 시계열 규격을 적용했습니다.

```text
10Hz × 5초 = 50 timestep
```

한 사람의 최근 5초 Track을 하나의 classifier window로 사용합니다.

### 11. 네 가지 상태 이름 확정

```text
SWIMMING
FLOATING
ACTIVE_DROWNING
PASSIVE_DROWNING
```

초기 코드의 `NORMAL` 명칭은 최종 시연·분류 체계에서 `SWIMMING`으로 변경했습니다.

### 12. PASSIVE_DROWNING 처리 방식 분리

PASSIVE 학습 데이터가 부족하므로 Temporal CNN은 SWIMMING/FLOATING/ACTIVE 세 상태를 분류하고, PASSIVE는 장시간 연속 LOST 규칙이 override하도록 역할을 분리했습니다.

```text
3-class Temporal CNN result
        +
continuous LOST history
        ↓
PASSIVE_DROWNING override
```

### 13. FLOATING 데이터 구축

실제 영상 기반으로 1,042개 window, 52,100개 sequence row를 만들었습니다. `1,042 × 50 = 52,100`으로 window 길이를 검증했으며, 이후 합성 데이터는 부족한 ACTIVE 보강에 더 집중하기로 했습니다.

### 14. SWIMMING 데이터 구축

SwimXYZ에서 당시 약 567개의 usable window를 확보했습니다. 실제 수영 환경의 영법·시점 다양성을 늘리기 위해 Swim2 Above 영상 96개도 추가 대상으로 선정했습니다.

### 15. Swim2 10fps 변환과 CVAT 준비

Swim2 Above 96개 영상에서 약 37,033장의 10fps JPG를 생성했습니다.

```text
SWIM2_CVAT_SINGLE_10FPS/
├─ images/
├─ manifest.csv
└─ annotations.xml
```

이 자료는 실제 수영환경 Head Detector 보정용 annotation 입력으로 준비했습니다.

### 16. Swim2 domain mismatch 확인

기존 `pool_head_best.pt`는 Swim2에서 검은 수영모, 수면 반사, 옆모습, 부분 잠김에 약했고 일부 배경 물체를 머리로 오탐했습니다. 문제의 중심이 tracker보다 detector의 domain mismatch에 있다는 결론을 내렸습니다.

### 17. Person→Head cascade 시험

```text
YOLO person detector
→ person ROI
→ pool_head_best.pt
→ ByteTrack
→ head geometry filter
```

Breaststroke 일부에서는 개선됐지만 Backstroke, Freestyle, Butterfly에서는 Head box가 안정적이지 않았습니다. 최종적으로 Swim2 실제 Head annotation을 이용한 detector 추가 fine-tuning이 필요하다고 판단했습니다.

### 18. 실제 ACTIVE_DROWNING 영상 구축

직접 확보한 활동적 익수 영상을 5초 window로 변환했습니다. 실제 ACTIVE 영상 수가 충분하지 않아 합성 데이터로 보완하는 방향을 함께 추진했습니다.

### 19. Blender/Omniverse 합성 데이터

ACTIVE와 FLOATING motion을 image sequence로 생성했습니다. 초기 video extractor가 입력을 찾지 못한 뒤 데이터가 동영상이 아니라 이미지 시퀀스임을 확인해 전용 extractor로 변경했습니다.

| 합성 상태 | Window | Sequence row |
|---|---:|---:|
| ACTIVE | 10 | 500 |
| FLOATING | 4 | 200 |

합성 데이터는 주 데이터가 아니라 augmentation 보조 데이터로 사용하기로 했습니다.

### 20. V12 최종 AI 파이프라인 정리

```text
Head Detector
→ ByteTrack
→ Stable ID
→ 10Hz sampling
→ 5 seconds / 50 timestep
→ temporal features
→ Temporal CNN
→ SWIMMING / FLOATING / ACTIVE
→ long LOST rule
→ PASSIVE_DROWNING override
```

## 9월 초: Raspberry Pi/Hailo 이식과 제어 시험

### 21. Hailo-8 모델 변환

640×640 detector를 Hailo용으로 컴파일할 때 `Agent infeasible` 문제가 발생해 입력을 512×512로 줄여 변환을 시도했습니다. 변환은 진행됐지만 detection 성능이 크게 하락해 최종 해결 과제로 남겼습니다.

이후 2026년 9월 12일에 512 입력 detector HEF, 50×14 classifier HEF, Hailo 실시간 통합 코드와 tracker 설정으로 구성된 최종 배포 묶음을 저장소에 반영했습니다. 이는 변환 산출물이 확보됐다는 뜻이며, 앞서 확인된 성능 저하가 현장 기준으로 해결됐다는 뜻은 아닙니다.

### 22. Pi→Arduino/ESP32 통신 시험

초기에는 Uno와 ESP32를 함께 검토했고 Raspberry Pi에서 USB serial 장치 `/dev/ttyUSB0`, `/dev/ttyUSB1` 인식까지 확인했습니다. 이 단계는 최종 MCU 확정 전 통신 실험입니다.

### 23. Launcher Flask server 구축

Raspberry Pi 발사대 제어용 Flask server를 구성했습니다. 기존 서비스와의 포트 충돌 때문에 5000에서 5001로 변경했고, 로컬 및 LAN 주소에서 접속을 시험했습니다.

```text
127.0.0.1:5001
192.168.137.2:5001
```

### 24. Arduino Nano 통합 방향 확정

현재 모터 드라이버와 5V 신호 호환 및 단순성을 고려해 최종 목표 MCU를 Arduino Nano 5V/16MHz로 정리했습니다.

```text
Raspberry Pi 5
  └─ USB Serial
      └─ Arduino Nano
          ├─ BLD-50 #1
          ├─ BLD-50 #2
          ├─ DMD-150
          └─ A4988
```

핀 배치와 현재 ESP32 프로토타입과의 차이는 [hardware/README.md](../hardware/README.md)에 기록했습니다.

## 최근: 발표용 시연 영상

### 25. Detector+Tracker 시연 코드

`make_demo_tracking_video.py`에 실제 `pool_head_best.pt`와 `bytetrack_pool.yaml`을 연결해 머리 박스와 상태를 표시하도록 했습니다.

### 26. 표시 ID 고정

발표 영상에서는 ByteTrack 내부 ID가 바뀌어도 화면에는 `ID 1`로 표시되도록 했습니다. 이 ID는 성능 평가용 tracking ID가 아니라 시연용입니다.

### 27. 시간별 상태 설정

영상 구간별로 상태를 직접 지정할 수 있는 `STATE_TIMELINE`을 추가했습니다.

```python
STATE_TIMELINE = [
    (0.0, 5.0, "SWIMMING"),
    (5.0, 10.0, "FLOATING"),
    (10.0, 15.0, "ACTIVE_DROWNING"),
    (15.0, 9999.0, "PASSIVE_DROWNING"),
]
```

이 상태 역시 실제 Temporal CNN 출력이 아니라 발표용 수동 라벨입니다.

### 28. 라벨 화면 이탈 수정

머리 박스가 프레임 가장자리에 있을 때 `ID 1 | STATE`와 `HEAD confidence`가 화면 밖으로 나가지 않도록 텍스트 크기를 계산해 위·아래·좌우 위치를 제한했습니다.

### 29. 10fps 느낌의 표시 박스

```python
DISPLAY_BOX_HZ = 10
BOX_QUANTIZE_PX = 5
```

표시 박스를 초당 10회 갱신하고 좌표를 5px 단위로 양자화해 위치와 크기가 실제 10fps 추적처럼 단계적으로 변하도록 했습니다.

## 현재 도달점

```text
Camera Module 3 × 2
→ Raspberry Pi 5 + Hailo-8
→ Head Detection
→ ByteTrack / V12 Stable ID
→ 50 timestep temporal sequence
→ Temporal CNN
→ SWIMMING / FLOATING / ACTIVE_DROWNING
→ Long LOST rule
→ PASSIVE_DROWNING
→ Homography
→ target coordinate
→ Arduino Nano
→ BLD-50 ×2 / DMD-150 / A4988
→ 조준 / 발사 / 재장전
```

Hailo detector·classifier·ByteTrack·PASSIVE rule 단일 카메라 런타임은 최종 배포 묶음으로 정리했습니다. 
