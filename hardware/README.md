# Hardware control

## 현재 확정된 MCU 구조

```text
Raspberry Pi 5
  ├─ USB Serial → Arduino Nano/UNO
  │                  └─ D2/IN1, D3/PWM, D4/IN2 → 수평 회전 모터 드라이버
  └─ USB Serial → ESP32
                     ├─ GPIO25/PWM1, GPIO26/DIR1 → BLDC #1 driver
                     └─ GPIO32/PWM2, GPIO33/DIR2 → BLDC #2 driver
```

두 MCU는 Raspberry Pi의 판단 결과를 받아 actuator 명령을 실행합니다. 모터 전력을 직접 공급하지 않으므로 드라이버와 모터 전원은 장치 사양에 맞게 분리하고, 신호 기준 전압·공통 GND·절연·비상정지를 실제 배선에서 검증해야 합니다.

## 최종 소스와 핀

| MCU | 소스 | 핀 |
|---|---|---|
| Arduino Nano/UNO | [`firmware/arduino_nano_uno/arduino_launcher_motor.ino`](firmware/arduino_nano_uno/arduino_launcher_motor.ino) | D2→IN1, D3→PWM, D4→IN2 |
| ESP32 | [`firmware/esp32/esp32_bldc_controller.ino`](firmware/esp32/esp32_bldc_controller.ino) | PWM1=25, DIR1=26, PWM2=32, DIR2=33 |

ESP32의 고정 회전 방향은 DIR1=`HIGH`, DIR2=`LOW`입니다.

## Raspberry Pi 텍스트 프로토콜

모든 명령은 ASCII 한 줄이며 `\n`으로 끝납니다. 숫자 명령과 읽기 쉬운 영문 명령을 함께 지원합니다.

### Arduino Nano/UNO: 수평 회전

| 명령 | 동작 |
|---|---|
| `0` 또는 `STOP` | 즉시 정지 |
| `1` 또는 `RIGHT` | 상태 각도 +10° |
| `2` 또는 `LEFT` | 상태 각도 -10° |
| `STATUS` 또는 `?` | 현재 각도와 동작 상태 조회 |

Serial은 9600 baud입니다. 소프트웨어 각도 상태는 -47°~+87° 범위로 관리되며, 다음 10° 이동이 범위를 넘으면 `ERR ANGLE_LIMIT`로 무시합니다. 실행 중 추가 이동 명령은 `ERR BUSY`, 정상 수신은 `ACK`, 상태 조회는 `STATUS`로 응답합니다. 10° 이동은 기본 500ms open-loop 시간 제어이므로 실제 장치에서 `MOVE_10_DEG_MS`를 보정해야 합니다.

### ESP32: BLDC 발사 휠 2개

| 명령 | 동작 |
|---|---|
| `0` 또는 `STOP` | 두 PWM 즉시 0% |
| `1` 또는 `SPEED_UP` | 목표 속도 +5%, 최대 80% |
| `2` 또는 `SPEED_DOWN` | 목표 속도 -5%, 최소 0% |
| `STATUS` 또는 `?` | 목표 속도와 kick 상태 조회 |

Serial은 115200 baud입니다. 정지 상태에서 처음 구동할 때 두 출력에 100% PWM을 450ms 적용한 뒤 목표 속도로 내려갑니다. 정상 수신은 `ACK`, 제한 또는 잘못된 명령은 `ERR`, 상태 조회는 `STATUS`로 응답합니다.

## 구현 범위와 안전 조건

이 두 소스는 수평 회전과 BLDC flywheel 속도만 담당합니다. Raspberry Pi가 두 serial 장치를 구분해 명령 순서를 조정해야 하며, 수직 A4988 제어, 장전·투입 액추에이터, 센서 기반 조준 완료, RPM feedback, limit/home, E-stop interlock은 이 펌웨어에 포함되지 않았습니다.

Nano/UNO 각도는 encoder가 아닌 이동시간으로 추정하고, ESP32 속도는 RPM feedback 없는 PWM 비율입니다. 실제 발사 허용은 좌표 유효성, 조준 완료, flywheel 속도 안정, 장전 상태, limit/E-stop을 별도 하드웨어와 상위 제어에서 모두 확인한 뒤에만 수행해야 합니다.

## 현재 저장소와의 차이

`pool-calibrator/firmware/`에 존재할 수 있는 ESP32/BDA3502P/TB6600 작업본은 보정 도구와 함께 개발된 별도 프로토타입입니다. 이 문서에 연결된 `hardware/firmware/`의 두 파일을 Raspberry Pi 연계용 최종 MCU 소스로 사용합니다.

고출력 회전체와 발사 장치 시험은 방호 덮개, 물리적 전원 차단, 무부하 단계 시험을 갖춘 상태에서 진행해야 합니다.
