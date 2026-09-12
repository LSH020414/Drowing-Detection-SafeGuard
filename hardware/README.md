# Hardware control

## 확정 목표 구조

```text
Raspberry Pi 5
  └─ USB Serial
      └─ Arduino Nano (ATmega328P, 5V, 16MHz)
          ├─ BLD-50 #1 → 상부 BLDC 발사 휠
          ├─ BLD-50 #2 → 하부 BLDC 발사 휠
          ├─ DMD-150   → 수평 DC 웜기어모터
          └─ A4988     → 수직 NEMA17
```

Nano는 모터 전력을 공급하지 않고 PWM/DIR/STEP/ENABLE 제어 신호만 전달합니다. 모터 전원과 로직 전원은 장치 사양에 맞게 분리하고 공통 GND·절연·비상정지를 설계해야 합니다.

## 제안 핀 배치

| Nano pin | 대상 | 신호 |
|---|---|---|
| D3 | BLD-50 #1 | PWM |
| D4 | BLD-50 #1 | DIR |
| D5 | BLD-50 #2 | PWM |
| D7 | BLD-50 #2 | DIR |
| D6 | DMD-150 | PWM |
| D8 | DMD-150 | DIR |
| D9 | A4988 | STEP |
| D10 | A4988 | DIR |
| D12 | A4988 | ENABLE |

`D0/RX`, `D1/TX`는 USB serial과 충돌을 피하기 위해 비워 둡니다. D2/D11/D13/A0~A5는 home/limit/E-stop 등 안전 입력 후보입니다.

## 명령 계층

```text
Pi5: TARGET(x, y)와 발사 조건 계산
→ Nano: AIM / SPIN / FIRE / STOP 명령 파싱
→ Driver: 장치별 제어 신호 생성
→ Sensor: home / limit / E-stop / RPM feedback
```

실제 발사 명령은 좌표 유효성, 조준 완료, flywheel 속도 안정, 장전 상태, limit/E-stop을 모두 통과한 뒤에만 허용해야 합니다.

## 현재 저장소와의 차이

`pool-calibrator/firmware/`의 현재 작업본은 ESP32/BDA3502P/TB6600 기반 프로토타입입니다. 위의 최종 목표인 **Arduino Nano + BLD-50×2 + DMD-150 + A4988**과 MCU·드라이버 구성이 다릅니다. 해당 프로토타입을 Nano 최종 펌웨어로 간주하지 않으며, 핀 전압과 통신 프로토콜을 확정한 뒤 별도 구현·검증해야 합니다.

고출력 회전체와 발사 장치 시험은 방호 덮개, 물리적 전원 차단, 무부하 단계 시험을 갖춘 상태에서 진행해야 합니다.
