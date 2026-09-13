# 익수 탐지 웹페이지 작업 인계

## 목적

Raspberry Pi 5와 Camera Module 3 두 대를 사용하는 익수 탐지 시스템의 최초 설치용 로컬 웹페이지다. 두 카메라를 실제 수영장 미터 좌표에 맞게 보정하고, 이후 YOLO·ByteTrack·Global ID 파이프라인에서 사용할 `config.json`을 만든다.

## 구현 완료

- 모바일·태블릿·PC 반응형 화면
- 카메라 1·2 이미지 업로드
- Raspberry Pi 부팅 시 CSI 카메라 두 대 자동 촬영
- 웹페이지에서 카메라 1·2 재촬영
- 수영장 모서리 네 점 선택 및 이미지 기준 자동 정렬
- 작은 빨간 점과 번호 표시
- 터치·마우스 드래그 수정
- 수영장 길이·폭 및 선택 영역 검증
- OpenCV 호모그래피 계산
- 원근 변환 탑뷰 미리보기
- 90도 회전과 좌우 반전의 실제 좌표 행렬 반영
- 두 카메라 탑뷰 50% 합성과 미터 단위 X·Y 격자
- 상대 WB R/B gain 및 노출 EV 계산
- `config.json` 원자적 저장과 다운로드

## 두 카메라 방향 기준

최종 탑뷰에서는 다음 실제 방향을 동일하게 맞춘다.

```text
같은 실제 수영장 끝부분 → 모두 위쪽
같은 실제 왼쪽 벽면 → 모두 왼쪽
같은 실제 오른쪽 벽면 → 모두 오른쪽
```

카메라가 서로 반대편을 바라봐 원본 좌우가 반대일 때는 한쪽만 좌우 반전한다. 수면의 창문·조명 반사는 시점에 따라 달라지므로 검증 기준으로 삼지 않는다. 타일, 배수구, 사다리, 수중 표시나 별도 기준점을 사용한다.

## WB 및 노출

원근 변환된 두 수영장 영역의 중간 RGB·휘도를 비교한다. 지나치게 밝거나 어두운 픽셀은 제외하고 두 영상의 기하평균을 공통 목표로 사용한다.

저장 위치:

```text
config.cameras[n].image_adjustments.white_balance
config.cameras[n].image_adjustments.exposure
```

현재 값은 상대 보정값이다. Raspberry Pi에서는 수렴한 `ColourGains`에 WB 배율을 곱하고 노출값은 `ExposureValue`에 적용한다. 절대 `ExposureTime`과 `AnalogueGain` 저장은 실제 카메라 메타데이터 연결 단계에서 추가해야 한다.

## Raspberry Pi 자동 촬영

```text
부팅
→ 카메라 두 대 감지
→ 카메라 인덱스 0과 1을 1920×1080으로 차례로 촬영
→ 웹서버 시작
→ 접속 시 최신 이미지를 두 점 지정 영역에 자동 표시
```

카메라 연결 순서는 다음 명령으로 확인한다.

```bash
rpicam-hello --list-cameras
```

카메라 촬영에 실패해도 웹서버는 시작되며 직접 업로드하거나 웹의 **카메라 1·2 다시 촬영** 버튼을 사용할 수 있다.

## 인터넷 없는 현장 접속

설치 스크립트가 Raspberry Pi 자체 Wi-Fi를 만든다.

```text
기본 Wi-Fi: PoolSight-Setup
고정 주소: http://10.42.0.1
```

설치:

```bash
sudo bash deploy/install-hotspot.sh
```

최초 설치 때만 패키지 다운로드를 위한 인터넷 연결이 필요하다. 이후 현장에서는 인터넷이나 공유기 없이 동작한다.

Wi-Fi 이름, 호스트 이름, 페이지 표시 이름은 따로 지정할 수 있다.

```bash
sudo bash deploy/install-hotspot.sh \
  --ssid SchoolPool-Setup \
  --host-name school-pool \
  --site-name "학교 수영장"
```

## 알려진 보류 사항

`.local` 이름 주소는 실제 테스트에서 열리지 않았다.

```text
http://poolsight.local
http://school-pool.local
```

이 문제는 사용자 요청에 따라 수정하지 않고 보류했다. 실제 테스트와 현장 사용에서는 `http://10.42.0.1`을 우선 사용한다.

## config.json

- 스키마 버전: 4
- 수영장 길이·폭
- 이미지 크기와 사용 이미지 경로
- 네 모서리 픽셀 좌표
- 회전과 좌우 반전 상태
- 픽셀→미터 호모그래피
- 미리보기 호모그래피와 최종 좌표계 크기
- WB 보정 배율과 노출 EV
- 생성 시각

## 검증 상태

자동 테스트 12개가 통과했다.

- 상태 API와 홈 화면
- 이미지 업로드
- 네 점 자동 정렬
- 호모그래피·회전·좌우 반전
- 두 카메라 XY 합성
- WB·노출 계산
- 자동 촬영 이미지 로드
- 수동 재촬영 API
- 사이트 이름 변경
- `config.json` 저장
- 서로 다른 미리보기 방향 거부

모바일 화면의 가로 넘침과 JavaScript 오류도 확인했다.

## 실제 장비에서 남은 검증

1. Raspberry Pi 5에 Camera Module 3 두 대 연결
2. 카메라 인덱스와 설치 위치 대응 확인
3. 부팅 자동 촬영 확인
4. 휴대폰에서 `http://10.42.0.1` 접속
5. 두 이미지 자동 표시와 재촬영 확인
6. 실제 고정 기준점으로 호모그래피 오차 측정
7. WB·EV 적용 후 두 영상 색과 밝기 비교
8. Picamera2의 `ExposureTime`, `AnalogueGain`, `ColourGains` 메타데이터 저장 연결

## 다음 단계

```text
자동 촬영
→ 모서리 보정
→ 호모그래피 저장
→ YOLO 검출
→ ByteTrack Local ID
→ 미터 좌표 변환
→ 두 카메라 Global ID 통합
→ FSM
→ Motion Score
→ 익수 판별
```
