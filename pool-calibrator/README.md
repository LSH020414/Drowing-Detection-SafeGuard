# PoolSight 최초 설치용 카메라 보정 페이지

두 카메라의 수영장 모서리를 지정해 픽셀 좌표를 실제 수영장 미터 좌표로 변환하는 로컬 웹 도구입니다. 이미지와 설정은 외부 서버로 전송하지 않고 이 장치의 `data/` 폴더에 저장됩니다.

## 포함 기능

- 휴대폰/태블릿/PC 반응형 화면
- 카메라 1·2 이미지 촬영 또는 파일 업로드
- Raspberry Pi 부팅 시 CSI 카메라 1·2 자동 촬영 및 점 지정 영역 자동 표시
- 웹페이지의 카메라 1·2 수동 재촬영
- 네 모서리를 아무 순서로 선택하면 이미지 기준으로 자동 정렬
- 터치나 마우스로 지정점 드래그 수정
- 수영장 길이·폭 및 선택 영역 검증
- OpenCV 호모그래피 계산과 탑뷰 미리보기
- 미리보기 90° 회전·좌우 반전 및 방향 변환의 호모그래피·설정 반영
- 두 카메라 탑뷰를 겹친 미터 단위 X·Y 좌표 이미지
- 두 탑뷰의 수영장 영역을 비교한 상대 WB(R/B gain)·노출(EV) 자동 계산
- 두 카메라의 `config.json` 원자적 저장 및 다운로드

## PC에서 실행

Python 3.11 이상을 권장합니다.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

같은 PC에서는 `http://127.0.0.1:5000`, 같은 Wi-Fi의 휴대폰에서는 `http://PC의-IP주소:5000`을 엽니다. Windows 방화벽이 묻는 경우 개인 네트워크 접근만 허용합니다.

## 사용 순서

1. 수영장 길이와 폭을 미터 단위로 입력합니다.
2. Raspberry Pi 카메라가 연결되어 있으면 부팅 때 촬영된 카메라 1·2 이미지가 자동으로 표시됩니다. PC 테스트에서는 이미지를 직접 추가합니다.
3. 실제 물 영역의 네 모서리를 순서와 관계없이 누릅니다. 0° 미리보기는 원본 이미지 위쪽이 위로 오도록 자동 정렬됩니다.
4. 원을 드래그해 위치를 조정하고 **원근 변환 미리보기**를 누릅니다.
5. 미리보기 아래의 좌회전·우회전·좌우 반전 버튼으로 실제 수영장 방향을 맞춥니다. 서로 거울상처럼 보이면 두 카메라 중 하나만 좌우 반전합니다.
6. 카메라 2도 같은 방식으로 보정하고, 두 탑뷰의 방향을 동일하게 맞춥니다.
7. **XY 좌표 적용 이미지 만들기**를 눌러 두 탑뷰와 미터 격자가 포개지는지 확인합니다. 이때 WB와 노출 보정값도 자동으로 계산됩니다.
8. 수영장 형태와 WB·노출 결과를 확인한 뒤 **config.json 저장**을 누릅니다.

서버에서 저장된 파일은 기본적으로 `data/config.json`입니다. `POOL_CALIBRATOR_DATA_DIR` 환경 변수를 설정하면 저장 위치를 바꿀 수 있습니다.

## config.json 좌표 정의

- 원점 `(0, 0)`: 수영장 좌상단
- X축: 수영장 길이 방향
- Y축: 수영장 폭 방향
- 단위: 미터
- `rotation_degrees_clockwise`: 미리보기에서 선택한 시계 방향 회전값
- `flip_horizontal`: 회전 후 미리보기에 적용한 좌우 반전 여부
- `homography_pixel_to_meter`: 선택한 회전까지 적용해 카메라 픽셀 `[x, y, 1]`을 최종 미터 좌표로 바꾸는 3×3 행렬
- `image_adjustments.white_balance`: 두 카메라 색을 맞추기 위해 현재 `ColourGains`에 곱할 R/G/B 배율
- `image_adjustments.exposure.compensation_ev`: 자동 노출 목표에 더할 EV 보정값

```python
import cv2
import numpy as np

pixel = np.array([[[camera_x, camera_y]]], dtype=np.float32)
pool_xy = cv2.perspectiveTransform(pixel, homography_matrix)[0, 0]
```

## WB와 노출값 적용

웹페이지는 원근 변환된 수영장 영역에서 밝기 상·하위 10%와 잘린 픽셀을 제외하고 두 카메라의 중간 RGB·휘도를 비교합니다. 한쪽 카메라를 절대 기준으로 삼지 않고 두 영상의 기하평균을 공통 목표로 사용하므로, 수영장 고유의 푸른색을 흰색으로 만들어 버리는 일반적인 그레이월드 방식보다 카메라 간 색상 일치에 적합합니다.

업로드된 JPEG만으로 원래 센서의 `ExposureTime`, `AnalogueGain`, `ColourGains` 절대값을 복원할 수는 없습니다. 따라서 저장값은 다음처럼 사용합니다.

```python
metadata = picam2.capture_metadata()
red, blue = metadata["ColourGains"]
adjustment = config["cameras"][camera_index]["image_adjustments"]

picam2.set_controls({
    "AwbEnable": False,
    "ColourGains": (
        red * adjustment["white_balance"]["red_gain_multiplier"],
        blue * adjustment["white_balance"]["blue_gain_multiplier"],
    ),
    "ExposureValue": adjustment["exposure"]["compensation_ev"],
})
```

이 방식은 현재 자동 WB가 수렴한 값을 기준으로 색을 고정하고, 자동 노출은 유지하면서 두 카메라의 목표 밝기를 맞춥니다. 추후 카메라 직접 캡처 기능이 연결되면 캡처 메타데이터의 노출시간과 아날로그 게인도 함께 저장할 수 있습니다.

## Raspberry Pi 5 배포

Raspberry Pi OS 64-bit(Bookworm 이상), Python 3.11+, 같은 로컬 네트워크를 기준으로 합니다.

```bash
sudo apt update
sudo apt install -y python3-venv libgl1 libglib2.0-0
sudo mkdir -p /opt/pool-calibrator /var/lib/pool-calibrator
sudo cp -a . /opt/pool-calibrator/
sudo chown -R pi:pi /opt/pool-calibrator /var/lib/pool-calibrator
cd /opt/pool-calibrator
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

수동 실행으로 먼저 확인합니다.

```bash
POOL_CALIBRATOR_DATA_DIR=/var/lib/pool-calibrator .venv/bin/gunicorn --workers 1 --threads 4 --bind 0.0.0.0:5000 app:app
```

휴대폰에서 `http://라즈베리파이-IP:5000`을 열어 동작을 확인합니다. IP는 Raspberry Pi에서 `hostname -I`로 확인할 수 있습니다.

자동 시작을 설정하려면 아래를 실행합니다. Raspberry Pi 사용자명이 `pi`가 아니라면 먼저 `deploy/pool-calibrator.service`의 `User`와 `Group`을 바꿉니다.

```bash
sudo cp deploy/pool-calibrator.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now pool-calibrator
sudo systemctl status pool-calibrator
```

로그 확인:

```bash
journalctl -u pool-calibrator -f
```

실제 설치 환경에서는 이 페이지를 인터넷에 직접 노출하지 말고, 카메라 설치용 로컬 Wi-Fi 또는 신뢰할 수 있는 내부망에서만 사용하세요.

## Wi-Fi와 공유기가 없는 현장: 권장 방식

라즈베리파이가 `PoolSight-Setup`이라는 자체 Wi-Fi를 만들도록 한 번만 준비해 두면 됩니다. 현장에서는 인터넷이 전혀 없어도 라즈베리파이 전원을 켜고 휴대폰으로 직접 연결할 수 있습니다.

> 최초 설치 과정에서 운영체제 패키지와 Python 패키지를 받기 위해 인터넷 연결이 한 번 필요합니다. 설치가 끝난 뒤 현장 사용에는 인터넷이 필요하지 않습니다.

Raspberry Pi OS Bookworm 이상에서 프로젝트 폴더로 이동한 뒤 실행합니다.

```bash
sudo bash deploy/install-hotspot.sh
```

설치 파일이 다음 작업을 자동으로 처리합니다.

- 2.4GHz 자체 Wi-Fi와 고정 주소 `10.42.0.1` 생성
- Wi-Fi 보안 암호 자동 생성
- 웹페이지를 80번 포트에서 부팅 시 자동 실행
- 휴대폰 연결용 Wi-Fi QR 및 웹 주소 QR 생성
- 보정 결과를 `/var/lib/pool-calibrator`에 보존
- 부팅 시 두 CSI 카메라를 차례로 촬영하고 웹페이지에 자동 표시
- `poolsight.local` 이름 주소와 `10.42.0.1` 예비 주소 제공

설치가 완료되면 화면에 표시되는 암호를 기록합니다. 인쇄용 연결 카드는 `/var/lib/pool-calibrator/poolsight-access-card.html`에 생성됩니다.

현장에서는 다음 세 단계만 수행합니다.

1. 라즈베리파이 전원을 켜고 약 30~60초 기다립니다.
2. 휴대폰으로 `PoolSight-Setup` Wi-Fi에 연결합니다. “인터넷 없음” 표시는 정상입니다.
3. 브라우저 주소창에 `http://poolsight.local`을 입력합니다. 열리지 않으면 `http://10.42.0.1`을 사용합니다.

Wi-Fi 이름이나 암호를 직접 정하려면 최초 설치할 때 옵션을 지정합니다.

```bash
sudo bash deploy/install-hotspot.sh --ssid PoolSight-Setup --password MyPool2026
```

Wi-Fi 이름, 브라우저 접속 주소, 페이지 표시 이름은 각각 따로 지정할 수 있습니다.

```bash
sudo bash deploy/install-hotspot.sh \
  --ssid SchoolPool-Setup \
  --host-name school-pool \
  --site-name "학교 수영장"
```

위 예시의 결과는 다음과 같습니다.

- 휴대폰에 표시되는 Wi-Fi: `SchoolPool-Setup`
- 브라우저 주소: `http://school-pool.local`
- 웹페이지 제목: `학교 수영장`
- 이름 주소가 지원되지 않는 휴대폰의 예비 주소: `http://10.42.0.1`

## 부팅 자동 촬영

오프라인 설치 파일은 `rpicam-still --list-cameras`에 표시되는 0번 카메라를 카메라 1, 1번 카메라를 카메라 2로 사용합니다. 부팅할 때 자동 노출·AWB·자동초점이 안정될 시간을 준 뒤 1920×1080 이미지를 차례로 촬영합니다.

촬영 결과는 `/var/lib/pool-calibrator/latest-captures.json`과 `uploads/`에 저장됩니다. 페이지를 열면 최신 이미지가 각 점 지정 영역에 자동으로 들어갑니다. 설치 후 카메라 연결 순서는 다음 명령으로 확인할 수 있습니다.

```bash
rpicam-hello --list-cameras
```

카메라가 빠져 있거나 촬영에 실패해도 웹페이지 자체는 시작됩니다. 연결을 바로잡은 뒤 페이지의 **카메라 1·2 다시 촬영**을 누르면 됩니다.

자체 Wi-Fi 설정만 해제하려면 다음을 실행합니다. 저장된 보정 결과는 삭제하지 않습니다.

```bash
sudo bash deploy/uninstall-hotspot.sh
```

상태와 로그 확인:

```bash
systemctl status pool-calibrator-hotspot
journalctl -u pool-calibrator-hotspot -f
nmcli connection show PoolSight-Hotspot
```

## 테스트

```bash
python -m unittest discover -s tests -v
```

테스트는 이미지 업로드, 네 점 자동 정렬, 회전·좌우 반전 호모그래피, XY 겹침 미리보기, 두 카메라 설정 저장을 확인합니다.
