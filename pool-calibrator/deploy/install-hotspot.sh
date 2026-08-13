#!/usr/bin/env bash
set -Eeuo pipefail

SSID="PoolSight-Setup"
PASSWORD=""
WIFI_INTERFACE="wlan0"
WIFI_COUNTRY="KR"
CONNECTION_NAME="PoolSight-Hotspot"
POOLSIGHT_HOST_NAME="poolsight"
SITE_NAME="PoolSight"
BACKUP_URL="http://10.42.0.1"
DATA_DIR="/var/lib/pool-calibrator"

usage() {
  printf '%s\n' \
    "사용법: sudo bash deploy/install-hotspot.sh [옵션]" \
    "" \
    "옵션:" \
    "  --ssid NAME        자체 Wi-Fi 이름 (기본값: PoolSight-Setup)" \
    "  --password PASS    Wi-Fi 암호 (생략하면 안전한 암호 자동 생성)" \
    "  --host-name NAME   접속 주소 이름 (기본값: poolsight → poolsight.local)" \
    "  --site-name NAME   페이지에 표시할 이름 (기본값: PoolSight)" \
    "  --interface NAME   Wi-Fi 장치명 (기본값: wlan0)" \
    "  --country CODE     Wi-Fi 국가 코드 (기본값: KR)" \
    "  --help             도움말"
}

while (($#)); do
  case "$1" in
    --ssid) SSID="${2:-}"; shift 2 ;;
    --password) PASSWORD="${2:-}"; shift 2 ;;
    --host-name) POOLSIGHT_HOST_NAME="${2:-}"; shift 2 ;;
    --site-name) SITE_NAME="${2:-}"; shift 2 ;;
    --interface) WIFI_INTERFACE="${2:-}"; shift 2 ;;
    --country) WIFI_COUNTRY="${2:-}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) printf '알 수 없는 옵션: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "${EUID}" -ne 0 ]]; then
  printf 'sudo를 붙여 다시 실행하세요.\n' >&2
  exit 1
fi

if [[ ! "$SSID" =~ ^[A-Za-z0-9_-]{1,32}$ ]]; then
  printf 'Wi-Fi 이름은 영문, 숫자, 밑줄, 하이픈만 사용해 1~32자로 입력하세요.\n' >&2
  exit 1
fi
if [[ ! "$WIFI_COUNTRY" =~ ^[A-Za-z]{2}$ ]]; then
  printf '국가 코드는 KR처럼 영문 두 글자로 입력하세요.\n' >&2
  exit 1
fi
if [[ ! "$POOLSIGHT_HOST_NAME" =~ ^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$ ]]; then
  printf '접속 주소 이름은 영문 소문자, 숫자, 하이픈만 사용해 1~63자로 입력하세요.\n' >&2
  exit 1
fi
if [[ -z "$SITE_NAME" || ${#SITE_NAME} -gt 60 || "$SITE_NAME" == *$'\n'* ]]; then
  printf '페이지 이름은 줄바꿈 없이 1~60자로 입력하세요.\n' >&2
  exit 1
fi
SITE_URL="http://${POOLSIGHT_HOST_NAME}.local"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
APP_USER="${SUDO_USER:-}"
if [[ -z "$APP_USER" || "$APP_USER" == "root" ]]; then
  APP_USER="$(getent passwd 1000 | cut -d: -f1 || true)"
fi
if [[ -z "$APP_USER" ]]; then
  printf '앱을 실행할 일반 사용자를 찾지 못했습니다. root가 아닌 사용자로 로그인한 뒤 sudo로 실행하세요.\n' >&2
  exit 1
fi
APP_GROUP="$(id -gn "$APP_USER")"

if [[ ! -f "$APP_DIR/app.py" || ! -f "$APP_DIR/requirements.txt" ]]; then
  printf '프로젝트 폴더의 deploy 디렉터리에서 이 설치 파일을 실행해야 합니다.\n' >&2
  exit 1
fi

printf '[1/6] 필요한 프로그램을 설치합니다. 최초 1회는 인터넷 연결이 필요합니다.\n'
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv libgl1 libglib2.0-0 network-manager avahi-daemon rpicam-apps qrencode curl

if [[ -z "$PASSWORD" ]]; then
  PASSWORD="$(python3 -c 'import secrets; alphabet="ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789"; print("".join(secrets.choice(alphabet) for _ in range(14)))')"
fi
if ((${#PASSWORD} < 8 || ${#PASSWORD} > 63)); then
  printf 'Wi-Fi 암호는 8~63자로 입력하세요.\n' >&2
  exit 1
fi
if [[ ! "$PASSWORD" =~ ^[A-Za-z0-9_-]+$ ]]; then
  printf 'Wi-Fi 암호는 QR 호환성을 위해 영문, 숫자, 밑줄, 하이픈만 사용하세요.\n' >&2
  exit 1
fi

printf '[2/6] 전용 Python 실행 환경을 준비합니다.\n'
chown -R "$APP_USER:$APP_GROUP" "$APP_DIR"
if [[ ! -x "$APP_DIR/.venv-pi/bin/python" ]]; then
  sudo -u "$APP_USER" python3 -m venv "$APP_DIR/.venv-pi"
fi
sudo -u "$APP_USER" "$APP_DIR/.venv-pi/bin/python" -m pip install --upgrade pip
sudo -u "$APP_USER" "$APP_DIR/.venv-pi/bin/pip" install -r "$APP_DIR/requirements.txt"

printf '[3/6] 웹페이지 자동 시작을 설정합니다.\n'
install -d -o "$APP_USER" -g "$APP_GROUP" -m 0750 "$DATA_DIR"
python3 -c 'import json,sys; open(sys.argv[1], "w", encoding="utf-8").write(json.dumps({"site_name": sys.argv[2], "host_name": sys.argv[3]}, ensure_ascii=False, indent=2))' "$DATA_DIR/site-settings.json" "$SITE_NAME" "$POOLSIGHT_HOST_NAME"
chown "$APP_USER:$APP_GROUP" "$DATA_DIR/site-settings.json"
EXTRA_GROUPS="video"
if getent group render >/dev/null; then EXTRA_GROUPS="${EXTRA_GROUPS},render"; fi
usermod -a -G "$EXTRA_GROUPS" "$APP_USER"
sed \
  -e "s|@APP_USER@|$APP_USER|g" \
  -e "s|@APP_GROUP@|$APP_GROUP|g" \
  -e "s|@APP_DIR@|$APP_DIR|g" \
  -e "s|@DATA_DIR@|$DATA_DIR|g" \
  "$SCRIPT_DIR/pool-calibrator-hotspot.service.template" \
  > /etc/systemd/system/pool-calibrator-hotspot.service
systemctl disable --now pool-calibrator.service 2>/dev/null || true
systemctl daemon-reload
systemctl enable pool-calibrator-hotspot.service
systemctl restart pool-calibrator-hotspot.service

printf '[4/6] 휴대폰 연결용 QR 코드와 안내 카드를 만듭니다.\n'
qrencode -l H -s 8 -m 2 -o "$DATA_DIR/poolsight-wifi-qr.png" "WIFI:T:WPA;S:${SSID};P:${PASSWORD};H:false;;"
qrencode -l H -s 8 -m 2 -o "$DATA_DIR/poolsight-site-qr.png" "$SITE_URL"
sed \
  -e "s|@SSID@|$SSID|g" \
  -e "s|@PASSWORD@|$PASSWORD|g" \
  -e "s|@SITE_URL@|$SITE_URL|g" \
  -e "s|@BACKUP_URL@|$BACKUP_URL|g" \
  "$SCRIPT_DIR/access-card.html.template" \
  > "$DATA_DIR/poolsight-access-card.html"
printf '페이지 이름: %s\nWi-Fi 이름: %s\nWi-Fi 암호: %s\n웹 주소: %s\n예비 주소: %s\n' "$SITE_NAME" "$SSID" "$PASSWORD" "$SITE_URL" "$BACKUP_URL" > "$DATA_DIR/hotspot-info.txt"
chown "$APP_USER:$APP_GROUP" "$DATA_DIR"/poolsight-*.png "$DATA_DIR/poolsight-access-card.html" "$DATA_DIR/hotspot-info.txt"
chmod 0640 "$DATA_DIR/hotspot-info.txt"

printf '[5/6] 라즈베리파이 자체 Wi-Fi를 설정합니다.\n'
systemctl enable --now NetworkManager.service
hostnamectl set-hostname "$POOLSIGHT_HOST_NAME"
systemctl enable --now avahi-daemon.service
if command -v raspi-config >/dev/null 2>&1; then
  raspi-config nonint do_wifi_country "${WIFI_COUNTRY^^}"
fi
rfkill unblock wifi 2>/dev/null || true

if ! nmcli -t -f DEVICE device status | grep -Fxq "$WIFI_INTERFACE"; then
  printf 'Wi-Fi 장치 %s를 찾을 수 없습니다. --interface 옵션을 확인하세요.\n' "$WIFI_INTERFACE" >&2
  exit 1
fi
AP_SUPPORT="$(nmcli -g WIFI-PROPERTIES.AP device show "$WIFI_INTERFACE" 2>/dev/null || true)"
if [[ "${AP_SUPPORT,,}" != "yes" ]]; then
  printf 'Wi-Fi 장치 %s가 핫스팟 모드를 지원하지 않습니다.\n' "$WIFI_INTERFACE" >&2
  exit 1
fi

if nmcli -t -f NAME connection show | grep -Fxq "$CONNECTION_NAME"; then
  nmcli connection delete "$CONNECTION_NAME"
fi
nmcli connection add type wifi ifname "$WIFI_INTERFACE" con-name "$CONNECTION_NAME" ssid "$SSID"
nmcli connection modify "$CONNECTION_NAME" \
  connection.autoconnect yes \
  connection.autoconnect-priority 100 \
  802-11-wireless.mode ap \
  802-11-wireless.band bg \
  802-11-wireless.channel 6 \
  802-11-wireless-security.key-mgmt wpa-psk \
  802-11-wireless-security.psk "$PASSWORD" \
  ipv4.method shared \
  ipv4.addresses 10.42.0.1/24 \
  ipv6.method disabled

printf '[6/6] 자체 Wi-Fi를 시작합니다. 현재 Wi-Fi로 SSH 접속 중이면 연결이 끊길 수 있습니다.\n'
printf '\n접속 정보를 먼저 기록하세요.\n'
printf '  Wi-Fi 이름: %s\n' "$SSID"
printf '  Wi-Fi 암호: %s\n' "$PASSWORD"
printf '  웹 주소: %s\n' "$SITE_URL"
printf '5초 뒤 자체 Wi-Fi로 전환합니다.\n\n'
sleep 5
nmcli connection up "$CONNECTION_NAME"

WEB_READY=false
for _ in {1..15}; do
  if curl --silent --fail --max-time 2 "$BACKUP_URL/health" >/dev/null; then
    WEB_READY=true
    break
  fi
  sleep 1
done
if [[ "$WEB_READY" != true ]]; then
  printf '자체 Wi-Fi는 시작했지만 웹페이지 응답을 확인하지 못했습니다.\n' >&2
  systemctl status pool-calibrator-hotspot.service --no-pager >&2 || true
  exit 1
fi

printf '\n설치가 완료되었습니다.\n'
printf '  1. 휴대폰 Wi-Fi에서 %s 선택\n' "$SSID"
printf '  2. 암호: %s\n' "$PASSWORD"
printf '  3. 브라우저에서 %s 열기\n' "$SITE_URL"
printf '     이름 주소가 열리지 않으면 %s 사용\n' "$BACKUP_URL"
printf '  4. 인쇄용 접속 카드: %s/poolsight-access-card.html\n' "$DATA_DIR"
printf '\n이 정보는 %s/hotspot-info.txt에도 저장되었습니다.\n' "$DATA_DIR"
