#!/usr/bin/env bash
set -Eeuo pipefail

CONNECTION_NAME="PoolSight-Hotspot"
SERVICE_FILE="/etc/systemd/system/pool-calibrator-hotspot.service"

if [[ "${EUID}" -ne 0 ]]; then
  printf 'sudo를 붙여 다시 실행하세요.\n' >&2
  exit 1
fi

systemctl disable --now pool-calibrator-hotspot.service 2>/dev/null || true
if [[ -f "$SERVICE_FILE" ]]; then
  rm -f -- "$SERVICE_FILE"
  systemctl daemon-reload
fi

if nmcli -t -f NAME connection show | grep -Fxq "$CONNECTION_NAME"; then
  nmcli connection down "$CONNECTION_NAME" 2>/dev/null || true
  nmcli connection delete "$CONNECTION_NAME"
fi

printf 'PoolSight 자체 Wi-Fi와 자동 시작 설정을 해제했습니다.\n'
printf '보정 결과와 QR 파일은 /var/lib/pool-calibrator에 그대로 보존했습니다.\n'
