# Demo tracking video

`make_demo_tracking_video.py`는 발표 영상용입니다. 실제 TCN 판별 결과가 아니라 사용자가 지정한 시간 구간에 상태 라벨을 표시합니다.

주요 설정:

- `VIDEO_NAME`: `VIDEO_ROOT` 아래 입력 영상 이름
- `STATE_TIMELINE`: 시간별 `SWIMMING/FLOATING/ACTIVE_DROWNING/PASSIVE_DROWNING`
- `DISPLAY_BOX_HZ = 10`: 표시 박스 갱신 주파수
- `BOX_QUANTIZE_PX = 5`: 박스 좌표 양자화
- `DISPLAY_ID = 1`: 발표용 고정 표시 ID

검출은 실제 `pool_head_best.pt`와 ByteTrack을 사용하지만, ID와 상태는 시연용 표시값입니다. 결과를 실제 모델 성능 자료로 사용하면 안 됩니다.
