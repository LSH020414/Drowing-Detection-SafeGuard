# Model files

최종 실행과 변환 재현에 필요한 모델 바이너리는 Git LFS로 관리합니다. 원본 데이터, 중간 checkpoint, calibration 배열과 HAR 중간 산출물은 포함하지 않습니다.

## 현재 기준 모델

| 파일 | 크기 | SHA-256 |
|---|---:|---|
| `pool_head_best.pt` | 19,219,025 bytes | `39536439D62B43C7DE210043CB48341C34A53461B7659191AD0A7269B12BE1DE` |
| `classifier_best.pt` | 5,599,231 bytes | `031A31349A245E7357FF6BDC0B0489DBFBC85E142F96A09B9979C2D640821AEA` |
| `drowning_classifier_50x14_hailo.onnx` | 5,597,170 bytes | `3E8D05B8A805AC895106AAFAB9F57814C1B81AD88C9836928623F0B8F72F8982` |
| `pool_head_best_512.onnx` | 37,867,235 bytes | `CA7DAABC445EA6F5DA1A0F040A7A08AC431E080127BB65CAB3A3202343B32FFC` |
| `pool_head_reid_yolo11n_cls_best.pt` | 12,714,657 bytes | `F464DB72995BFD97799388F3F8F02C7C79AC4CCC3918650A883457314BBB2303` |
| `pool_head_best_512.hef` | 17,908,563 bytes | `41563FE40DAE86F24BB07253A2844F724ECE0482DFA6F9B234611CA9E9983FF8` |
| `drowning_classifier_50x14_hailo.hef` | 1,221,442 bytes | `C027C241ED97B6F125124612917C3BCD6EBEB014D7F8DD92CB6B8B4832AB38BD` |

기본 배치 위치:

```text
ai/models/
├─ pool_head_best.pt
├─ classifier_best.pt
├─ drowning_classifier_50x14_hailo.onnx
├─ pool_head_best_512.onnx
└─ pool_head_reid_yolo11n_cls_best.pt
```

두 HEF 파일은 [`deployment/pi5_hailo/`](../../deployment/pi5_hailo/)에 둡니다. 위 모델과 HEF는 모두 Git LFS 객체이며, HailoRT/DFC 버전 호환성을 실제 Pi에서 확인해야 합니다.

clone 후 실제 모델을 받습니다.

```powershell
git lfs install
git lfs pull
```

PowerShell에서 무결성을 확인합니다.

```powershell
Get-FileHash ai/models/pool_head_best.pt -Algorithm SHA256
```

## Git LFS를 사용할 때

저장소의 `.gitattributes`에는 모델 확장자용 LFS 규칙이 이미 들어 있습니다.

```powershell
git lfs install
git add .gitattributes ai/models/
```

GitHub LFS 저장 공간 정책을 확인하고, 모델이 파생된 CrowdHuman·Swim2·실제 영상의 이용 조건도 배포 전에 다시 검토해야 합니다.
