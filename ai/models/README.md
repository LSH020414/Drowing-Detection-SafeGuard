# Model files

모델 바이너리는 기본 Git 이력에 포함하지 않습니다. 필요한 모델은 GitHub Release/외부 저장소로 배포하거나, 저장소에 넣어야 할 때만 Git LFS를 사용합니다.

## 현재 기준 모델

| 파일 | 크기 | SHA-256 |
|---|---:|---|
| `pool_head_best.pt` | 19,219,025 bytes | `39536439D62B43C7DE210043CB48341C34A53461B7659191AD0A7269B12BE1DE` |
| `classifier_best.pt` | 5,599,231 bytes | `031A31349A245E7357FF6BDC0B0489DBFBC85E142F96A09B9979C2D640821AEA` |
| `drowning_classifier_50x14_hailo.onnx` | 5,597,170 bytes | `3E8D05B8A805AC895106AAFAB9F57814C1B81AD88C9836928623F0B8F72F8982` |

기본 배치 위치:

```text
ai/models/
├─ pool_head_best.pt
├─ classifier_best.pt
└─ drowning_classifier_50x14_hailo.onnx
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

GitHub LFS 저장 공간 정책을 먼저 확인하고, 모델 라이선스·학습 데이터 배포 권한도 검토해야 합니다.
