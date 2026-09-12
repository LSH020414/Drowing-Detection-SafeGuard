# Source manifest

정리 시점에 확인한 원본 파일의 수정 시각과 원본 바이트 기준 SHA-256입니다. 저장소의 코드 복사본은 Git 정책에 맞춰 줄바꿈만 LF로 정규화했으며, 정규화 후 텍스트 내용이 원본과 일치하는 것을 확인했습니다. PDF 발표서는 원본 바이트를 변경하지 않고 그대로 복사했습니다.

| 저장소 파일 | 원본 수정 시각 | SHA-256 |
|---|---|---|
| `ai/pipeline/drowning_main_v12.py` | 2026-08-29 14:34:48 | `F0E4F19AB88B934C9C11E74DE1B472CCD951702ABE75BD6F8CA6298BEAD55E64` |
| `ai/training/train_classifier_hailo_v1.py` | 2026-09-03 21:04:06 | `E4108773D38D1BBB23B39F42BF90822FED8424D626FE52A0D206C4722A77326A` |
| `data/processing/build_final_classifier_dataset_v1.py` | 2026-09-03 21:08:13 | `A3C9F650F16448315EEA834A1975DF84D81CA7B9A692C1FC889F0F55F638A71F` |
| `ai/export/export_classifier_onnx_hailo_v2.py` | 2026-09-03 21:20:30 | `F72F3176A3753659FFBC5F98C7B4CDA0961775A791732EA88AD9E70DF928991E` |
| `ai/export/make_classifier_calib_npy.py` | 2026-09-03 21:22:57 | `95861FFF81E2A3281B9D08BB331CB9851F4A4EA3E25C1103A230F91E79D396A3` |
| `data/extraction/extract_blender_image_sequences_v12.py` | 2026-09-02 04:54:43 | `4D931152193605D335B6077BE52AAB09B8C5827B32C0B26E7837BFDEDC603486` |
| `data/extraction/extract_swim2_real_normal.py` | 2026-08-31 00:51:30 | `55130EF4D74A6E2A8FB1510CFACF62B9B7EF68F7CCDCDE76A0DF37A97709A9C0` |
| `data/validation/test_swim2_three_strokes_cascade_v2.py` | 2026-09-02 07:00:20 | `098A41BBC07EC7E6B2EB62FE8716DEA11EE8562EC0537FBB5D62309EBBB49C4C` |
| `data/validation/test_swim2_person_head_cascade_0039.py` | 2026-09-02 06:55:13 | `B59997C2097921AB3F8703C330B1929457E805A513136A8A6656ED1DB5DA540D` |
| `demo/make_demo_tracking_video.py` | 2026-09-03 20:03:52 | `277A264F81DB0DD08469798A57BCB9FE84C3C1E349B9B4939E3713700F7892F4` |
| `ai/configs/bytetrack_pool.yaml` | 2026-08-24 12:23:42 | `110F518D6E05F32E2B0CB874855BDD9DF26B70ABC642E07F6FA0A4B48D351F15` |
| `docs/presentation/AI_학습_기반_수영장_익수_감지_및_구조지원_시스템.pdf` | 2026-09-12 20:53:36 | `12C75C527174FD8EA60B93C70952619630355905DC6572DCE7E84FCD8256FC6D` |

모델 해시는 [`ai/models/README.md`](../ai/models/README.md)에 별도로 기록합니다. 원본 절대 경로는 개인 PC 구조에 의존하므로 공개 문서에는 넣지 않았습니다.
