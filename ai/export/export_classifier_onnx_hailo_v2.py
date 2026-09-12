from pathlib import Path

import torch
import torch.nn as nn
import onnx


CKPT_PATH = Path(
    r"D:\임베디드 경진대회\머리 추적"
    r"\익수자 판별 학습"
    r"\FINAL_CLASSIFIER_DATASET_V1"
    r"\TRAINED_HAILO_CLASSIFIER_V1"
    r"\classifier_best.pt"
)

OUT_PATH = CKPT_PATH.parent / "drowning_classifier_50x14_hailo.onnx"

WINDOW_SIZE = 50
NUM_FEATURES = 14
NUM_CLASSES = 3


class HailoTemporalCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                1,
                32,
                kernel_size=(3, 1),
                stride=(1, 1),
                padding=(1, 0),
                bias=True
            ),
            nn.ReLU(),

            nn.Conv2d(
                32,
                32,
                kernel_size=(3, 1),
                stride=(1, 1),
                padding=(1, 0),
                bias=True
            ),
            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=(2, 1),
                stride=(2, 1)
            ),

            nn.Conv2d(
                32,
                64,
                kernel_size=(3, 1),
                stride=(1, 1),
                padding=(1, 0),
                bias=True
            ),
            nn.ReLU(),

            nn.Conv2d(
                64,
                64,
                kernel_size=(3, 1),
                stride=(1, 1),
                padding=(1, 0),
                bias=True
            ),
            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=(2, 1),
                stride=(2, 1)
            ),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                64 * 12 * NUM_FEATURES,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                128,
                NUM_CLASSES
            )
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def main():
    print("=" * 70)
    print("HAILO LEGACY ONNX EXPORT")
    print("=" * 70)

    checkpoint = torch.load(
        CKPT_PATH,
        map_location="cpu"
    )

    model = HailoTemporalCNN()

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    dummy = torch.zeros(
        1,
        1,
        WINDOW_SIZE,
        NUM_FEATURES,
        dtype=torch.float32
    )

    if OUT_PATH.exists():
        OUT_PATH.unlink()

    data_path = Path(
        str(OUT_PATH) + ".data"
    )

    if data_path.exists():
        data_path.unlink()

    print("Exporting...")

    torch.onnx.export(
        model,
        dummy,
        str(OUT_PATH),

        export_params=True,

        opset_version=13,

        do_constant_folding=True,

        input_names=[
            "sequence"
        ],

        output_names=[
            "logits"
        ],

        dynamic_axes=None,

        # 중요
        # 최신 dynamo exporter 대신
        # 예전 TorchScript ONNX exporter 사용
        dynamo=False,

        external_data=False
    )

    print()
    print("Checking ONNX...")

    m = onnx.load(
        str(OUT_PATH)
    )

    onnx.checker.check_model(
        m
    )

    print("ONNX CHECK: OK")

    print(
        "Opset:",
        [x.version for x in m.opset_import]
    )

    print(
        "IR version:",
        m.ir_version
    )

    print(
        "Size MB:",
        round(
            OUT_PATH.stat().st_size
            /
            1024
            /
            1024,
            2
        )
    )

    print(
        "External data:",
        data_path.exists()
    )

    # Conv 노드 kernel_shape 확인
    print()
    print("Conv nodes:")

    for node in m.graph.node:
        if node.op_type == "Conv":

            attrs = {
                a.name: list(a.ints)
                for a in node.attribute
                if len(a.ints) > 0
            }

            print(
                node.name,
                attrs
            )

    print()
    print("=" * 70)
    print("DONE")
    print(OUT_PATH)
    print("=" * 70)


if __name__ == "__main__":
    main()
