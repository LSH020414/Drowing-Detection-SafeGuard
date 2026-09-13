from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    r"D:\임베디드 경진대회\머리 추적\익수자 판별 학습\FINAL_CLASSIFIER_DATASET_V1"
)

TRAIN_CSV = ROOT / "CLASSIFIER_TRAIN.csv"
VAL_CSV = ROOT / "CLASSIFIER_VAL.csv"
TEST_CSV = ROOT / "CLASSIFIER_TEST.csv"

OUTPUT_DIR = ROOT / "TRAINED_HAILO_CLASSIFIER_V1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

WINDOW_SIZE = 50

FEATURE_COLUMNS = [
    "detected",
    "confidence",
    "cx_norm",
    "cy_norm",
    "head_scale",
    "dx",
    "dy",
    "speed",
    "scale_change",
    "lost_event",
    "redetect_event",
    "consecutive_missing",
    "visible_ratio",
    "missing_ratio",
]

NUM_FEATURES = len(FEATURE_COLUMNS)

CLASS_NAMES = [
    "SWIMMING",
    "FLOATING",
    "ACTIVE",
]

NUM_CLASSES = len(CLASS_NAMES)

BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

PATIENCE = 8

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# SEED
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATASET
# ============================================================

class SequenceDataset(Dataset):

    def __init__(self, csv_path):

        print()
        print("Loading:", csv_path)

        df = pd.read_csv(csv_path)

        self.samples = []

        grouped = df.groupby(
            "window_id",
            sort=False
        )

        skipped = 0

        for window_id, g in grouped:

            if "timestep" in g.columns:
                g = g.sort_values("timestep")
            elif "frame_idx" in g.columns:
                g = g.sort_values("frame_idx")
            elif "time_sec" in g.columns:
                g = g.sort_values("time_sec")

            if len(g) != WINDOW_SIZE:
                skipped += 1
                continue

            x = (
                g[FEATURE_COLUMNS]
                .apply(
                    pd.to_numeric,
                    errors="coerce"
                )
                .replace(
                    [np.inf, -np.inf],
                    np.nan
                )
                .fillna(0.0)
                .to_numpy(
                    dtype=np.float32
                )
            )

            class_id = int(
                g.iloc[0]["class_id"]
            )

            # Hailo-friendly image-like layout
            #
            # [50, 14]
            # -> [1, 50, 14]
            #
            x = np.expand_dims(
                x,
                axis=0
            )

            self.samples.append(
                (
                    torch.tensor(
                        x,
                        dtype=torch.float32
                    ),
                    torch.tensor(
                        class_id,
                        dtype=torch.long
                    ),
                    str(window_id)
                )
            )

        print(
            "usable windows:",
            len(self.samples)
        )

        print(
            "skipped:",
            skipped
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ============================================================
# MODEL
# ============================================================

class HailoTemporalCNN(nn.Module):

    def __init__(self):
        super().__init__()

        # Input:
        # [B, 1, 50, 14]

        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=(3, 1),
                stride=1,
                padding=(1, 0),
                bias=True
            ),

            nn.ReLU(),

            nn.Conv2d(
                32,
                32,
                kernel_size=(3, 1),
                stride=1,
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
                stride=1,
                padding=(1, 0),
                bias=True
            ),

            nn.ReLU(),

            nn.Conv2d(
                64,
                64,
                kernel_size=(3, 1),
                stride=1,
                padding=(1, 0),
                bias=True
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=(2, 1),
                stride=(2, 1)
            ),
        )

        # 50 -> 25 -> 12
        #
        # output:
        # [B, 64, 12, 14]

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                64 * 12 * NUM_FEATURES,
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                p=0.2
            ),

            nn.Linear(
                128,
                NUM_CLASSES
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0

    correct = 0
    total = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():

        for x, y, _ in loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)

            loss = criterion(
                logits,
                y
            )

            total_loss += (
                loss.item()
                *
                x.size(0)
            )

            pred = logits.argmax(
                dim=1
            )

            correct += (
                pred == y
            ).sum().item()

            total += y.size(0)

            all_targets.extend(
                y.cpu().numpy().tolist()
            )

            all_preds.extend(
                pred.cpu().numpy().tolist()
            )

    avg_loss = (
        total_loss / total
        if total
        else 0.0
    )

    accuracy = (
        correct / total
        if total
        else 0.0
    )

    return (
        avg_loss,
        accuracy,
        np.asarray(all_targets),
        np.asarray(all_preds)
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

def confusion_matrix_np(
    y_true,
    y_pred,
    num_classes
):

    cm = np.zeros(
        (
            num_classes,
            num_classes
        ),
        dtype=np.int64
    )

    for t, p in zip(
        y_true,
        y_pred
    ):
        cm[t, p] += 1

    return cm


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    print()
    print("=" * 70)
    print("DEVICE:", DEVICE)
    print("=" * 70)

    train_dataset = SequenceDataset(
        TRAIN_CSV
    )

    val_dataset = SequenceDataset(
        VAL_CSV
    )

    test_dataset = SequenceDataset(
        TEST_CSV
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    class_counts = np.zeros(
        NUM_CLASSES,
        dtype=np.int64
    )

    for _, y, _ in train_dataset.samples:
        class_counts[
            int(y.item())
        ] += 1

    weights = (
        class_counts.sum()
        /
        (
            NUM_CLASSES
            *
            np.maximum(
                class_counts,
                1
            )
        )
    )

    print()
    print(
        "Train class counts:",
        class_counts
    )

    print(
        "Class weights:",
        weights
    )

    class_weights = torch.tensor(
        weights,
        dtype=torch.float32,
        device=DEVICE
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = HailoTemporalCNN().to(
        DEVICE
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # ========================================================
    # TRAIN
    # ========================================================

    best_val_loss = float("inf")

    best_epoch = 0

    patience_count = 0

    history = []

    best_path = (
        OUTPUT_DIR
        /
        "classifier_best.pt"
    )

    print()
    print("=" * 70)
    print("TRAIN START")
    print("=" * 70)

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        running_loss = 0.0

        correct = 0
        total = 0

        for x, y, _ in train_loader:

            x = x.to(
                DEVICE,
                non_blocking=True
            )

            y = y.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad()

            logits = model(x)

            loss = criterion(
                logits,
                y
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                *
                x.size(0)
            )

            pred = logits.argmax(
                dim=1
            )

            correct += (
                pred == y
            ).sum().item()

            total += y.size(0)

        train_loss = (
            running_loss
            /
            total
        )

        train_acc = (
            correct
            /
            total
        )

        val_loss, val_acc, _, _ = (
            evaluate(
                model,
                val_loader,
                criterion
            )
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        print(
            f"Epoch "
            f"{epoch:03d}/{EPOCHS} | "
            f"train loss "
            f"{train_loss:.4f} | "
            f"train acc "
            f"{train_acc:.4f} | "
            f"val loss "
            f"{val_loss:.4f} | "
            f"val acc "
            f"{val_acc:.4f}"
        )

        # ====================================================
        # EARLY STOP
        # ====================================================

        if val_loss < best_val_loss:

            best_val_loss = (
                val_loss
            )

            best_epoch = epoch

            patience_count = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "feature_columns":
                        FEATURE_COLUMNS,

                    "class_names":
                        CLASS_NAMES,

                    "window_size":
                        WINDOW_SIZE,

                    "num_features":
                        NUM_FEATURES,
                },
                best_path
            )

        else:

            patience_count += 1

            if (
                patience_count
                >=
                PATIENCE
            ):

                print()
                print(
                    "Early stopping"
                )

                break

    # ========================================================
    # LOAD BEST
    # ========================================================

    checkpoint = torch.load(
        best_path,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    print()
    print(
        "Best epoch:",
        best_epoch
    )

    # ========================================================
    # TEST
    # ========================================================

    test_loss, test_acc, y_true, y_pred = (
        evaluate(
            model,
            test_loader,
            criterion
        )
    )

    cm = confusion_matrix_np(
        y_true,
        y_pred,
        NUM_CLASSES
    )

    print()
    print("=" * 70)
    print("TEST RESULT")
    print("=" * 70)

    print(
        "Test loss:",
        round(
            test_loss,
            5
        )
    )

    print(
        "Test accuracy:",
        round(
            test_acc,
            5
        )
    )

    print()
    print(
        "Confusion matrix"
    )

    print(
        "rows=true / cols=pred"
    )

    print(
        CLASS_NAMES
    )

    print(
        cm
    )

    # ========================================================
    # SAVE HISTORY
    # ========================================================

    pd.DataFrame(
        history
    ).to_csv(
        OUTPUT_DIR
        /
        "training_history.csv",
        index=False
    )

    pd.DataFrame(
        cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES
    ).to_csv(
        OUTPUT_DIR
        /
        "confusion_matrix.csv"
    )

    # ========================================================
    # ONNX EXPORT
    # ========================================================

    model.eval()

    model_cpu = model.to(
        "cpu"
    )

    dummy = torch.zeros(
        1,
        1,
        WINDOW_SIZE,
        NUM_FEATURES,
        dtype=torch.float32
    )

    onnx_path = (
        OUTPUT_DIR
        /
        "drowning_classifier_50x14.onnx"
    )

    torch.onnx.export(
        model_cpu,
        dummy,
        str(onnx_path),

        input_names=[
            "sequence"
        ],

        output_names=[
            "logits"
        ],

        opset_version=13,

        do_constant_folding=True,

        dynamic_axes=None
    )

    # ========================================================
    # SAVE META
    # ========================================================

    metadata = {
        "classes": CLASS_NAMES,

        "class_map": {
            name: idx
            for idx, name
            in enumerate(
                CLASS_NAMES
            )
        },

        "input_shape": [
            1,
            1,
            WINDOW_SIZE,
            NUM_FEATURES
        ],

        "features":
            FEATURE_COLUMNS,

        "passive_rule":
            "PASSIVE is not classifier output. "
            "Long LOST condition overrides "
            "classifier result on CPU.",

        "best_epoch":
            best_epoch,

        "test_accuracy":
            float(test_acc)
    }

    with open(
        OUTPUT_DIR
        /
        "classifier_metadata.json",

        "w",

        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2
        )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        "PT:",
        best_path
    )

    print(
        "ONNX:",
        onnx_path
    )

    print(
        "Input shape:",
        dummy.shape
    )


if __name__ == "__main__":
    main()
