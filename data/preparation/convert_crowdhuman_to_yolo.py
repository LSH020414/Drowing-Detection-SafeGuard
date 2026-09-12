import os
import json
import shutil
from pathlib import Path
from PIL import Image

ROOT = Path(r"D:\HeadDetector\datasets\CrowdHuman")
OUT = Path(r"D:\HeadDetector\yolo_crowdhuman")

TRAIN_IMAGE_DIRS = [
    ROOT / "train01" / "Images",
    ROOT / "train02" / "Images",
    ROOT / "train03" / "Images",
]

VAL_IMAGE_DIR = ROOT / "val" / "Images"

TRAIN_ANN = ROOT / "annotation_train.odgt"
VAL_ANN = ROOT / "annotation_val.odgt"


def build_image_index(image_dirs):
    index = {}

    for image_dir in image_dirs:
        for image_path in image_dir.glob("*.jpg"):
            index[image_path.stem] = image_path

    return index


def convert_split(annotation_file, image_index, split):
    out_images = OUT / "images" / split
    out_labels = OUT / "labels" / split

    count_images = 0
    count_boxes = 0
    missing_images = 0

    with open(annotation_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)

            image_id = data["ID"]

            if image_id not in image_index:
                missing_images += 1
                continue

            src_image = image_index[image_id]

            try:
                with Image.open(src_image) as img:
                    img_width, img_height = img.size
            except Exception as e:
                print(f"[ERROR] 이미지 읽기 실패: {src_image} / {e}")
                continue

            yolo_labels = []

            for obj in data.get("gtboxes", []):
                if "hbox" not in obj:
                    continue

                # ignore annotation 처리
                extra = obj.get("extra", {})
                if extra.get("ignore", 0) == 1:
                    continue

                x, y, w, h = obj["hbox"]

                if w <= 0 or h <= 0:
                    continue

                # 이미지 경계를 벗어나는 박스 보정
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(img_width, x + w)
                y2 = min(img_height, y + h)

                box_w = x2 - x1
                box_h = y2 - y1

                if box_w <= 1 or box_h <= 1:
                    continue

                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                norm_w = box_w / img_width
                norm_h = box_h / img_height

                yolo_labels.append(
                    f"0 {x_center:.6f} {y_center:.6f} "
                    f"{norm_w:.6f} {norm_h:.6f}"
                )

            # 이미지 복사
            dst_image = out_images / src_image.name
            shutil.copy2(src_image, dst_image)

            # 라벨 저장
            label_path = out_labels / f"{image_id}.txt"

            with open(label_path, "w", encoding="utf-8") as lf:
                lf.write("\n".join(yolo_labels))

            count_images += 1
            count_boxes += len(yolo_labels)

            if count_images % 1000 == 0:
                print(
                    f"[{split}] {count_images} images / "
                    f"{count_boxes} head boxes"
                )

    print()
    print(f"=== {split.upper()} 완료 ===")
    print(f"Images: {count_images}")
    print(f"Head boxes: {count_boxes}")
    print(f"Missing images: {missing_images}")


def main():
    print("CrowdHuman image index 생성 중...")

    train_index = build_image_index(TRAIN_IMAGE_DIRS)
    val_index = build_image_index([VAL_IMAGE_DIR])

    print(f"Train images found: {len(train_index)}")
    print(f"Val images found: {len(val_index)}")

    print("\nTrain 변환 시작...")
    convert_split(
        TRAIN_ANN,
        train_index,
        "train"
    )

    print("\nValidation 변환 시작...")
    convert_split(
        VAL_ANN,
        val_index,
        "val"
    )


if __name__ == "__main__":
    main()