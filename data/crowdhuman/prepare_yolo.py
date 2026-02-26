import json
import os
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

CROWDHUMAN_ROOT = "data/crowdhuman"

TRAIN_IMG_DIRS = ["train01/Images", "train02/Images", "train03/Images"]
VAL_IMG_DIRS   = ["val/Images"]

TRAIN_ANN = "data/crowdhuman/annotation_train.odgt"
VAL_ANN   = "data/crowdhuman/annotation_val.odgt"

YOLO_ROOT = "data/crowdhuman_yolo"


def build_image_index(img_dirs, root):
    """Build a dict: filename_stem -> full_path for fast lookup."""
    index = {}
    for d in img_dirs:
        folder = Path(root) / d
        if not folder.exists():
            print(f"WARNING: {folder} does not exist")
            continue
        for p in folder.glob("*.jpg"):
            index[p.stem] = str(p)
        for p in folder.glob("*.jpeg"):
            index[p.stem] = str(p)
    print(f"  Indexed {len(index)} images")
    return index


def convert_odgt_to_yolo(odgt_path, img_index, out_img_dir, out_lbl_dir, split_name):
    """
    Read .odgt, write YOLO label files and copy/link images.
    Uses fbox (full body box) as the ground truth box.
    Skips instances with ignore=1.
    YOLO format: class cx cy w h (normalized 0-1)
    class 0 = person
    """
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    skipped_img = 0
    skipped_ann = 0
    written = 0

    with open(odgt_path, "r") as f:
        lines = f.readlines()

    print(f"  Processing {len(lines)} images for {split_name}...")

    for line in tqdm(lines, desc=split_name):
        record = json.loads(line.strip())
        img_id = record["ID"]  

        img_path = img_index.get(img_id)
        if img_path is None:
            skipped_img += 1
            continue

        try:
            with Image.open(img_path) as img:
                W, H = img.size
        except Exception:
            skipped_img += 1
            continue

        yolo_lines = []
        for gt in record.get("gtboxes", []):
            if gt.get("tag") == "mask":
                continue 
            extra = gt.get("extra", {})
            if extra.get("ignore", 0) == 1:
                skipped_ann += 1
                continue

            fbox = gt.get("fbox")  
            if fbox is None:
                continue

            x, y, w, h = fbox

            x = max(0, x)
            y = max(0, y)
            w = min(w, W - x)
            h = min(h, H - y)
            if w <= 0 or h <= 0:
                continue

            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            nw = w / W
            nh = h / H

            cx = min(max(cx, 0), 1)
            cy = min(max(cy, 0), 1)
            nw = min(max(nw, 0), 1)
            nh = min(max(nh, 0), 1)

            yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        if len(yolo_lines) == 0:
            skipped_img += 1
            continue

        out_img_path = Path(out_img_dir) / f"{img_id}.jpg"
        if not out_img_path.exists():
            shutil.copy2(img_path, out_img_path)

        out_lbl_path = Path(out_lbl_dir) / f"{img_id}.txt"
        with open(out_lbl_path, "w") as f:
            f.write("\n".join(yolo_lines))

        written += 1

    print(f"  Written: {written} | Skipped images: {skipped_img} | Skipped annotations: {skipped_ann}")
    return written


def write_yaml(yolo_root):
    yaml_content = f"""path: {os.path.abspath(yolo_root)}
train: images/train
val: images/val

nc: 1
names: ['person']
"""
    yaml_path = Path(yolo_root) / "crowdhuman.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"\nYAML written to {yaml_path}")
    return str(yaml_path)


if __name__ == "__main__":
    print("Building image index for train...")
    train_idx = build_image_index(TRAIN_IMG_DIRS, CROWDHUMAN_ROOT)

    print("Building image index for val...")
    val_idx = build_image_index(VAL_IMG_DIRS, CROWDHUMAN_ROOT)

    print("\nConverting train annotations...")
    convert_odgt_to_yolo(
        TRAIN_ANN, train_idx,
        out_img_dir=f"{YOLO_ROOT}/images/train",
        out_lbl_dir=f"{YOLO_ROOT}/labels/train",
        split_name="train"
    )

    print("\nConverting val annotations...")
    convert_odgt_to_yolo(
        VAL_ANN, val_idx,
        out_img_dir=f"{YOLO_ROOT}/images/val",
        out_lbl_dir=f"{YOLO_ROOT}/labels/val",
        split_name="val"
    )

    yaml_path = write_yaml(YOLO_ROOT)

    train_imgs = list(Path(f"{YOLO_ROOT}/images/train").glob("*.jpg"))
    val_imgs   = list(Path(f"{YOLO_ROOT}/images/val").glob("*.jpg"))
    train_lbls = list(Path(f"{YOLO_ROOT}/labels/train").glob("*.txt"))
    val_lbls   = list(Path(f"{YOLO_ROOT}/labels/val").glob("*.txt"))

    print(f"\n── Sanity Check ──")
    print(f"Train images : {len(train_imgs)}")
    print(f"Train labels : {len(train_lbls)}")
    print(f"Val images   : {len(val_imgs)}")
    print(f"Val labels   : {len(val_lbls)}")

    assert len(train_imgs) == len(train_lbls), "MISMATCH: train images vs labels count"
    assert len(val_imgs) == len(val_lbls),     "MISMATCH: val images vs labels count"
    assert len(train_imgs) > 10000,            f"Too few train images: {len(train_imgs)}"
    assert len(val_imgs) > 3000,               f"Too few val images: {len(val_imgs)}"

    sample_lbl = train_lbls[0]
    with open(sample_lbl) as f:
        lines = f.readlines()
    print(f"\nSample label ({sample_lbl.name}):")
    for l in lines[:3]:
        print(f"  {l.strip()}")

    print(f"\nCrowdHuman YOLO dataset ready at {YOLO_ROOT}")
    print(f"   YAML: {yaml_path}")