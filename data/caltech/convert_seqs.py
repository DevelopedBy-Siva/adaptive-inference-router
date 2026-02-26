import struct
import os
import cv2
import numpy as np
from pathlib import Path
import argparse

def read_seq_file(seq_path, out_dir, max_frames=None):
    """
    Caltech .seq files are a custom format:
    - 2-byte magic header
    - fixed-size header block
    - then raw JPEG frames, each preceded by a 4-byte length field
    """
    os.makedirs(out_dir, exist_ok=True)
    
    with open(seq_path, 'rb') as f:
        # Read and verify magic bytes
        magic = f.read(4)
        
        # Skip the 28-byte header (Norpix seq format)
        # Header is 1024 bytes total
        f.seek(548)  # offset to first frame length in standard Norpix format
        
        # Try a different approach: scan for JPEG SOI markers (FF D8)
        f.seek(0)
        data = f.read()
    
    # Find all JPEG start markers
    frames = []
    i = 0
    while i < len(data) - 1:
        if data[i] == 0xFF and data[i+1] == 0xD8:
            # Found JPEG start — find its end (FF D9)
            j = i + 2
            while j < len(data) - 1:
                if data[j] == 0xFF and data[j+1] == 0xD9:
                    frames.append(data[i:j+2])
                    i = j + 2
                    break
                j += 1
            else:
                break
        else:
            i += 1
        
        if max_frames and len(frames) >= max_frames:
            break
    
    print(f"  Found {len(frames)} frames in {Path(seq_path).name}")
    
    saved = 0
    for idx, jpg_bytes in enumerate(frames):
        # Decode and verify
        arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            continue
        out_path = os.path.join(out_dir, f"frame_{idx:06d}.jpg")
        cv2.imwrite(out_path, img)
        saved += 1
    
    print(f"  Saved {saved} frames to {out_dir}")
    return saved


def convert_sequences(caltech_root, sequences, out_root, max_frames_per_seq=None):
    """
    sequences: list of (set_name, video_name) tuples
    e.g. [("set00", "V000"), ("set00", "V001")]
    """
    total = 0
    for set_name, vid_name in sequences:
        # Try Train first, then Test
        for split in ["Train", "Test"]:
            seq_path = Path(caltech_root) / split / set_name / set_name / f"{vid_name}.seq"
            if seq_path.exists():
                break
        else:
            print(f"  WARNING: {set_name}/{vid_name}.seq not found in Train or Test")
            continue
        
        out_dir = Path(out_root) / set_name / vid_name
        print(f"\nConverting {set_name}/{vid_name} ({split})...")
        n = read_seq_file(str(seq_path), str(out_dir), max_frames=max_frames_per_seq)
        total += n
    
    print(f"\n✅ Total frames extracted: {total}")
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--caltech_root", default="data/caltech", help="Root of caltech data dir")
    parser.add_argument("--out_root", default="data/caltech/frames", help="Where to save extracted frames")
    parser.add_argument("--max_frames", type=int, default=500, help="Max frames per sequence (None = all)")
    parser.add_argument("--sequences", nargs="+", default=None, help="Sequences to convert, e.g. set00/V000")
    args = parser.parse_args()

    # Default: first 5 sequences of set00 for Day 2 baseline
    if args.sequences is None:
        seqs = [
            ("set00", "V000"),
            ("set00", "V001"),
            ("set00", "V002"),
            ("set00", "V003"),
            ("set00", "V004"),
        ]
    else:
        seqs = [tuple(s.split("/")) for s in args.sequences]

    convert_sequences(args.caltech_root, seqs, args.out_root, max_frames_per_seq=args.max_frames)