# data/caltech/parse_annotations.py
# .vbb files are MATLAB binary files. We parse them manually.
import struct
import numpy as np
import json
import os
from pathlib import Path


def parse_vbb(vbb_path):
    """
    Parse Caltech .vbb annotation file.
    Returns dict: {frame_idx: [{"bbox": [x,y,w,h], "occl": int, "label": str}]}
    occl: 0=none, 1=partial, 2=heavy
    """
    # .vbb is a MATLAB v5 .mat file — use scipy if available, else fall back
    try:
        import scipy.io as sio
        mat = sio.loadmat(vbb_path, squeeze_me=True, struct_as_record=False)
        A = mat['A']
        
        frame_annotations = {}
        n_frames = int(A.nFrame)
        obj_lists = A.objLists
        obj_lbl = A.objLbl  # label names array
        
        for frame_idx in range(n_frames):
            objs = obj_lists[frame_idx]
            if objs is None or (hasattr(objs, '__len__') and len(objs) == 0):
                frame_annotations[frame_idx] = []
                continue
            
            # Handle single object (not array)
            if not hasattr(objs, '__len__'):
                objs = [objs]
            
            anns = []
            for obj in objs:
                try:
                    lbl = str(obj.lbl) if hasattr(obj, 'lbl') else 'person'
                    if lbl not in ['person', 'people']:
                        continue
                    
                    bb = obj.pos  # [x, y, w, h]
                    occl = int(obj.occl) if hasattr(obj, 'occl') else 0
                    
                    if hasattr(bb, '__len__') and len(bb) == 4:
                        anns.append({
                            "bbox": [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])],
                            "occl": occl,
                            "label": lbl
                        })
                except Exception as e:
                    continue
            
            frame_annotations[frame_idx] = anns
        
        return frame_annotations
    
    except ImportError:
        print("scipy not found. Install it: pip install scipy")
        raise


def load_annotations_for_sequences(caltech_root, sequences):
    """
    Load annotations for a list of (set_name, vid_name) sequences.
    Returns: {(set_name, vid_name): {frame_idx: [ann]}}
    """
    all_anns = {}
    ann_root = Path(caltech_root) / "annotations"
    
    for set_name, vid_name in sequences:
        vbb_path = ann_root / set_name / f"{vid_name}.vbb"
        if not vbb_path.exists():
            print(f"WARNING: {vbb_path} not found")
            continue
        print(f"Parsing {set_name}/{vid_name}.vbb ...")
        anns = parse_vbb(str(vbb_path))
        all_anns[(set_name, vid_name)] = anns
        total_instances = sum(len(v) for v in anns.values())
        print(f"  {len(anns)} frames, {total_instances} total pedestrian instances")
    
    return all_anns


if __name__ == "__main__":
    seqs = [
        ("set00", "V000"),
        ("set00", "V001"),
        ("set00", "V002"),
        ("set00", "V003"),
        ("set00", "V004"),
    ]
    
    anns = load_annotations_for_sequences("data/caltech", seqs)
    
    # Quick sanity check — print a few frames
    for (s, v), frames in anns.items():
        non_empty = {f: a for f, a in frames.items() if len(a) > 0}
        print(f"\n{s}/{v}: {len(non_empty)} frames with pedestrians")
        # Print first 3 annotated frames
        for i, (fi, fa) in enumerate(list(non_empty.items())[:3]):
            print(f"  Frame {fi}: {len(fa)} pedestrian(s)")
            for ann in fa[:2]:
                print(f"    bbox={ann['bbox']}, occl={ann['occl']}")
        break  # just show first sequence

    print("\nAnnotation parsing works.")