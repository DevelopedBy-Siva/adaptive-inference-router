import sys, os
sys.path.insert(0, os.path.abspath("."))

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from src.scheduler import AdaptiveScheduler
from data.caltech.parse_annotations import load_annotations_for_sequences

LIGHT = "runs/detect/models/finetuned/yolov8n_finetuned/weights/best.pt"
HEAVY = "runs/detect/models/finetuned/yolov8l_finetuned/weights/best.pt"

CALTECH_ROOT     = "data/caltech"
FRAMES_ROOT      = "data/caltech/frames"
FULL_TABLE_PATH  = "results/full_comparison.csv"
RESULTS_PATH     = "results/adaptive_metrics.csv"
IOU_THRESH       = 0.5
MAX_FRAMES       = 500

SEQUENCES = [
    ("set00", "V000"),
    ("set00", "V001"),
    ("set00", "V002"),
    ("set00", "V003"),
    ("set00", "V004"),
]

BEST_T = 0.35   


def compute_iou(boxA, boxB):
    ax1,ay1=boxA[0],boxA[1]; ax2,ay2=boxA[0]+boxA[2],boxA[1]+boxA[3]
    bx1,by1=boxB[0],boxB[1]; bx2,by2=boxB[0]+boxB[2],boxB[1]+boxB[3]
    ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    if inter==0: return 0.0
    return inter/((ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter)


def match_detections(gt_boxes, pred_boxes, iou_thresh=0.5):
    if len(gt_boxes)==0: return 0,0,len(pred_boxes)
    if len(pred_boxes)==0: return 0,len(gt_boxes),0
    matched=set(); tp=0
    for gt in gt_boxes:
        best_iou,best_j=0,-1
        for j,pred in enumerate(pred_boxes):
            if j in matched: continue
            iou=compute_iou(gt,pred)
            if iou>best_iou: best_iou,best_j=iou,j
        if best_iou>=iou_thresh: tp+=1; matched.add(best_j)
    return tp,len(gt_boxes)-tp,len(pred_boxes)-len(matched)


def get_frame_paths(set_name, vid_name):
    return sorted((Path(FRAMES_ROOT)/set_name/vid_name).glob("frame_*.jpg"))[:MAX_FRAMES]


if __name__ == "__main__":
    print(f"Running Adaptive Pipeline eval at T={BEST_T}")
    print("Loading annotations...")
    all_annotations = load_annotations_for_sequences(CALTECH_ROOT, SEQUENCES)

    scheduler = AdaptiveScheduler(
        light_weights=LIGHT,
        heavy_weights=HEAVY,
        conf_threshold=BEST_T,
        warmup_frames=10,
    )

    total_frames=total_tp=total_fn=total_fp=0
    occl_tp={0:0,1:0,2:0}; occl_fn={0:0,1:0,2:0}

    for (set_name,vid_name),frame_anns in all_annotations.items():
        frame_paths=get_frame_paths(set_name,vid_name)
        print(f"  {set_name}/{vid_name}: {len(frame_paths)} frames")

        for fp in frame_paths:
            frame_idx=int(fp.stem.split("_")[1])
            gt_anns=frame_anns.get(frame_idx,[])
            gt_boxes=[a["bbox"] for a in gt_anns if a["label"] in ["person","people"]]
            gt_occls=[a["occl"] for a in gt_anns if a["label"] in ["person","people"]]

            frame_bgr=cv2.imread(str(fp))
            detections,_,_,_=scheduler.run_frame(frame_bgr)
            pred_boxes=[d["bbox"] for d in detections]

            tp,fn,fp_c=match_detections(gt_boxes,pred_boxes,IOU_THRESH)
            total_tp+=tp; total_fn+=fn; total_fp+=fp_c; total_frames+=1

            for i,gt_box in enumerate(gt_boxes):
                occl=min(gt_occls[i] if i<len(gt_occls) else 0,2)
                tp_i,fn_i,_=match_detections([gt_box],pred_boxes,IOU_THRESH)
                occl_tp[occl]+=tp_i; occl_fn[occl]+=fn_i

    summary=scheduler.summary()
    recall    = total_tp/(total_tp+total_fn) if (total_tp+total_fn)>0 else 0.0
    precision = total_tp/(total_tp+total_fp) if (total_tp+total_fp)>0 else 0.0
    md_100    = (total_fn/total_frames)*100 if total_frames>0 else 0.0

    def miss_rate(o):
        d=occl_tp[o]+occl_fn[o]
        return occl_fn[o]/d if d>0 else float('nan')

    print(f"\n── Adaptive Pipeline Results (T={BEST_T}) ──")
    print(f"  Frames        : {total_frames}")
    print(f"  Recall        : {recall:.4f}")
    print(f"  Precision     : {precision:.4f}")
    print(f"  MD@100        : {md_100:.2f}")
    print(f"  Effective FPS : {summary['effective_fps']}")
    print(f"  Heavy %       : {summary['pct_heavy']}%")
    print(f"  Miss rate (none)   : {miss_rate(0):.4f}")
    print(f"  Miss rate (partial): {miss_rate(1):.4f}")

    adaptive_row = {
        "model": f"adaptive_T{BEST_T}",
        "frames": total_frames,
        "recall": round(recall,4),
        "precision": round(precision,4),
        "md_100": round(md_100,2),
        "fps": summary["effective_fps"],
        "miss_rate_none": round(miss_rate(0),4),
        "miss_rate_partial": round(miss_rate(1),4),
        "miss_rate_heavy": None,
        "pct_heavy": summary["pct_heavy"],
    }

    pd.DataFrame([adaptive_row]).to_csv(RESULTS_PATH, index=False)

    df_full = pd.read_csv(FULL_TABLE_PATH)
    df_full = df_full[~df_full["model"].str.startswith("adaptive")]
    df_full = pd.concat([df_full, pd.DataFrame([adaptive_row])], ignore_index=True)
    df_full.to_csv(FULL_TABLE_PATH, index=False)

    print(f"\n── Final Comparison Table ──")
    print(df_full[["model","fps","recall","md_100","miss_rate_partial"]].to_string(index=False))

    rows = {r["model"]:r for r in df_full.to_dict("records")}
    l_ft = rows.get("yolov8l_finetuned",{})
    adap = rows.get(f"adaptive_T{BEST_T}",{})
    n_ft = rows.get("yolov8n_finetuned",{})

    print(f"\n── Sanity Checks ──")
    checks_pass = True

    if adap.get("recall",0) >= n_ft.get("recall",0):
        print(f"Adaptive recall >= YOLOv8n finetuned ({adap['recall']} >= {n_ft['recall']})")
    else:
        print(f"Adaptive recall < YOLOv8n finetuned — scheduler is hurting recall")
        checks_pass = False

    if n_ft.get("fps",0) >= adap.get("fps",0) >= l_ft.get("fps",0):
        print(f"Adaptive FPS between light ({n_ft['fps']}) and heavy ({l_ft['fps']}): {adap['fps']}")
    else:
        print(f"Adaptive FPS {adap['fps']} — outside expected range [{l_ft.get('fps')}, {n_ft.get('fps')}]")

    if checks_pass:
        print(f"\nAdaptive pipeline validated. Ready for Day 10: plots + demo video.")
    else:
        print(f"\nIssues found — review before plotting.")