#!/usr/bin/env python3
"""
YOLOv12 (Ultralytics) inference using OpenVINO Runtime directly.
- If an OpenVINO IR (.xml/.bin) for the given weights doesn't exist, auto-export via Ultralytics. If IR export fails, falls back to ONNX export and uses OpenVINO to run the ONNX.
- Reads an input video, performs detection, draws annotations, and writes an output video.
- Robust post-processing for typical Ultralytics OpenVINO/ONNX outputs (1xN x (4+nc) or 1x(4+nc) x N, with or without objectness column).

Usage examples:
  python yolov12_openvino_video.py --input footage/5.mp4 --weights yolo12n.pt --show
  python yolov12_openvino_video.py --input footage/walking.mp4 --ir yolov8n_openvino_model/yolov8n.xml --conf 0.25
  python yolov12_openvino_video.py --input footage/walking.mp4 --ir yolo12n.onnx --conf 0.25

Notes:
- Designed to mirror the structure and UX of mediapipe_on_openvino.py.
- Default model is YOLOv12n (COCO 80 classes). If metadata.yaml is found, class names will be read from it.
"""
import os
import cv2
import time
import json
import argparse
import numpy as np
from datetime import datetime
from typing import List, Tuple, Optional

from openvino.runtime import Core

# Optional: used for auto-export if IR is missing
try:
    from ultralytics import YOLO  # only used to export to OpenVINO if needed
    _ULTRALYTICS_AVAILABLE = True
except Exception:
    _ULTRALYTICS_AVAILABLE = False

# Default COCO80 class names as fallback if metadata.yaml not found
COCO80 = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
    'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
    'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
    'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
    'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange',
    'broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed',
    'dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven',
    'toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
]


def letterbox(im: np.ndarray, new_shape: Tuple[int, int], color=(114, 114, 114)):
    """Resize and pad image to meet stride-multiple constraints while keeping aspect ratio.
    Returns: img, ratio, (dw, dh)
    """
    shape = im.shape[:2]  # current shape [h, w]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    h, w = shape
    new_h, new_w = new_shape

    # Scale ratio (new / old)
    r = min(new_h / h, new_w / w)
    nh, nw = int(round(h * r)), int(round(w * r))

    # Compute padding
    dh, dw = new_h - nh, new_w - nw
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2

    # Resize
    if (h, w) != (nh, nw):
        im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)

    # Pad
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return im, r, (left, top)


def load_class_names_from_metadata(metadata_path: str) -> Optional[List[str]]:
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            import yaml
            data = yaml.safe_load(f)
            names = data.get('names', None)
            if isinstance(names, dict):
                # Convert dict of {id: name} to list
                max_id = max(int(k) for k in names.keys())
                arr: List[str] = [''] * (max_id + 1)
                for k, v in names.items():
                    arr[int(k)] = str(v)
                return arr
            elif isinstance(names, list):
                return [str(x) for x in names]
    except Exception:
        pass
    return None


def ensure_ir_exported(weights: str) -> Tuple[str, List[str]]:
    """Ensure a model readable by OpenVINO exists for given weights; prefer OpenVINO IR, fallback to ONNX.
    Returns (model_path_xml_or_onnx, class_names)
    """
    stem = os.path.splitext(os.path.basename(weights))[0]
    ir_dir = os.path.join(os.getcwd(), f"{stem}_openvino_model")
    xml_path = os.path.join(ir_dir, f"{stem}.xml")
    bin_path = os.path.join(ir_dir, f"{stem}.bin")

    # If IR exists, use it
    if os.path.exists(xml_path) and os.path.exists(bin_path):
        metadata_path = os.path.join(os.path.dirname(xml_path), 'metadata.yaml')
        names = load_class_names_from_metadata(metadata_path) or COCO80
        return xml_path, names

    # Else we need to export something
    if not _ULTRALYTICS_AVAILABLE:
        raise RuntimeError("Aucun modèle lisible par OpenVINO trouvé et Ultralytics n'est pas disponible pour exporter.")

    print(f"[Export] Tentative d'export IR OpenVINO pour {weights}...")
    model = YOLO(weights)
    try:
        model.export(format="openvino", dynamic=True, half=False)
        if os.path.exists(xml_path) and os.path.exists(bin_path):
            metadata_path = os.path.join(os.path.dirname(xml_path), 'metadata.yaml')
            names = load_class_names_from_metadata(metadata_path) or COCO80
            return xml_path, names
        # Try to locate generated folder by scanning working dir
        for name in os.listdir(os.getcwd()):
            if name.endswith('_openvino_model') and os.path.isdir(name):
                candidate = os.path.join(os.getcwd(), name, f"{name.replace('_openvino_model','')}.xml")
                if os.path.exists(candidate):
                    xml_path = candidate
                    metadata_path = os.path.join(os.path.dirname(xml_path), 'metadata.yaml')
                    names = load_class_names_from_metadata(metadata_path) or COCO80
                    return xml_path, names
        print("[Export] IR introuvable après export. On va tenter ONNX...")
    except Exception as e:
        print(f"[Export] Échec de l'export IR OpenVINO: {e}\n[Export] On tente ONNX...")

    # Fallback: export ONNX and use OpenVINO to run it
    onnx_path = os.path.join(os.getcwd(), f"{stem}.onnx")
    try:
        model.export(format="onnx", dynamic=True, opset=12)
    except Exception as e:
        raise RuntimeError(f"Échec de l'export ONNX pour {weights}: {e}")

    if not os.path.exists(onnx_path):
        # Try common alt filename (Ultralytics sometimes writes to weights stem)
        # Or scan for any .onnx in CWD matching stem
        candidates = [f for f in os.listdir(os.getcwd()) if f.startswith(stem) and f.endswith('.onnx')]
        if candidates:
            onnx_path = os.path.join(os.getcwd(), candidates[0])
        else:
            raise FileNotFoundError("Export ONNX terminé mais fichier .onnx introuvable.")

    print(f"[Export] Utilisation du modèle ONNX: {onnx_path}")
    return onnx_path, COCO80


def load_ir_model(xml_path: str, device: str = 'CPU'):
    print(f"Chargement du modèle OpenVINO: {xml_path}")
    core = Core()
    print(f"Périphériques disponibles: {core.available_devices}")
    # Supports .xml (IR) and .onnx
    model = core.read_model(model=xml_path)
    compiled = core.compile_model(model=model, device_name=device)

    inp = compiled.input(0)
    try:
        # Avoid converting dynamic PartialShape to static
        pshape = inp.get_partial_shape()
        print(f"Entrée modèle (PartialShape): {pshape}")
    except Exception:
        print("Entrée modèle: (information de forme non disponible)")

    out = compiled.output(0)
    try:
        pshape_out = out.get_partial_shape()
        print(f"Sortie modèle (PartialShape): {pshape_out}")
    except Exception:
        print("Sortie modèle: (information de forme non disponible)")

    return compiled


def xywh2xyxy(x):
    # Convert [cx, cy, w, h] -> [x1, y1, x2, y2]
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
    return y


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
    """Simple NMS returning indices to keep.
    boxes: [N, 4] in xyxy
    scores: [N]
    """
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        ious = bbox_iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_thres]
    return keep


def bbox_iou(box1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    # box1: [4], boxes2: [M,4]
    xA = np.maximum(box1[0], boxes2[:, 0])
    yA = np.maximum(box1[1], boxes2[:, 1])
    xB = np.minimum(box1[2], boxes2[:, 2])
    yB = np.minimum(box1[3], boxes2[:, 3])

    interW = np.maximum(0, xB - xA)
    interH = np.maximum(0, yB - yA)
    inter = interW * interH

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union = area1 + area2 - inter + 1e-6
    return inter / union


def postprocess(pred: np.ndarray, img_shape: Tuple[int, int], model_shape: Tuple[int, int],
                pad: Tuple[int, int], conf_thres: float, iou_thres: float, names: List[str],
                classes: Optional[List[int]] = None, max_det: int = 300):
    """Post-process raw model output to boxes, scores, class_ids in original image scale.
    - Supports outputs shaped as (1, N, no) or (1, no, N)
    - Supports no = 4 + nc (YOLOv8/10/11/12 style) and no = 5 + nc (objectness + classes)
    """
    if pred.ndim == 3:
        bs, a, b = pred.shape
        # Arrange to [N, no]
        if a < b:  # (1, no, N)
            pred = np.transpose(pred, (0, 2, 1))  # -> (1, N, no)
    elif pred.ndim == 2:
        pred = pred[None, ...]  # (1, N)
    else:
        raise RuntimeError(f"Format de sortie non supporté: shape={pred.shape}")

    pred = pred[0]  # (N, no)
    if pred.size == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)

    N, no = pred.shape

    # Split components
    boxes_xywh = pred[:, :4]
    if no <= 5:
        raise RuntimeError(f"Sortie inattendue, no={no}. Attendu >= 4+nc.")

    # Determine presence of objectness
    # Case A: 4 + nc -> cls scores in columns 4:
    # Case B: 5 + nc -> obj at 4, classes at 5:
    if no - 4 in (80, 81, 20, 1) or (no - 4) > 5:  # likely 4+nc
        cls_scores = pred[:, 4:]
        obj = None
    else:
        obj = pred[:, 4]
        cls_scores = pred[:, 5:]

    # Compute per-detection conf & class
    if obj is None:
        class_ids = np.argmax(cls_scores, axis=1)
        confs = cls_scores[np.arange(N), class_ids]
    else:
        class_ids = np.argmax(cls_scores, axis=1)
        confs = obj * cls_scores[np.arange(N), class_ids]

    # Filter by conf and classes
    mask = confs >= conf_thres
    if classes is not None and len(classes) > 0:
        cls_mask = np.isin(class_ids, np.array(classes, dtype=np.int32))
        mask = mask & cls_mask

    boxes_xywh = boxes_xywh[mask]
    confs = confs[mask]
    class_ids = class_ids[mask]

    if boxes_xywh.size == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)

    # Convert to xyxy in model space
    boxes_xyxy = xywh2xyxy(boxes_xywh.copy())

    # Undo letterbox to original image coords
    ih, iw = img_shape  # original image h,w
    mh, mw = model_shape  # model input h,w
    padw, padh = pad
    gain = min(mw / iw, mh / ih)

    boxes_xyxy[:, [0, 2]] -= padw
    boxes_xyxy[:, [1, 3]] -= padh
    boxes_xyxy[:, :4] /= gain

    # Clip
    boxes_xyxy[:, 0] = boxes_xyxy[:, 0].clip(0, iw - 1)
    boxes_xyxy[:, 1] = boxes_xyxy[:, 1].clip(0, ih - 1)
    boxes_xyxy[:, 2] = boxes_xyxy[:, 2].clip(0, iw - 1)
    boxes_xyxy[:, 3] = boxes_xyxy[:, 3].clip(0, ih - 1)

    # NMS per class (classic approach: offset boxes by class id)
    # To keep it simple, we'll do a class-agnostic NMS per class bucket
    final_boxes = []
    final_scores = []
    final_classes = []

    for c in np.unique(class_ids):
        inds = np.where(class_ids == c)[0]
        b = boxes_xyxy[inds]
        s = confs[inds]
        keep = nms(b, s, iou_thres)
        keep = keep[:max_det]
        final_boxes.append(b[keep])
        final_scores.append(s[keep])
        final_classes.append(np.full(len(keep), c, dtype=np.int32))

    if len(final_boxes) == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)

    boxes = np.concatenate(final_boxes, axis=0)
    scores = np.concatenate(final_scores, axis=0)
    classes_out = np.concatenate(final_classes, axis=0)

    # Sort by confidence desc
    order = scores.argsort()[::-1]
    boxes, scores, classes_out = boxes[order], scores[order], classes_out[order]
    return boxes, scores, classes_out


def draw_detections(frame: np.ndarray, boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, names: List[str]):
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].astype(int)
        cls = int(classes[i])
        conf = float(scores[i])
        label = names[cls] if 0 <= cls < len(names) else f"{cls}"
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return frame


def find_default_video() -> Optional[str]:
    # Prefer 'footage/5.mp4' if exists; else pick any from footage
    candidates = [
        os.path.join('footage', '5.mp4'),
        os.path.join('footage', 'walking.mp4'),
        os.path.join('footage', '1.mp4'),
        os.path.join('footage', '2.mp4'),
        os.path.join('footage', '3.mp4'),
        os.path.join('footage', '4.mp4'),
        os.path.join('footage', '6.mp4'),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def infer_video(input_path: str,
                weights: Optional[str] = 'yolo12n.pt',
                ir_xml: Optional[str] = None,
                output_path: Optional[str] = None,
                device: str = 'CPU',
                conf: float = 0.25,
                iou: float = 0.5,
                imgsz: Tuple[int, int] = (640, 640),
                classes: Optional[List[int]] = None,
                show: bool = False,
                max_frames: Optional[int] = None):
    # Prepare model
    if ir_xml:
        xml_path = ir_xml
        # Try to read names from sibling metadata.yaml
        names = load_class_names_from_metadata(os.path.join(os.path.dirname(xml_path), 'metadata.yaml')) or COCO80
    else:
        if weights is None:
            raise ValueError("Spécifiez --weights ou --ir")
        xml_path, names = ensure_ir_exported(weights)

    compiled = load_ir_model(xml_path, device=device)

    # Determine model input size
    model_h, model_w = int(imgsz[0]), int(imgsz[1])
    try:
        pshape = compiled.input(0).get_partial_shape()
        if not pshape.is_dynamic:
            shape = list(pshape.to_shape())  # [N,C,H,W]
            if len(shape) == 4:
                model_h, model_w = int(shape[2]), int(shape[3])
    except Exception:
        # keep imgsz fallback
        pass

    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la vidéo: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output path
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join('runs', 'detect', f'ov_yolov12_{timestamp}')
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, 'annotated.mp4')
    else:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Stats
    frame_idx = 0
    inference_times = []
    total_times = []
    detections_total = 0

    print("\nDémarrage du traitement vidéo...")
    print(f"Entrée: {input_path}")
    print(f"Sortie: {output_path}")

    try:
        while True:
            t_frame0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess: letterbox to model size, BGR->RGB, NCHW, float32 [0,1]
            img_lb, ratio, (padw, padh) = letterbox(frame, (model_h, model_w))
            img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
            inp = img_rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

            # Inference
            t0 = time.time()
            result = compiled([inp])
            inf_ms = (time.time() - t0) * 1000
            inference_times.append(inf_ms)

            # Extract first output
            ov_outs = [result[o] for o in compiled.outputs]
            pred = ov_outs[0]

            # Postprocess
            boxes, scores, cls_ids = postprocess(
                pred=pred,
                img_shape=(frame.shape[0], frame.shape[1]),
                model_shape=(model_h, model_w),
                pad=(padw, padh),
                conf_thres=conf,
                iou_thres=iou,
                names=names,
                classes=classes,
                max_det=300,
            )

            detections_total += len(boxes)

            # Draw
            annotated = draw_detections(frame.copy(), boxes, scores, cls_ids, names)
            cv2.putText(annotated, f"Frame {frame_idx}/{total_frames}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(annotated, f"Detections {len(boxes)} | Inference {inf_ms:.1f}ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            out.write(annotated)

            if show:
                cv2.imshow('YOLOv12 OpenVINO', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            total_ms = (time.time() - t_frame0) * 1000
            total_times.append(total_ms)
            frame_idx += 1

            if max_frames is not None and frame_idx >= max_frames:
                break

    finally:
        cap.release()
        out.release()
        if show:
            cv2.destroyAllWindows()

    # Save stats
    avg_inf = float(np.mean(inference_times)) if inference_times else 0.0
    avg_total = float(np.mean(total_times)) if total_times else 0.0
    stats = {
        'video': {
            'path': os.path.abspath(input_path),
            'resolution': f"{width}x{height}",
            'fps': fps,
            'total_frames': total_frames,
        },
        'processing': {
            'frames_processed': frame_idx,
            'detections_total': detections_total,
            'elapsed_s': round(sum(total_times)/1000.0, 3) if total_times else 0.0,
            'effective_fps': round(frame_idx / max(1e-6, sum(total_times)/1000.0), 2) if total_times else 0.0,
        },
        'performance': {
            'avg_inference_ms': round(avg_inf, 2),
            'avg_total_ms': round(avg_total, 2),
            'min_inference_ms': round(float(np.min(inference_times)), 2) if inference_times else 0.0,
            'max_inference_ms': round(float(np.max(inference_times)), 2) if inference_times else 0.0,
        },
        'output': {
            'video': os.path.abspath(output_path),
        }
    }
    stats_path = output_path.replace('.mp4', '_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    print("\nTraitement terminé.")
    print(f"Frames: {frame_idx} | Detections: {detections_total} | Avg inf: {avg_inf:.2f}ms | Avg total: {avg_total:.2f}ms")
    print(f"Vidéo de sortie: {output_path}")
    print(f"Statistiques: {stats_path}")



def main():
    parser = argparse.ArgumentParser(description='YOLOv12 avec OpenVINO sur une vidéo (détection).')
    parser.add_argument('--input', type=str, default=None, help='Chemin vidéo en entrée (ex: footage/5.mp4)')
    parser.add_argument('--weights', type=str, default='yolo12n.pt', help='Poids Ultralytics (ex: yolo12n.pt)')
    parser.add_argument('--ir', type=str, default=None, help='Chemin du modèle OpenVINO .xml (si fourni, ignore --weights)')
    parser.add_argument('--output', type=str, default=None, help='Chemin vidéo de sortie')
    parser.add_argument('--device', type=str, default='CPU', help='Périphérique OpenVINO (CPU, GPU, AUTO, etc)')
    parser.add_argument('--conf', type=float, default=0.25, help='Seuil de confiance')
    parser.add_argument('--iou', type=float, default=0.50, help='Seuil IoU NMS')
    parser.add_argument('--imgsz', type=int, nargs=2, default=[640, 640], help='Taille entrée modèle (h, w) si dynamique')
    parser.add_argument('--classes', type=int, nargs='*', default=None, help='Limiter aux classes (ex: 0 2 5)')
    parser.add_argument('--show', action='store_true', help='Afficher un aperçu')
    parser.add_argument('--max-frames', type=int, default=None, help='Limiter le nombre de frames pour un test rapide')

    args = parser.parse_args()

    input_path = args.input or find_default_video()
    if not input_path or not os.path.exists(input_path):
        raise FileNotFoundError("Aucune vidéo d'entrée trouvée (essayez --input footage/walking.mp4)")

    infer_video(
        input_path=input_path,
        weights=args.weights,
        ir_xml=args.ir,
        output_path=args.output,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        imgsz=(int(args.imgsz[0]), int(args.imgsz[1])),
        classes=args.classes,
        show=args.show,
        max_frames=args.max_frames,
    )


if __name__ == '__main__':
    main()
