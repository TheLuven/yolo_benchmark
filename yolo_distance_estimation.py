import math
from typing import Tuple, Optional

import cv2
from ultralytics import YOLO


CALIBRATION_SCALE = 1.0  # keep at 1.0 for now (no extra calibration)

# -----------------------------
# Camera utilities
# -----------------------------

def get_camera_resolution(cap: cv2.VideoCapture) -> Tuple[int, int]:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return width, height


def get_camera_fov(cap: cv2.VideoCapture):
    fov = cap.get(39)
    if fov > 0:
        return float(fov), float(fov)
    return None, None


def focal_length_pixels_from_fov(image_width: int, fov_deg: float) -> float:
    """Compute focal length in pixels from horizontal FOV and image width.

    f = W / (2 * tan(FOV / 2))
    """
    f = image_width / (2.0 * math.tan(math.radians(fov_deg / 2.0)))
    return f


# -----------------------------
# Distance estimation
# -----------------------------

HUMAN_AVG_HEIGHT_M = 1.70

def estimate_distance_from_bbox(
    bbox_xyxy: Tuple[float, float, float, float],
    image_height: int,
    focal_length_px: float,
    real_height_m: float = HUMAN_AVG_HEIGHT_M,
) -> float:
    """Estimate distance camera -> person using pinhole camera geometry.

    Parameters
    ----------
    bbox_xyxy : (x1, y1, x2, y2) in pixels, in image coordinates
    image_height : image height in pixels (currently unused, kept for future use)
    focal_length_px : focal length in pixels (from FOV or calibration)
    real_height_m : real-world height of the object (meters)

    Returns
    -------
    distance_m : float
        Estimated distance in meters.

    Formula (pinhole model)
    -----------------------
    D = (H_real * f) / h_image
    where:
        - D       : distance camera -> object (meters)
        - H_real  : real height of the object (meters)
        - f       : focal length (pixels)
        - h_image : object height in the image (pixels)
    """
    x1, y1, x2, y2 = bbox_xyxy
    bbox_h_px = max(1.0, y2 - y1)  # avoid division by zero

    distance_m = (real_height_m * focal_length_px) / bbox_h_px
    distance_m *= CALIBRATION_SCALE

    return distance_m


# -----------------------------
# YOLO human detection + distance
# -----------------------------


def run_yolo_distance_estimation(
    model_path: str = "yolo11n.pt",
    source: Optional[str] = None,
    camera_index: int = 0,
    default_fov_deg: float = 78.0,
    person_class_names: Tuple[str, ...] = ("person",),
) -> None:
    model = YOLO(model_path)
    if source is not None:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")
    else:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}")

    img_w, img_h = get_camera_resolution(cap)
    print(f"Input resolution: {img_w}x{img_h}")

    fov_h, fov_v = (None, None)
    if source is None:
        fov_h, fov_v = get_camera_fov(cap)

    if fov_h is None:
        fov_h = default_fov_deg
        print(
            f"Using default horizontal FOV = {fov_h} degrees.\n"
            "Adjust this value in the script if you know your real camera FOV."
        )
    else:
        print(f"Auto-detected camera FOV (horizontal): {fov_h:.2f} deg")

    f_px = focal_length_pixels_from_fov(img_w, fov_h)
    print(f"Estimated focal length: {f_px:.2f} pixels")

    model_names = model.names

    def is_person(cls_id: int) -> bool:
        name = model_names.get(cls_id, str(cls_id)) if isinstance(model_names, dict) else str(cls_id)
        return name.lower() in [n.lower() for n in person_class_names]

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream or failed to read frame")
            break

        results = model.predict(frame, verbose=False)
        if results:
            result = results[0]
            boxes = result.boxes

            for box in boxes:
                cls_id = int(box.cls[0].item())
                score = float(box.conf[0].item())

                if not is_person(cls_id):
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Estimate distance
                distance_m = estimate_distance_from_bbox((x1, y1, x2, y2), img_h, f_px)

                # Draw bbox and distance
                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2,
                )

                label = f"Person {score:.2f}, {distance_m:.2f} m"
                cv2.putText(
                    frame,
                    label,
                    (int(x1), int(max(0, y1 - 10))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        cv2.imshow("YOLO distance", frame)

        # For video, ESC or 'q' to quit early
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Run on the walking video inside footage folder, using default FOV and calibration
    run_yolo_distance_estimation(
        model_path="yolo11n.pt",
        source="footage/walking.webm",
        default_fov_deg=78.0,
    )
