#!/usr/bin/env python3
import cv2
import numpy as np
from pathlib import Path

input_video = "greenscreen-asteroid.mp4"
background = "party.jpg"
output_video = "output_video.mp4"

# -------------------- Globals & State --------------------
WIN = "ChromaKey"
TRACK_TOL = "Tolerance (0-100)"
TRACK_SOFT = "Softness (0-50)"

roi_start = None
roi_end = None
roi_ready = False
hsv_mean = None  # (H, S, V) mean of selected patch

writer = None
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or 'avc1' on some systems
DEFAULT_GREEN_HSV = (60.0, 180.0, 180.0)

# -------------------- Helpers --------------------
def compute_mask(frame_bgr, hsv_mean, tol, soft):

    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("Empty frame received in compute_mask")

    # --- normalize channels to BGR ---
    if frame_bgr.ndim == 2:  # GRAY -> BGR
        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
    elif frame_bgr.ndim == 3 and frame_bgr.shape[2] == 4:  # BGRA -> BGR
        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGRA2BGR)
    elif frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"Unexpected frame shape {frame_bgr.shape}, expected (H,W,3)")

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    H0, S0, V0 = hsv_mean

    hue_tol = max(6, int(1.6 * tol))           # 0..160 (OpenCV hue is 0..179)
    sat_tol = max(15, int(0.9 * tol) + 25)
    val_tol = max(15, int(0.8 * tol) + 25)

    H = hsv[:, :, 0].astype(np.int16)
    S = hsv[:, :, 1].astype(np.int16)
    V = hsv[:, :, 2].astype(np.int16)

    dh = np.minimum(np.abs(H - int(H0)), 180 - np.abs(H - int(H0)))
    ds = np.abs(S - int(S0))
    dv = np.abs(V - int(V0))

    mask = (dh <= hue_tol) & (ds <= sat_tol) & (dv <= val_tol)
    mask = (mask.astype(np.uint8) * 255)

    # Clean up and soften edges
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    if soft > 0:
        k = int(soft) * 2 + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)

    return mask

def composite(frame_bgr, bg_bgr, mask):
    """
    - alpha kept 2D (H, W)
    - background resized if needed
    """
    h, w = frame_bgr.shape[:2]
    if bg_bgr.shape[:2] != (h, w):
        bg_bgr = cv2.resize(bg_bgr, (w, h), interpolation=cv2.INTER_LINEAR)

    alpha = (mask.astype(np.float32) / 255.0)        # (H, W)
    alpha3 = alpha[..., None]                        # (H, W, 1)

    out = frame_bgr.astype(np.float32) * (1.0 - alpha3) + bg_bgr.astype(np.float32)  * alpha3
    return out.astype(np.uint8), alpha  # alpha returned 2D for downstream ops

def fit_background(bg_img, size_hw):
    h, w = size_hw
    return cv2.resize(bg_img, (w, h), interpolation=cv2.INTER_LINEAR)

def draw_roi_rect(img, p1, p2, color=(0, 255, 255)):
    if p1 is not None and p2 is not None:
        cv2.rectangle(img, p1, p2, color, 2)

# -------------------- Mouse Callback --------------------
def on_mouse(event, x, y, flags, userdata):
    global roi_start, roi_end, roi_ready, hsv_mean

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_start = (x, y)
        roi_end = (x, y)
        roi_ready = False

    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
        roi_end = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        roi_end = (x, y)
        if roi_start is None or roi_end is None:
            return

        x1, y1 = roi_start
        x2, y2 = roi_end
        if x1 == x2 or y1 == y2:
            return

        frame = userdata["frame"]
        h, w = frame.shape[:2]
        x1, x2 = np.clip([x1, x2], 0, w - 1)
        y1, y2 = np.clip([y1, y2], 0, h - 1)

        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])

        patch = frame[y_min:y_max, x_min:x_max]
        if patch.size == 0:
            return

        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        Hm = np.mean(hsv[:, :, 0])
        Sm = np.mean(hsv[:, :, 1])
        Vm = np.mean(hsv[:, :, 2])
        hsv_mean = (Hm, Sm, Vm)
        roi_ready = True
        print(f"[INFO] Sampled HSV mean: ({Hm:.1f}, {Sm:.1f}, {Vm:.1f}) from ROI {x_min,y_min} -> {x_max,y_max}")

# -------------------- Trackbars --------------------
def _noop(_): pass

def setup_ui(init_tol=35, init_soft=6):
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.createTrackbar(TRACK_TOL, WIN, init_tol, 100, _noop)
    cv2.createTrackbar(TRACK_SOFT, WIN, init_soft, 50, _noop)

def get_trackbar_values():
    tol = cv2.getTrackbarPos(TRACK_TOL, WIN)
    soft = cv2.getTrackbarPos(TRACK_SOFT, WIN)
    return tol, soft

# -------------------- Main --------------------
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    raise SystemExit(f"Cannot open video: {input_video}")

# Read first frame to set up windows & background
ret, frame0 = cap.read()
if not ret:
    raise SystemExit("Failed to read first frame.")

bg_img = cv2.imread(background, cv2.IMREAD_COLOR)
bg_img = fit_background(bg_img, frame0.shape[:2])

setup_ui()

userdata = {"frame": frame0.copy()}
cv2.setMouseCallback(WIN, on_mouse, userdata)

h, w = frame0.shape[:2]

out_path = Path(output_video)
out_dir = out_path.parent
out_dir.mkdir(parents=True, exist_ok=True)

print("Click-drag on the frame to sample green patch")
print("Press S: save and exit")

mask = np.zeros((h, w), np.uint8)

tol_f, soft_f = 35, 6
hsv_f = DEFAULT_GREEN_HSV

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if not ret:
            break
    userdata["frame"] = frame

    tol, soft = get_trackbar_values()

    if roi_ready and hsv_mean is not None:
        mask = compute_mask(frame, hsv_mean, tol, soft)
        bg = bg_img
        comp, alpha = composite(frame, bg, mask)
        display_frame = comp
    else:
        # Draw instructions and any ROI box overlay on the raw frame
        display_frame = frame.copy()
        if roi_start and roi_end:
            draw_roi_rect(display_frame, roi_start, roi_end)
        cv2.putText(display_frame, "Drag a green area to sample.",
                    (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5, cv2.LINE_AA)

    # Small preview of current mask (bottom-right)
    mask_small = cv2.resize(mask, (w // 4, h // 4))
    mask_bgr = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
    display_frame[-mask_bgr.shape[0]:, -mask_bgr.shape[1]:] = mask_bgr

    cv2.imshow(WIN, display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('s'), ord('S')):  # Press S to save and quit
        tol_f, soft_f = get_trackbar_values()
        if roi_ready and hsv_mean is not None:
            hsv_f = hsv_mean
        break

# Save output video
print("[INFO] Start save output video:", output_video)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
while True:
    ret2, f2 = cap.read()
    if not ret2:
        break
    m2 = compute_mask(f2, hsv_f, tol_f, soft_f)
    comp2, _ = composite(f2, bg_img, m2)  
    writer.write(comp2)

writer.release()
print("[INFO] Saved:", output_video)

# Cleanup
cap.release()
cv2.destroyAllWindows()
