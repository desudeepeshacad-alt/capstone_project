# inference.py
import threading

import cv2
import numpy as np
import torch
from PIL import Image

from data.transforms import FaceBBox, face_crop
from models.deepfake_shield import DeepfakeShield

CLIP_LENGTH = 96
FACE_SIZE = 224
FPS = 30.0
FREEZE_FIRST_N_BLOCKS = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_lock = threading.Lock()   # one inference at a time (GPU and detector are not shared-safe)


def load_model(ckpt_path: str) -> DeepfakeShield:
    model = DeepfakeShield(clip_length=CLIP_LENGTH, face_size=FACE_SIZE, fps=FPS,
                           vit_freeze_blocks=FREEZE_FIRST_N_BLOCKS, pretrained=False)
    ck = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ck["model_state_dict"])
    model.eval().to(DEVICE)
    print(f"Loaded {ckpt_path} (epoch {ck.get('epoch', '?')}) on {DEVICE}")
    return model


def _read_frames(path: str, start: int, L: int):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        raise ValueError("Could not open the video.")
    # Skip sequentially; seeking with CAP_PROP_POS_FRAMES is unreliable on some codecs.
    for _ in range(start):
        if not cap.grab():
            break
    frames = []
    while len(frames) < L:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def read_window(path: str, L: int = CLIP_LENGTH):
    """L contiguous RGB frames from the middle of the video, same as eval (center window)."""
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    cap.release()

    start = max(0, (n - L) // 2) if n > 0 else 0
    frames = _read_frames(path, start, L)
    if not frames and start > 0:          # frame count metadata was wrong
        frames = _read_frames(path, 0, L)
    if not frames:
        raise ValueError("Could not decode any frames from the video.")
    while len(frames) < L:                # pad short clips by repeating the last frame
        frames.append(frames[-1])
    return frames


def detect_face(frame: np.ndarray):
    """First Haar detection, matching the training fallback. None if no face."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = _cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    return FaceBBox(int(x), int(y), int(x + w), int(y + h))


def neck_box(bbox: FaceBBox, w: int, h: int):
    """Same geometry as the training pipeline."""
    face_w, face_h = max(bbox.width, 1), max(bbox.height, 1)
    nw = int(face_w * 0.9)
    cx = (bbox.x1 + bbox.x2) // 2
    nx1, nx2 = max(0, cx - nw // 2), min(w, cx + nw // 2)
    ny1 = max(0, bbox.y2 + int(face_h * 0.05))
    ny2 = min(h, ny1 + face_h)
    if ny2 <= ny1 or nx2 <= nx1:
        ny1 = min(h - 2, bbox.y2)
        ny2 = min(h, ny1 + 2)
        nx1, nx2 = max(0, cx - 1), min(w, cx + 1)
    return nx1, ny1, nx2, ny2


def mean_rgb(img: np.ndarray, box) -> np.ndarray:
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = max(x1 + 1, min(x2, img.shape[1])), max(y1 + 1, min(y2, img.shape[0]))
    return img[y1:y2, x1:x2].reshape(-1, 3).mean(axis=0) / 255.0


@torch.no_grad()
def predict_video(model: DeepfakeShield, path: str) -> float:
    """Returns the fake probability in [0, 1]. Raises ValueError for unusable input."""
    frames = read_window(path)
    bbox = detect_face(frames[0])
    if bbox is None:
        raise ValueError("No face detected in the analysed frame.")
    if bbox.width <= 0 or bbox.height <= 0:
        raise ValueError("Invalid face region.")

    # Half-resolution traces with INTER_AREA, matching the evaluation code.
    halves = [cv2.resize(f, (f.shape[1] // 2, f.shape[0] // 2),
                         interpolation=cv2.INTER_AREA) for f in frames]
    h2, w2 = halves[0].shape[:2]
    half = FaceBBox(bbox.x1 // 2, bbox.y1 // 2, bbox.x2 // 2, bbox.y2 // 2)
    face_box = (half.x1, half.y1, half.x2, half.y2)
    nb = neck_box(half, w2, h2)

    face_rgb = np.stack([mean_rgb(f, face_box) for f in halves]).astype(np.float32)
    neck_rgb = np.stack([mean_rgb(f, nb) for f in halves]).astype(np.float32)

    face_img = face_crop(Image.fromarray(frames[0]), bbox, FACE_SIZE, False, None)

    with _lock:
        face_img = face_img.unsqueeze(0).to(DEVICE)
        face_rgb = torch.from_numpy(face_rgb).unsqueeze(0).to(DEVICE)
        neck_rgb = torch.from_numpy(neck_rgb).unsqueeze(0).to(DEVICE)
        out = model(face_img, face_rgb, neck_rgb)
        prob = torch.sigmoid(out["logits"].float().view(-1)).item()
    return prob