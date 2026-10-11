# app.py
import logging
import os
import tempfile
import urllib.request

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from inference import load_model, predict_video

# Local checkpoint path. If it doesn't exist, it is downloaded from CKPT_URL.
CKPT = os.environ.get("DEEPFAKE_CKPT", "checkpoints/finetune_resume.pt")
CKPT_URL = os.environ.get(
    "DEEPFAKE_CKPT_URL",
    "https://huggingface.co/DeepeshDesu/deepfake_detection/resolve/main/finetune_resume.pt",
)
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
MAX_BYTES = 100 * 1024 * 1024
CHUNK = 1 << 20
MIN_CKPT_BYTES = 50 * 1024 * 1024        # an HTML error page is far smaller than the checkpoint
ALLOWED_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("deepfake")


def ensure_checkpoint(path: str, url: str) -> str:
    """Use the local checkpoint if present, otherwise download it once from url."""
    if os.path.isfile(path):
        return path
    if not url:
        raise RuntimeError(
            f"Checkpoint not found at '{path}'. Set DEEPFAKE_CKPT or DEEPFAKE_CKPT_URL."
        )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".part"
    log.info("Downloading checkpoint from %s (this can take a few minutes)", url)
    try:
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            while chunk := r.read(CHUNK):
                f.write(chunk)
        if os.path.getsize(tmp) < MIN_CKPT_BYTES:
            raise RuntimeError("Downloaded file is too small; check the URL.")
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    log.info("Checkpoint saved to %s", path)
    return path


app = FastAPI(title="Deepfake Shield")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

MODEL = load_model(ensure_checkpoint(CKPT, CKPT_URL))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in ALLOWED_EXT:
        raise HTTPException(
            400, f"Unsupported format. Use one of: {', '.join(sorted(ALLOWED_EXT))}")

    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        written = 0
        with os.fdopen(fd, "wb") as tmp:
            while chunk := file.file.read(CHUNK):
                written += len(chunk)
                if written > MAX_BYTES:
                    raise HTTPException(
                        413, f"File larger than {MAX_BYTES // (1024 * 1024)} MB.")
                tmp.write(chunk)
        if written == 0:
            raise HTTPException(400, "Empty file.")

        try:
            p = predict_video(MODEL, path)
        except ValueError as e:
            raise HTTPException(422, str(e))
        except Exception:
            log.exception("prediction failed")
            raise HTTPException(500, "Internal error while analysing the video.")
    finally:
        if os.path.exists(path):
            os.remove(path)

    return {
        "fake_probability": round(p, 4),
        "label": "fake" if p > 0.5 else "real",
        "threshold": 0.5,
        "note": "Model output, not a verdict. Accuracy drops on re-compressed video.",
    }