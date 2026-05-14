import base64
import io
import os
import tempfile

import cv2
import numpy as np
import torch
import torch.nn as nn
from fastapi import BackgroundTasks, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import hf_hub_download
from PIL import Image
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms
from torchvision.models import efficientnet_b3

MODEL_REPO = "samhitak10/fetal-echo-quality-classifier"

GUIDANCE = {
    "good":         "Image quality is good. Proceed with assessment.",
    "blurry":       "Press the probe firmly against the skin and apply more ultrasound gel. Ask the patient to hold their breath.",
    "too_dark":     "Increase the gain setting on the ultrasound machine, or press the probe more firmly.",
    "low_contrast": "Re-position the probe — try a slightly different angle. Ensure the patient has a full bladder.",
    "noisy":        "Wipe off old gel, apply fresh gel, and reduce probe pressure slightly. Ask the patient to remain still.",
    "angled":       "Rotate the probe slightly until the cardiac structure appears centred on screen.",
}

# BGR colours for video overlay
COLORS_BGR = {
    "good":         (105, 185, 16),
    "blurry":       (11, 158, 245),
    "too_dark":     (235, 102, 99),
    "low_contrast": (235, 184, 59),
    "noisy":        (185, 73, 236),
    "angled":       (176, 184, 20),
}


def load_model():
    path  = hf_hub_download(repo_id=MODEL_REPO, filename="ultrasound_efficientnet_b3.pth")
    ckpt  = torch.load(path, map_location="cpu", weights_only=False)
    names = ckpt["class_names"]
    m     = efficientnet_b3(weights=None)
    in_f  = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(p=0.3, inplace=False), nn.Linear(in_f, len(names)))
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    return m, names


model, CLASS_NAMES = load_model()

val_tf = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# EigenCAM: PCA-based, no gradient issues, no corner bias
cam_extractor = EigenCAM(model=model, target_layers=[model.features[-2]])

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def run_inference(pil_img: Image.Image):
    tensor = val_tf(pil_img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze().numpy()
    pred_idx   = int(probs.argmax())
    pred_label = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    cam_map  = cam_extractor(input_tensor=tensor)[0]
    orig_np  = np.array(pil_img.resize((300, 300)), dtype=np.float32) / 255.0
    cam_img  = show_cam_on_image(orig_np, cam_map, use_rgb=True)

    return pred_label, confidence, probs, cam_img


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        data = await file.read()
        orig = Image.open(io.BytesIO(data)).convert("RGB")

        pred_label, confidence, probs, cam_img = run_inference(orig)

        # Encode cam + original as base64
        _, buf   = cv2.imencode(".jpg", cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR))
        cam_b64  = base64.b64encode(buf).decode()

        orig_buf = io.BytesIO()
        orig.resize((300, 300)).save(orig_buf, format="JPEG")
        orig_b64 = base64.b64encode(orig_buf.getvalue()).decode()

        return {
            "label":      pred_label,
            "confidence": round(confidence, 4),
            "guidance":   GUIDANCE.get(pred_label, ""),
            "probs":      {c: round(float(p), 4) for c, p in zip(CLASS_NAMES, probs)},
            "cam_image":  cam_b64,
            "orig_image": orig_b64,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/predict-video")
async def predict_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        # Write uploaded video to temp file
        suffix  = os.path.splitext(file.filename or ".mp4")[1] or ".mp4"
        tmp_in  = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp_in.write(await file.read())
        tmp_in.close()

        out_path = tmp_in.name.replace(suffix, "_analyzed.mp4")

        cap  = cv2.VideoCapture(tmp_in.name)
        fps  = cap.get(cv2.CAP_PROP_FPS) or 25
        w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out    = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        frame_idx   = 0
        last_label  = None
        last_conf   = None
        last_cam    = None
        STEP        = 5   # run inference every 5 frames

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % STEP == 0:
                rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil  = Image.fromarray(rgb)
                last_label, last_conf, _, cam_rgb = run_inference(pil)
                cam_full = cv2.resize(cam_rgb, (w, h))
                last_cam = cv2.cvtColor(cam_full, cv2.COLOR_RGB2BGR)

            if last_cam is not None:
                frame = cv2.addWeighted(frame, 0.55, last_cam, 0.45, 0)

            if last_label is not None:
                color = COLORS_BGR.get(last_label, (200, 200, 200))
                label_text = f"{last_label.replace('_',' ').upper()}  {last_conf:.0%}"
                # Dark pill background
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (12, 12), (24 + tw, 20 + th + 8), (0, 0, 0), -1)
                cv2.putText(frame, label_text, (18, 12 + th + 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()

        background_tasks.add_task(os.unlink, tmp_in.name)
        background_tasks.add_task(os.unlink, out_path)

        return FileResponse(
            out_path,
            media_type="video/mp4",
            filename="analyzed.mp4",
            background=background_tasks,
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


app.mount("/", StaticFiles(directory="static", html=True), name="static")
