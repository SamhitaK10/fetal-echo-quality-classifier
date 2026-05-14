import gradio as gr
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from huggingface_hub import hf_hub_download
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

MODEL_REPO = "samhitak10/fetal-echo-quality-classifier"

GUIDANCE = {
    "good":         "Image quality is good. Proceed with assessment.",
    "blurry":       "Press the probe firmly against the skin and apply more ultrasound gel. Ask the patient to hold their breath.",
    "too_dark":     "Increase the gain setting on the ultrasound machine, or press the probe more firmly.",
    "low_contrast": "Re-position the probe — try a slightly different angle. Ensure the patient has a full bladder.",
    "noisy":        "Wipe off old gel, apply fresh gel, and reduce probe pressure slightly. Ask the patient to remain still.",
    "angled":       "Rotate the probe slightly until the cardiac structure appears centred on screen. Small adjustments work best.",
}

COLORS = {
    "good":         "#10b981",
    "blurry":       "#f59e0b",
    "too_dark":     "#6366f1",
    "low_contrast": "#3b82f6",
    "noisy":        "#ec4899",
    "angled":       "#14b8a6",
}

ICONS = {
    "good":         "✅",
    "blurry":       "🌫️",
    "too_dark":     "🌑",
    "low_contrast": "🔆",
    "noisy":        "📡",
    "angled":       "📐",
}


def load_model():
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename="ultrasound_efficientnet_b3.pth")
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    class_names = ckpt["class_names"]
    m = efficientnet_b3(weights=None)
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(p=0.3, inplace=True), nn.Linear(in_f, len(class_names)))
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    return m, class_names


model, CLASS_NAMES = load_model()

val_tf = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

gradcam = GradCAM(model=model, target_layers=[model.features[-1]])


def predict(image):
    if image is None:
        return None, build_empty_html()

    orig   = Image.fromarray(image).convert("RGB")
    tensor = val_tf(orig).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze().numpy()

    pred_idx   = int(probs.argmax())
    pred_label = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    # GradCAM overlay
    cam_map  = gradcam(input_tensor=tensor, targets=[ClassifierOutputTarget(pred_idx)])[0]
    orig_np  = np.array(orig.resize((300, 300)), dtype=np.float32) / 255.0
    cam_img  = show_cam_on_image(orig_np, cam_map, use_rgb=True)
    cam_img  = cv2.resize(cam_img, (image.shape[1], image.shape[0]))

    return cam_img, build_result_html(pred_label, confidence, probs)


def build_empty_html():
    return """
    <div style="font-family:'Inter',sans-serif;padding:40px;text-align:center;color:#9ca3af;">
        <div style="font-size:48px;margin-bottom:12px;">🫀</div>
        <p style="font-size:16px;">Upload an ultrasound image to begin analysis</p>
    </div>"""


def build_result_html(pred_label, confidence, probs):
    color = COLORS.get(pred_label, "#6b7280")
    icon  = ICONS.get(pred_label, "❓")
    guidance = GUIDANCE.get(pred_label, "")

    status_bg = "#dcfce7" if pred_label == "good" else "#fef9c3"
    status_border = "#16a34a" if pred_label == "good" else "#ca8a04"
    status_text = "PASS" if pred_label == "good" else "ACTION REQUIRED"

    bars = ""
    for cls, prob in sorted(zip(CLASS_NAMES, probs), key=lambda x: -x[1]):
        c       = COLORS.get(cls, "#6b7280")
        bold    = "font-weight:700;" if cls == pred_label else ""
        bars   += f"""
        <div style="display:flex;align-items:center;gap:10px;margin:7px 0;">
            <span style="width:105px;font-size:13px;color:#374151;{bold}">{cls.replace('_',' ')}</span>
            <div style="flex:1;background:#e5e7eb;border-radius:99px;height:12px;overflow:hidden;">
                <div style="background:{c};width:{prob*100:.1f}%;height:12px;border-radius:99px;"></div>
            </div>
            <span style="width:44px;text-align:right;font-size:13px;color:#6b7280;{bold}">{prob:.1%}</span>
        </div>"""

    return f"""
    <div style="font-family:'Inter',sans-serif;max-width:520px;">

      <!-- Status badge -->
      <div style="display:inline-flex;align-items:center;gap:6px;
                  background:{status_bg};border:1.5px solid {status_border};
                  padding:4px 14px;border-radius:99px;margin-bottom:16px;">
        <span style="width:8px;height:8px;background:{status_border};border-radius:50%;display:inline-block;"></span>
        <span style="font-size:12px;font-weight:700;color:{status_border};letter-spacing:0.05em;">{status_text}</span>
      </div>

      <!-- Main result card -->
      <div style="background:white;border:1.5px solid #e5e7eb;border-radius:16px;
                  padding:20px;margin-bottom:14px;box-shadow:0 1px 4px rgba(0,0,0,0.06);">
        <div style="display:flex;align-items:center;gap:14px;margin-bottom:16px;">
          <div style="width:52px;height:52px;border-radius:12px;background:{color}22;
                      display:flex;align-items:center;justify-content:center;font-size:26px;">
            {icon}
          </div>
          <div>
            <div style="font-size:22px;font-weight:700;color:#111827;">
              {pred_label.replace('_',' ').title()}
            </div>
            <div style="font-size:13px;color:#6b7280;">Confidence: <strong style="color:{color};">{confidence:.1%}</strong></div>
          </div>
        </div>

        <!-- Confidence ring visual -->
        <div style="background:#f9fafb;border-radius:10px;padding:12px 14px;margin-bottom:0;">
          <div style="font-size:11px;font-weight:600;color:#9ca3af;letter-spacing:0.08em;margin-bottom:8px;">
            CONFIDENCE BREAKDOWN
          </div>
          {bars}
        </div>
      </div>

      <!-- Guidance card -->
      <div style="background:#fffbeb;border:1.5px solid #fde68a;border-radius:16px;padding:16px;
                  {'display:none;' if pred_label == 'good' else ''}">
        <div style="display:flex;gap:10px;align-items:flex-start;">
          <div style="font-size:20px;margin-top:1px;">💡</div>
          <div>
            <div style="font-size:12px;font-weight:700;color:#92400e;letter-spacing:0.06em;margin-bottom:4px;">
              OPERATOR GUIDANCE
            </div>
            <div style="font-size:14px;color:#78350f;line-height:1.6;">{guidance}</div>
          </div>
        </div>
      </div>
      {'<div style="background:#f0fdf4;border:1.5px solid #bbf7d0;border-radius:16px;padding:16px;"><div style="display:flex;gap:10px;align-items:center;"><div style="font-size:20px;">✅</div><div style="font-size:14px;color:#166534;line-height:1.6;"><strong>Good quality image.</strong> Proceed with fetal cardiac assessment.</div></div></div>' if pred_label == "good" else ""}

    </div>"""


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

body, .gradio-container { font-family: 'Inter', sans-serif !important; background: #f1f5f9 !important; }

.gradio-container { max-width: 1100px !important; margin: 0 auto !important; }

#header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
    border-radius: 20px;
    padding: 32px 36px;
    margin-bottom: 24px;
    color: white;
}
#header h1 { font-size: 28px; font-weight: 700; margin: 0 0 6px; }
#header p  { font-size: 14px; opacity: 0.75; margin: 0; }

.panel {
    background: white;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    border: 1px solid #e2e8f0;
}

button.primary { background: #1e3a5f !important; border-radius: 12px !important; font-weight: 600 !important; }
button.primary:hover { background: #2d5282 !important; }

.label-wrap { font-size: 12px !important; font-weight: 600 !important; color: #6b7280 !important; letter-spacing: 0.05em; text-transform: uppercase; }
"""

with gr.Blocks(css=CSS, title="Fetal Echo Quality Classifier") as demo:

    gr.HTML("""
    <div id="header">
      <h1>🫀 Fetal Echocardiography Quality Classifier</h1>
      <p>AI-powered real-time image quality assessment for first-trimester fetal cardiac screening &nbsp;·&nbsp; EfficientNet-B3 &nbsp;·&nbsp; GradCAM explanations</p>
    </div>
    """)

    with gr.Row(equal_height=True):
        with gr.Column(scale=1, elem_classes="panel"):
            gr.HTML('<p style="font-size:12px;font-weight:600;color:#6b7280;letter-spacing:0.05em;margin:0 0 10px;">INPUT IMAGE</p>')
            img_in  = gr.Image(label="", type="numpy", height=300)
            run_btn = gr.Button("▶  Analyse Image", variant="primary", size="lg")

        with gr.Column(scale=1, elem_classes="panel"):
            gr.HTML('<p style="font-size:12px;font-weight:600;color:#6b7280;letter-spacing:0.05em;margin:0 0 10px;">GRADCAM ATTENTION MAP</p>')
            cam_out = gr.Image(label="", height=300)

    gr.HTML('<div style="height:16px;"></div>')

    with gr.Row():
        with gr.Column(elem_classes="panel"):
            gr.HTML('<p style="font-size:12px;font-weight:600;color:#6b7280;letter-spacing:0.05em;margin:0 0 12px;">ASSESSMENT RESULTS</p>')
            result_out = gr.HTML(value=build_empty_html())

    run_btn.click(fn=predict, inputs=img_in, outputs=[cam_out, result_out])
    img_in.change(fn=predict, inputs=img_in, outputs=[cam_out, result_out])

    gr.HTML("""
    <div style="text-align:center;padding:20px;font-size:12px;color:#94a3b8;">
      Built with EfficientNet-B3 · Trained on LIFE Project fetal echocardiography data with synthetic augmentations
    </div>""")

demo.launch()
