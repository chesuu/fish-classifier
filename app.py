import os
import base64
import inspect
import numpy as np
import tensorflow as tf
import keras
import gradio as gr
from PIL import Image, ImageOps

CLASSES = ["bangus", "galunggong", "tilapia"]  # alphabetical, must match training
THRESHOLD = 0.60
COLORS = {"bangus": "#38bdf8", "galunggong": "#3ddc84", "tilapia": "#fbbf24"}
ACCENT, GOOD, WARN = "#2d9cff", "#3ddc84", "#ffd166"

# Load once at startup, not per request
MODELS = {
    "MobileNetV2": keras.models.load_model("models/mobilenetv2.keras"),
    "Custom CNN": keras.models.load_model("models/custom_cnn.keras"),
}


def preprocess(img):
    img = ImageOps.exif_transpose(img).convert("RGB")  # fix phone rotation, force 3 channels
    arr = tf.convert_to_tensor(np.array(img), dtype=tf.float32)  # raw 0-255, NO /255
    arr = tf.image.resize_with_pad(arr, 224, 224)  # pad, don't stretch
    return tf.expand_dims(arr, 0)


def run_model(model, batch):
    return [float(p) for p in model.predict(batch, verbose=0)[0]]


# ---------- Optional real photo: put an underwater picture at assets/ocean.jpg ----------
def load_bg():
    for ext in ("jpg", "jpeg", "png", "webp"):
        path = os.path.join("assets", f"ocean.{ext}")
        if os.path.exists(path):
            mime = "jpeg" if ext in ("jpg", "jpeg") else ext
            with open(path, "rb") as f:
                return f"data:image/{mime};base64," + base64.b64encode(f.read()).decode()
    return None


BG = load_bg()

# ---------- Deep Ocean theme ----------
CSS = """
body, .gradio-container {
  background: radial-gradient(circle at 20% 0%, rgba(45,156,255,0.10), transparent 45%),
              linear-gradient(180deg, #050a14 0%, #08111f 60%, #050a14 100%) !important;
  color: #e8f1ff !important;
  font-family: Inter, "Segoe UI", system-ui, sans-serif !important;
}
.gradio-container {
  max-width: 1180px !important; margin: auto !important;
  --body-text-color: #e8f1ff !important; --body-text-color-subdued: #8aa4c8 !important;
  --background-fill-primary: rgba(16,26,44,0.78) !important; --background-fill-secondary: rgba(16,26,44,0.6) !important;
  --block-background-fill: rgba(16,26,44,0.78) !important; --block-border-color: rgba(255,255,255,0.08) !important;
  --border-color-primary: rgba(255,255,255,0.1) !important; --input-background-fill: rgba(16,26,44,0.78) !important;
  --block-label-background-fill: rgba(5,10,20,0.85) !important; --block-label-text-color: #7fc4ff !important;
  --panel-background-fill: rgba(16,26,44,0.78) !important; --color-accent: #2d9cff !important;
  --button-primary-background-fill: linear-gradient(90deg,#2d9cff,#22d3ee) !important;
  --button-primary-background-fill-hover: linear-gradient(90deg,#22d3ee,#2d9cff) !important;
  --button-primary-text-color: #03111f !important;
}
#photo { border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 20px !important;
  box-shadow: 0 10px 30px rgba(0,0,0,0.4); }
#go { font-weight: 700 !important; border: none !important; border-radius: 14px !important;
  box-shadow: 0 0 20px rgba(45,156,255,0.4); }
.plain { background: transparent !important; border: none !important; box-shadow: none !important; padding: 0 !important; }
footer { display: none !important; }
.hero-bg {
  background: radial-gradient(ellipse at 70% 0%, rgba(80,170,255,0.38), transparent 55%),
              linear-gradient(180deg, #0c3a63 0%, #07213d 45%, #030c1b 100%);
}
.hero-bg::before {
  content: ""; position: absolute; inset: -20% -10% 0 -10%; pointer-events: none;
  background: repeating-linear-gradient(104deg, rgba(150,210,255,0.13) 0 16px, transparent 16px 78px);
  -webkit-mask-image: linear-gradient(to bottom, #000 0%, transparent 80%);
  mask-image: linear-gradient(to bottom, #000 0%, transparent 80%);
  animation: sway 9s ease-in-out infinite alternate;
}
.hero-bg::after { content: ""; position: absolute; inset: 0; pointer-events: none;
  background: linear-gradient(to top, rgba(2,8,18,0.75), transparent 55%); }
@keyframes sway { from { transform: translateX(-18px) skewX(-2deg); } to { transform: translateX(18px) skewX(2deg); } }
.bub { position: absolute; bottom: -20px; z-index: 1; border-radius: 50%; pointer-events: none;
  border: 1px solid rgba(180,225,255,0.45);
  background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.35), rgba(255,255,255,0.02));
  animation: rise linear infinite; }
@keyframes rise { 0% { transform: translateY(0) translateX(0); opacity: 0; } 10% { opacity: 0.9; }
  100% { transform: translateY(-340px) translateX(14px); opacity: 0; } }
"""
if BG:
    CSS += ('.hero-bg { background: linear-gradient(rgba(3,12,27,0.35), rgba(3,12,27,0.70)), '
            f'url("{BG}") center / cover no-repeat !important; }}')
THEME = gr.themes.Base(primary_hue="blue", neutral_hue="slate")

CARD = ("background:rgba(16,26,44,0.78);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);"
        "border:1px solid rgba(255,255,255,0.08);border-radius:20px;padding:20px;color:#e8f1ff;"
        "box-shadow:0 10px 30px rgba(0,0,0,0.4);")
MUTED = "color:#8aa4c8;"

FISH = ('<svg viewBox="0 0 200 100" width="200" style="position:absolute;right:150px;top:70px;opacity:0.2;z-index:1;">'
        '<path d="M10 50 C40 8 110 8 150 50 C110 92 40 92 10 50 Z M148 50 L196 18 L180 50 L196 82 Z" fill="#9fd8ff"/>'
        '<circle cx="42" cy="42" r="4" fill="#06223f"/></svg>')
BUBBLES = "".join(
    f'<span class="bub" style="left:{l}%;width:{s}px;height:{s}px;animation-duration:{d}s;animation-delay:{dl}s;"></span>'
    for l, s, d, dl in [(8, 8, 9, 0), (18, 5, 7, 2), (27, 11, 11, 1), (41, 6, 8, 4), (55, 9, 10, 3),
                        (63, 5, 7, 0.5), (74, 12, 12, 2), (86, 7, 9, 5), (93, 5, 8, 1)]
)
ICONS = {
    "agree": '<circle cx="9" cy="12" r="5"/><circle cx="15" cy="12" r="5"/>',
    "m1": '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/>',
    "m2": '<circle cx="5" cy="12" r="2"/><circle cx="19" cy="6" r="2"/><circle cx="19" cy="18" r="2"/><path d="M7 12l10-5M7 12l10 5"/>',
    "status": '<path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/><path d="M8.5 12l2.5 2.5L16 9.5"/>',
}


def stat(kind, value, label, color="#e8f1ff"):
    return (
        '<div style="text-align:center;min-width:76px;">'
        f'<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="{ACCENT}" stroke-width="1.6" '
        f'stroke-linecap="round" stroke-linejoin="round">{ICONS[kind]}</svg>'
        f'<div style="font-size:18px;font-weight:700;margin-top:6px;color:{color};">{value}</div>'
        f'<div style="font-size:12px;{MUTED}">{label}</div></div>'
    )


def header():
    return (
        '<div style="text-align:center;padding:10px 0 2px;">'
        '<div style="font-size:30px;font-weight:800;letter-spacing:.06em;">FISH '
        f'<span style="color:{ACCENT};">CLASSIFIER</span></div>'
        f'<div style="font-size:14px;{MUTED}">Snap a fish. Know the species.</div></div>'
    )


def hero(m=None, c=None):
    if m is None:
        name, color, conf, ring = "No photo yet", "#8aa4c8", 0.0, ACCENT
        sub, shown = "Upload or take a photo to identify the fish.", "--"
        stats = (stat("agree", "--", "Models agree") + stat("m1", "--", "MobileNetV2")
                 + stat("m2", "--", "Custom CNN") + stat("status", "--", "Reliability"))
    else:
        mi, ci = int(np.argmax(m)), int(np.argmax(c))
        name, color, conf = CLASSES[mi].capitalize(), COLORS[CLASSES[mi]], m[mi]
        agree, ok = mi == ci, m[mi] >= THRESHOLD
        sub = ("Both models agree on this species." if agree
               else f"Models disagree. Custom CNN says {CLASSES[ci].capitalize()}.")
        ring, shown = (GOOD if ok else WARN), f"{conf * 100:.0f}"
        stats = (stat("agree", "Yes" if agree else "No", "Models agree", GOOD if agree else WARN)
                 + stat("m1", f"{m[mi] * 100:.0f}%", "MobileNetV2")
                 + stat("m2", f"{c[ci] * 100:.0f}%", "Custom CNN")
                 + stat("status", "Confident" if ok else "Unsure", "Reliability", GOOD if ok else WARN))
    dash = conf * 326.73
    ring_svg = (
        '<div style="position:relative;width:128px;height:128px;flex:none;">'
        '<svg width="128" height="128" viewBox="0 0 128 128" style="transform:rotate(-90deg);">'
        '<circle cx="64" cy="64" r="52" fill="none" stroke="rgba(255,255,255,0.12)" stroke-width="9"/>'
        f'<circle cx="64" cy="64" r="52" fill="none" stroke="{ring}" stroke-width="9" stroke-linecap="round" '
        f'stroke-dasharray="{dash:.1f} 326.73" style="filter:drop-shadow(0 0 6px {ring});"/></svg>'
        '<div style="position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;">'
        f'<div style="font-size:34px;font-weight:800;line-height:1;">{shown}</div>'
        f'<div style="font-size:11px;letter-spacing:.1em;{MUTED}">CONFIDENCE</div></div></div>'
    )
    return (
        '<div class="hero-bg" style="position:relative;overflow:hidden;border-radius:24px;'
        'border:1px solid rgba(120,190,255,0.2);padding:24px;min-height:320px;box-shadow:0 14px 40px rgba(0,0,0,0.5);">'
        f'{BUBBLES}{FISH}<div style="position:relative;z-index:2;">'
        '<div style="display:flex;justify-content:space-between;align-items:center;gap:16px;">'
        '<div><div style="font-size:13px;letter-spacing:.12em;color:#b9d6f5;">FISH IDENTIFICATION</div>'
        f'<div style="font-size:42px;font-weight:800;color:{color};text-shadow:0 0 22px {color}66;margin-top:2px;">{name}</div>'
        f'<div style="font-size:15px;color:#cfe2f7;max-width:260px;">{sub}</div></div>{ring_svg}</div>'
        '<div style="display:flex;justify-content:space-around;flex-wrap:wrap;gap:12px;margin-top:26px;'
        f'padding-top:16px;border-top:1px solid rgba(255,255,255,0.1);">{stats}</div></div></div>'
    )


def empty_card(title):
    return (f'<div style="{CARD}"><div style="font-size:13px;letter-spacing:.1em;{MUTED}">{title.upper()}</div>'
            '<div style="font-size:17px;color:#5b7399;padding:30px 0;">Waiting for a photo...</div></div>')


def card(title, probs):
    order = sorted(range(3), key=lambda i: -probs[i])
    top, conf = CLASSES[order[0]], probs[order[0]]
    ok = conf >= THRESHOLD
    tag, tag_color = ("Confident", GOOD) if ok else ("Unsure", WARN)
    bars = ""
    for i in order:
        pct, col = probs[i] * 100, COLORS[CLASSES[i]]
        bars += (
            '<div style="margin-top:12px;">'
            f'<div style="display:flex;justify-content:space-between;font-size:14px;"><span>{CLASSES[i].capitalize()}</span><span>{pct:.1f}%</span></div>'
            '<div style="height:10px;background:rgba(255,255,255,0.1);border-radius:6px;overflow:hidden;margin-top:4px;">'
            f'<div style="width:{pct:.1f}%;height:100%;background:{col};box-shadow:0 0 10px {col};border-radius:6px;"></div></div></div>'
        )
    note = ('<div style="margin-top:16px;padding:9px 12px;border-radius:10px;background:rgba(255,209,102,0.12);'
            'border:1px solid rgba(255,209,102,0.5);color:#ffd166;font-weight:600;">Not sure, try a clearer photo</div>'
            if not ok else
            f'<div style="margin-top:16px;padding-top:12px;border-top:1px solid rgba(255,255,255,0.08);font-size:13px;{MUTED}">'
            'Clear match. Reliable result.</div>')
    return (
        f'<div style="{CARD}"><div style="font-size:13px;letter-spacing:.1em;{MUTED}">{title.upper()}</div>'
        '<div style="display:flex;align-items:baseline;gap:10px;margin-top:4px;">'
        f'<span style="font-size:30px;font-weight:800;color:{COLORS[top]};">{top.capitalize()}</span>'
        f'<span style="font-size:20px;font-weight:600;">{conf * 100:.1f}%</span></div>'
        f'<div style="font-size:15px;font-weight:600;color:{tag_color};">{tag}</div>{bars}{note}</div>'
    )


def predict(img):
    if img is None:
        return hero(), empty_card("MobileNetV2"), empty_card("Custom CNN")
    batch = preprocess(img)
    m = run_model(MODELS["MobileNetV2"], batch)
    c = run_model(MODELS["Custom CNN"], batch)
    return hero(m, c), card("MobileNetV2", m), card("Custom CNN", c)


examples = (
    [os.path.join("samples", f) for f in sorted(os.listdir("samples"))
     if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
    if os.path.isdir("samples")
    else []
)

# Newer Gradio versions take css/theme in launch(); older ones in Blocks()
USE_LAUNCH = "css" in inspect.signature(gr.Blocks.launch).parameters
page_args = {} if USE_LAUNCH else {"css": CSS, "theme": THEME}

with gr.Blocks(title="Fish Classifier", **page_args) as demo:
    gr.HTML(header(), elem_classes=["plain"])
    with gr.Row():
        with gr.Column(scale=4):
            inp = gr.Image(type="pil", sources=["upload", "webcam"], label="Fish photo",
                           height=360, elem_id="photo")
            btn = gr.Button("Classify", variant="primary", elem_id="go")
            if examples:
                gr.Examples(examples=examples, inputs=inp, label="Click a sample photo")
        with gr.Column(scale=6):
            out_h = gr.HTML(hero(), elem_classes=["plain"])
            gr.HTML(f'<div style="font-size:13px;letter-spacing:.12em;{MUTED}margin-top:4px;">MODEL COMPARISON</div>',
                    elem_classes=["plain"])
            with gr.Row():
                out_m = gr.HTML(empty_card("MobileNetV2"), elem_classes=["plain"])
                out_c = gr.HTML(empty_card("Custom CNN"), elem_classes=["plain"])
    outs = [out_h, out_m, out_c]
    btn.click(predict, inp, outs)
    inp.change(predict, inp, outs)  # classify automatically when a photo is added

if __name__ == "__main__":
    demo.launch(
        css=CSS,
        theme=THEME,
        share=True
    )