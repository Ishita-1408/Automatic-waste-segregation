"""
Automatic Waste Segregation — Flask Web Application
=====================================================
A lightweight web UI to classify waste images via drag-and-drop.

Run:
    python src/app.py
"""

import os
import io
import json
from flask import Flask, request, jsonify, render_template_string
from PIL import Image

# ── lazy imports (torch only loaded when needed) ──────────────────────────────
_model = None
_meta = None
_device = None


def get_model():
    global _model, _meta, _device
    if _model is None:
        import torch
        import torch.nn as nn
        from torchvision import models

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        meta_path = os.environ.get("META_PATH", "models/model_meta.json")
        model_path = os.environ.get("MODEL_PATH", "models/best_model.pth")

        with open(meta_path, "r", encoding="utf-8") as f:
            _meta = json.load(f)

        m = models.efficientnet_b0(weights=None)
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_f, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, _meta["num_classes"]),
        )

        m.load_state_dict(torch.load(model_path, map_location=_device))
        m.to(_device).eval()
        _model = m

    return _model, _meta, _device


# ── 12-class disposal guidance ────────────────────────────────────────────────
TIPS = {
    "battery": {
        "bin": "Red (Hazardous / E-Waste)",
        "tip": "Never throw batteries in regular trash. Drop them at an authorised battery or e-waste collection point.",
        "recyclable": False,
        "emoji": "🔋",
    },
    "biological": {
        "bin": "Green (Organic / Compost)",
        "tip": "Food scraps and biodegradable waste can go to compost if your local system supports it.",
        "recyclable": False,
        "emoji": "🍌",
    },
    "brown-glass": {
        "bin": "Green (Glass Recycling)",
        "tip": "Rinse glass containers before disposal. Wrap broken glass safely before discarding.",
        "recyclable": True,
        "emoji": "🍾",
    },
    "green-glass": {
        "bin": "Green (Glass Recycling)",
        "tip": "Rinse bottles and jars before recycling. Avoid mixing with ceramics or mirrors.",
        "recyclable": True,
        "emoji": "🍾",
    },
    "white-glass": {
        "bin": "Green (Glass Recycling)",
        "tip": "Clear glass is recyclable in most systems. Rinse and separate from lids if required.",
        "recyclable": True,
        "emoji": "🥛",
    },
    "cardboard": {
        "bin": "Blue (Paper / Cardboard Recycling)",
        "tip": "Flatten cardboard boxes and remove tape or food contamination before recycling.",
        "recyclable": True,
        "emoji": "📦",
    },
    "clothes": {
        "bin": "Donation / Textile Collection",
        "tip": "If wearable, donate it. If damaged, check for a textile recycling or cloth collection point.",
        "recyclable": False,
        "emoji": "👕",
    },
    "metal": {
        "bin": "Blue (Metal Recycling)",
        "tip": "Rinse cans and containers before disposal. Metal is highly recyclable.",
        "recyclable": True,
        "emoji": "🥫",
    },
    "paper": {
        "bin": "Blue (Paper Recycling)",
        "tip": "Keep paper dry and clean. Avoid recycling heavily oily or food-soiled paper.",
        "recyclable": True,
        "emoji": "📄",
    },
    "plastic": {
        "bin": "Yellow (Plastic Recycling)",
        "tip": "Rinse containers and check local recycling rules. Codes 1 and 2 are most widely accepted.",
        "recyclable": True,
        "emoji": "🧴",
    },
    "shoes": {
        "bin": "Donation / Textile Collection",
        "tip": "If usable, donate them. Worn-out shoes may need specialised textile or footwear recycling.",
        "recyclable": False,
        "emoji": "👟",
    },
    "trash": {
        "bin": "Black (Landfill)",
        "tip": "Use landfill only when the item cannot be reused, recycled, or composted.",
        "recyclable": False,
        "emoji": "🗑️",
    },
}

BIN_COLORS = {
    "Blue (Paper / Cardboard Recycling)": "#2563eb",
    "Blue (Metal Recycling)": "#1d4ed8",
    "Yellow (Plastic Recycling)": "#ca8a04",
    "Green (Glass Recycling)": "#16a34a",
    "Green (Organic / Compost)": "#22c55e",
    "Black (Landfill)": "#374151",
    "Red (Hazardous / E-Waste)": "#dc2626",
    "Donation / Textile Collection": "#7c3aed",
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB


HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>WasteAI — Smart Waste Segregation</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

  :root {
    --bg:     #0a0d12;
    --surface:#111620;
    --border: #1e2633;
    --text:   #e8edf5;
    --muted:  #6b7a99;
    --green:  #22c55e;
    --red:    #ef4444;
    --accent: #38bdf8;
    --radius: 16px;
  }

  * { box-sizing:border-box; margin:0; padding:0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  header {
    width: 100%;
    padding: 24px 40px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 14px;
    background: linear-gradient(to bottom, #0d1117, transparent);
  }
  .logo-mark {
    width: 40px; height: 40px;
    background: linear-gradient(135deg, #22c55e, #38bdf8);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
  }
  header h1 {
    font-family: 'Syne', sans-serif;
    font-size: 22px;
    font-weight: 800;
    letter-spacing: -0.5px;
  }
  header p  {
    font-size: 13px;
    color: var(--muted);
    margin-left: auto;
  }

  main {
    width: 100%;
    max-width: 900px;
    padding: 60px 24px 40px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 40px;
  }

  .hero { text-align: center; }
  .hero h2 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(28px, 5vw, 48px);
    font-weight: 800;
    line-height: 1.1;
    background: linear-gradient(135deg, #e8edf5 0%, var(--accent) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .hero p {
    margin-top: 12px;
    color: var(--muted);
    font-size: 16px;
  }

  .drop-zone {
    width: 100%;
    border: 2px dashed var(--border);
    border-radius: var(--radius);
    padding: 60px 30px;
    text-align: center;
    cursor: pointer;
    transition: all .25s ease;
    background: var(--surface);
    position: relative;
  }
  .drop-zone:hover, .drop-zone.dragover {
    border-color: var(--accent);
    background: rgba(56,189,248,.06);
  }
  .drop-zone .icon { font-size: 52px; margin-bottom: 16px; }
  .drop-zone h3 {
    font-family: 'Syne', sans-serif;
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 8px;
  }
  .drop-zone p { color: var(--muted); font-size: 14px; }
  .drop-zone input[type=file] {
    position: absolute; inset: 0; opacity: 0; cursor: pointer;
  }

  #preview-img {
    max-width: 100%;
    max-height: 300px;
    border-radius: 12px;
    margin-bottom: 16px;
    display: none;
    object-fit: contain;
    border: 1px solid var(--border);
  }

  .btn {
    background: linear-gradient(135deg, #22c55e, #16a34a);
    color: #fff;
    border: none;
    padding: 14px 36px;
    border-radius: 10px;
    font-size: 15px;
    font-weight: 600;
    cursor: pointer;
    font-family: 'Syne', sans-serif;
    letter-spacing: 0.3px;
    transition: opacity .2s;
    display: none;
  }
  .btn:hover { opacity: .88; }
  .btn:disabled { opacity: .4; cursor: not-allowed; }

  .spinner {
    display: none;
    width: 32px; height: 32px;
    border: 3px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin .8s linear infinite;
    margin: 0 auto;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  #result-card {
    display: none;
    width: 100%;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 32px;
    animation: fadeUp .35s ease;
  }
  @keyframes fadeUp {
    from { opacity:0; transform:translateY(16px); }
    to { opacity:1; transform:translateY(0); }
  }

  .result-top {
    display: flex;
    align-items: center;
    gap: 20px;
    margin-bottom: 28px;
  }
  .result-emoji { font-size: 56px; }
  .result-label h3 {
    font-family: 'Syne', sans-serif;
    font-size: 28px;
    font-weight: 800;
    text-transform: capitalize;
  }
  .result-label .conf {
    font-size: 14px;
    color: var(--muted);
    margin-top: 4px;
  }

  .badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 99px;
    font-size: 13px;
    font-weight: 500;
    margin-top: 8px;
  }
  .badge.recyclable { background: rgba(34,197,94,.15); color: var(--green); }
  .badge.landfill   { background: rgba(239,68,68,.15);  color: var(--red);   }

  .info-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 24px;
  }
  .info-box {
    background: rgba(255,255,255,.03);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 18px;
  }
  .info-box .label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--muted);
    margin-bottom: 6px;
  }
  .info-box .value {
    font-size: 15px;
    font-weight: 500;
  }
  .bin-dot {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    margin-right: 6px;
  }

  .bars-title {
    font-size: 13px;
    color: var(--muted);
    margin-bottom: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
  }
  .bar-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
  }
  .bar-label {
    width: 110px;
    font-size: 13px;
    text-transform: capitalize;
    flex-shrink: 0;
  }
  .bar-track {
    flex: 1;
    background: var(--border);
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
  }
  .bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, var(--accent), #6366f1);
    transition: width .5s ease;
  }
  .bar-pct {
    width: 48px;
    font-size: 13px;
    color: var(--muted);
    text-align: right;
  }

  .tip-box {
    background: rgba(56,189,248,.08);
    border: 1px solid rgba(56,189,248,.2);
    border-radius: 12px;
    padding: 14px 18px;
    font-size: 14px;
    margin-top: 20px;
    display: flex;
    gap: 10px;
    align-items: flex-start;
  }
  .tip-icon { font-size: 18px; flex-shrink: 0; }

  .new-btn {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--muted);
    padding: 10px 22px;
    border-radius: 8px;
    font-size: 14px;
    cursor: pointer;
    margin-top: 20px;
    transition: all .2s;
    font-family: 'DM Sans', sans-serif;
  }
  .new-btn:hover { border-color: var(--text); color: var(--text); }

  footer {
    margin-top: auto;
    padding: 28px;
    color: var(--muted);
    font-size: 13px;
    text-align: center;
  }

  @media (max-width: 700px) {
    header { padding: 18px 20px; }
    header p { display: none; }
    .info-grid { grid-template-columns: 1fr; }
    .result-top { align-items: flex-start; }
  }
</style>
</head>
<body>
<header>
  <div class="logo-mark">♻️</div>
  <h1>WasteAI</h1>
  <p>EfficientNet-B0 · 12 categories · Kaggle dataset</p>
</header>

<main>
  <div class="hero">
    <h2>Classify Waste.<br>Help the Planet.</h2>
    <p>Drop an image of any waste item and the AI will suggest the most suitable disposal method.</p>
  </div>

  <div style="width:100%;text-align:center;">
    <img id="preview-img" alt="Preview"/>
    <div class="drop-zone" id="drop-zone">
      <div class="icon">🗃️</div>
      <h3>Drop your waste image here</h3>
      <p>JPG, PNG, WEBP · max 16 MB</p>
      <input type="file" id="file-input" accept="image/*"/>
    </div>
    <br/>
    <button class="btn" id="classify-btn" onclick="classify()">Classify Waste ↗</button>
    <div class="spinner" id="spinner"></div>
  </div>

  <div id="result-card">
    <div class="result-top">
      <div class="result-emoji" id="res-emoji"></div>
      <div class="result-label">
        <h3 id="res-label"></h3>
        <div class="conf" id="res-conf"></div>
        <span class="badge" id="res-badge"></span>
      </div>
    </div>

    <div class="info-grid">
      <div class="info-box">
        <div class="label">Disposal Bin</div>
        <div class="value" id="res-bin"></div>
      </div>
      <div class="info-box">
        <div class="label">Recyclable</div>
        <div class="value" id="res-recyclable"></div>
      </div>
    </div>

    <div class="bars-title">Top Prediction Scores</div>
    <div id="bars-container"></div>

    <div class="tip-box">
      <div class="tip-icon">💡</div>
      <div id="res-tip"></div>
    </div>

    <button class="new-btn" onclick="reset()">← Try Another Image</button>
  </div>
</main>

<footer>WasteAI · EfficientNet-B0 fine-tuned on Garbage Classification (Kaggle) · Built for sustainable living</footer>

<script>
const BIN_COLORS = {{ bin_colors | tojson }};
let selectedFile = null;

const dropZone    = document.getElementById('drop-zone');
const fileInput   = document.getElementById('file-input');
const previewImg  = document.getElementById('preview-img');
const classifyBtn = document.getElementById('classify-btn');
const spinner     = document.getElementById('spinner');
const resultCard  = document.getElementById('result-card');

dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

function handleFile(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
    previewImg.style.display = 'block';
    dropZone.style.display = 'none';
    classifyBtn.style.display = 'inline-block';
    resultCard.style.display = 'none';
  };
  reader.readAsDataURL(file);
}

async function classify() {
  if (!selectedFile) return;

  classifyBtn.style.display = 'none';
  spinner.style.display = 'block';

  const fd = new FormData();
  fd.append('file', selectedFile);

  try {
    const resp = await fetch('/predict', { method: 'POST', body: fd });
    const data = await resp.json();

    if (data.error) {
      alert(data.error);
      classifyBtn.style.display = 'inline-block';
      return;
    }

    showResult(data);
  } catch (e) {
    alert('Prediction failed. Make sure the model files are present.');
    classifyBtn.style.display = 'inline-block';
  } finally {
    spinner.style.display = 'none';
  }
}

function showResult(d) {
  document.getElementById('res-emoji').textContent = d.emoji;
  document.getElementById('res-label').textContent = d.label.replaceAll('-', ' ');
  document.getElementById('res-conf').textContent  = `Confidence: ${(d.confidence * 100).toFixed(1)}%`;

  const badge = document.getElementById('res-badge');
  if (d.recyclable) {
    badge.textContent = '♻ Recyclable';
    badge.className = 'badge recyclable';
  } else {
    badge.textContent = '🗑 Special / Non-Recyclable';
    badge.className = 'badge landfill';
  }

  const color = BIN_COLORS[d.bin] || '#6b7a99';
  document.getElementById('res-bin').innerHTML =
    `<span class="bin-dot" style="background:${color}"></span>${d.bin}`;

  document.getElementById('res-recyclable').textContent = d.recyclable ? 'Yes' : 'No';
  document.getElementById('res-tip').textContent = d.tip;

  const cont = document.getElementById('bars-container');
  cont.innerHTML = '';

  d.top5.forEach(item => {
    const pct = (item.confidence * 100).toFixed(1);
    cont.innerHTML += `
      <div class="bar-row">
        <div class="bar-label">${item.class.replaceAll('-', ' ')}</div>
        <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
        <div class="bar-pct">${pct}%</div>
      </div>`;
  });

  resultCard.style.display = 'block';
}

function reset() {
  selectedFile = null;
  previewImg.style.display = 'none';
  dropZone.style.display = 'block';
  classifyBtn.style.display = 'none';
  resultCard.style.display = 'none';
  fileInput.value = '';
}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_PAGE, bin_colors=BIN_COLORS)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "Cannot open image"}), 400

    import torch
    from torchvision import transforms

    model, meta, device = get_model()

    transform = transforms.Compose([
        transforms.Resize((meta["img_size"], meta["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    class_names = meta["class_names"]
    conf, idx = probs.max(0)
    label = class_names[idx.item()]
    tip_info = TIPS.get(label, {
        "bin": "Unknown",
        "tip": "Please check your local municipal recycling guidelines for this item.",
        "recyclable": False,
        "emoji": "♻️",
    })

    top5 = sorted(
        [
            {
                "class": class_names[i],
                "confidence": round(probs[i].item(), 4)
            }
            for i in range(len(class_names))
        ],
        key=lambda x: -x["confidence"]
    )[:5]

    return jsonify({
        "label": label,
        "confidence": round(conf.item(), 4),
        "bin": tip_info.get("bin", "Unknown"),
        "tip": tip_info.get("tip", ""),
        "recyclable": tip_info.get("recyclable", False),
        "emoji": tip_info.get("emoji", "♻️"),
        "top5": top5,
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🌱 WasteAI running at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)