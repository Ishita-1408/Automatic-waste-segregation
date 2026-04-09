"""
Automatic Waste Segregation — Inference
========================================
Run predictions on single images or entire directories.

Usage:
    python src/predict.py --image path/to/image.jpg
    python src/predict.py --dir  path/to/folder/
"""

import json
import argparse
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models


# Recycling Tips

TIPS = {
    "cardboard": {
        "bin": "Blue (Recyclable)",
        "tip": "Flatten boxes before recycling. Remove tape and staples if possible.",
        "recyclable": True,
    },
    "glass": {
        "bin": "Green (Glass)",
        "tip": "Rinse containers. Never recycle broken glass in regular bins — wrap safely.",
        "recyclable": True,
    },
    "metal": {
        "bin": "Blue (Recyclable)",
        "tip": "Rinse cans. Aluminium is infinitely recyclable — a great choice!",
        "recyclable": True,
    },
    "paper": {
        "bin": "Blue (Recyclable)",
        "tip": "Keep paper dry. Shredded paper should go in a sealed bag.",
        "recyclable": True,
    },
    "plastic": {
        "bin": "Yellow (Plastic)",
        "tip": "Check the resin code (♺1–7). Codes 1 & 2 are most widely accepted.",
        "recyclable": True,
    },
    "trash": {
        "bin": "Black (Landfill)",
        "tip": "Consider if anything can be reused or repurposed before discarding.",
        "recyclable": False,
    },
}


# Model

def load_model(model_path: str, meta_path: str, device: torch.device):
    with open(meta_path) as f:
        meta = json.load(f)

    num_classes = meta["num_classes"]
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    return model, meta


def get_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


# Prediction

def predict_image(image_path: str, model, meta: dict, device: torch.device) -> dict:
    transform = get_transform(meta["img_size"])
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    class_names = meta["class_names"]
    conf, idx = probs.max(0)
    label = class_names[idx.item()]
    tip_info = TIPS.get(label, {})

    top5 = sorted(
        [{"class": class_names[i], "confidence": round(probs[i].item(), 4)}
         for i in range(len(class_names))],
        key=lambda x: -x["confidence"],
    )

    return {
        "image": image_path,
        "label": label,
        "confidence": round(conf.item(), 4),
        "bin": tip_info.get("bin", "Unknown"),
        "tip": tip_info.get("tip", ""),
        "recyclable": tip_info.get("recyclable", False),
        "top5": top5,
    }


def predict_dir(dir_path: str, model, meta: dict, device: torch.device) -> list:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    results = []
    paths = [p for p in Path(dir_path).rglob("*") if p.suffix.lower() in exts]
    for p in sorted(paths):
        result = predict_image(str(p), model, meta, device)
        results.append(result)
        recyclable_tag = "♻ " if result["recyclable"] else "🗑 "
        print(f"{recyclable_tag}{p.name:40s} → {result['label']:12s} "
              f"({result['confidence']:.1%})  [{result['bin']}]")
    return results


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Predict waste category for image(s)")
    parser.add_argument("--image", help="Path to a single image")
    parser.add_argument("--dir", help="Path to a folder of images")
    parser.add_argument("--model", default="models/best_model.pth",
                        help="Model weights path")
    parser.add_argument("--meta", default="models/model_meta.json",
                        help="Model metadata JSON")
    parser.add_argument("--output", default="",
                        help="Optional JSON output path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_model(args.model, args.meta, device)

    if args.image:
        result = predict_image(args.image, model, meta, device)
        print(f"\n{'='*50}")
        print(f"Image      : {result['image']}")
        print(f"Prediction : {result['label'].upper()} ({result['confidence']:.1%})")
        print(f"Bin        : {result['bin']}")
        print(f"Recyclable : {'Yes ♻' if result['recyclable'] else 'No 🗑'}")
        print(f"Tip        : {result['tip']}")
        print("\nTop predictions:")
        for item in result["top5"]:
            bar = "█" * int(item["confidence"] * 20)
            print(f"  {item['class']:12s} {bar:20s} {item['confidence']:.1%}")
        results = [result]

    elif args.dir:
        print(f"Scanning {args.dir} ...\n")
        results = predict_dir(args.dir, model, meta, device)
        from collections import Counter
        counts = Counter(r["label"] for r in results)
        print(f"\n{'='*50}")
        print("Summary:")
        for label, count in counts.most_common():
            print(f"  {label:12s}: {count}")
    else:
        parser.print_help()
        return

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
