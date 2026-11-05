from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_history(history: list[dict[str, float]], output_dir: Path) -> None:
    if not history:
        raise ValueError("No hay historial de entrenamiento para guardar.")
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "loss_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    epochs = [int(item["epoch"]) for item in history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [item["train_loss"] for item in history], label="Train")
    plt.plot(epochs, [item["validation_loss"] for item in history], label="Validation")
    plt.xlabel("Época")
    plt.ylabel("Pérdida total")
    plt.title("Pérdida de entrenamiento y validación")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=160)
    plt.close()


def save_predictions(
    predictions: list[dict[str, Any]],
    categories: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    from data import ids_to_rgb

    annotation_dir = output_dir / "panoptic"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    annotations: list[dict[str, Any]] = []
    for prediction in sorted(predictions, key=lambda item: int(item["image_id"])):
        image_id = int(prediction["image_id"])
        file_name = f"panoptic/{int(image_id)}.png"
        segmentation = np.asarray(prediction["segmentation"], dtype=np.uint32)
        Image.fromarray(ids_to_rgb(segmentation), mode="RGB").save(output_dir / file_name)
        segments_info = []
        for segment in prediction["segments_info"]:
            segment_id = int(segment["id"])
            category_id = int(segment["category_id"])
            mask = segmentation == segment_id
            ys, xs = np.nonzero(mask)
            if not len(xs):
                continue
            segments_info.append(
                {
                    "id": segment_id,
                    "category_id": category_id,
                    "area": int(len(xs)),
                    "bbox": [
                        int(xs.min()),
                        int(ys.min()),
                        int(xs.max() - xs.min() + 1),
                        int(ys.max() - ys.min() + 1),
                    ],
                    "iscrowd": 0,
                    "score": float(segment.get("score", 1.0)),
                }
            )
        annotations.append(
            {
                "image_id": image_id,
                "file_name": file_name,
                "segments_info": segments_info,
            }
        )

    payload = {
        "images": [
            {
                "id": int(item["image_id"]),
                "file_name": item["file_name"],
                "height": int(item["height"]),
                "width": int(item["width"]),
            }
            for item in predictions
        ],
        "annotations": annotations,
        "categories": categories,
    }
    json_path = output_dir / "panoptic.json"
    temporary_path = json_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(json_path)
    return json_path


def save_metrics(metrics: dict[str, Any], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_suffix(output_file.suffix + ".tmp")
    temporary_file.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_file.replace(output_file)
