from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


TRAINING_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_ROOT.parent
sys.path.insert(0, str(TRAINING_ROOT))

from data import load_records, rgb_to_ids
from metrics import evaluate_panoptic
from utils import save_metrics, save_predictions


CITYSCAPES_TO_PROJECT = {
    7: "acera",
    8: "carretera",
    11: "manzana_construida",
    12: "zona_verde",
}


def load_official_predictions(
    prediction_dir: Path,
    annotation_file: Path,
    dataset_root: Path,
    split_file: Path,
    split_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records, categories = load_records(
        annotation_file, dataset_root, split_file, split_name
    )
    records_by_stem: dict[str, dict[str, Any]] = {}
    for record in records:
        stem = Path(record["file_name"]).stem
        if stem in records_by_stem:
            raise ValueError(f"El nombre de imagen no es único: {stem}.")
        records_by_stem[stem] = record

    category_ids_by_name = {
        item["name"]: int(item["id"]) for item in categories
    }
    project_category_by_cityscapes_id = {
        cityscapes_id: category_ids_by_name[name]
        for cityscapes_id, name in CITYSCAPES_TO_PROJECT.items()
    }
    json_dir = prediction_dir / "tmp_json"
    mask_dir = prediction_dir / "tmp"
    if not json_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(
            "No encuentro las predicciones oficiales en "
            f"{prediction_dir}; se esperaba tmp/ y tmp_json/."
        )

    predictions: list[dict[str, Any]] = []
    seen_stems: set[str] = set()
    for json_path in sorted(json_dir.glob("*.json")):
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        for annotation in payload.get("annotations", []):
            stem = str(annotation["image_id"])
            if stem not in records_by_stem:
                raise ValueError(f"Predicción inesperada para {stem}.")
            if stem in seen_stems:
                raise ValueError(f"Predicción duplicada para {stem}.")
            record = records_by_stem[stem]
            mask_path = mask_dir / annotation["file_name"]
            with Image.open(mask_path) as source:
                official_ids = rgb_to_ids(np.asarray(source.convert("RGB")))
            if official_ids.shape != (record["height"], record["width"]):
                raise ValueError(f"Dimensiones inesperadas en {mask_path}.")

            project_ids = np.zeros(official_ids.shape, dtype=np.uint32)
            segments: list[dict[str, Any]] = []
            next_segment_id = 1
            for segment in annotation["segments_info"]:
                cityscapes_id = int(segment["category_id"])
                if cityscapes_id not in project_category_by_cityscapes_id:
                    raise ValueError(
                        f"EfficientPS devolvió category_id={cityscapes_id}; "
                        "no existe en el mapeo de cartografía."
                    )
                mask = official_ids == int(segment["id"])
                if not mask.any():
                    continue
                project_ids[mask] = next_segment_id
                segments.append(
                    {
                        "id": next_segment_id,
                        "category_id": project_category_by_cityscapes_id[
                            cityscapes_id
                        ],
                        "score": float(segment.get("score", 1.0)),
                    }
                )
                next_segment_id += 1

            predictions.append(
                {
                    "image_id": record["image_id"],
                    "file_name": record["file_name"],
                    "height": record["height"],
                    "width": record["width"],
                    "segmentation": project_ids,
                    "segments_info": segments,
                }
            )
            seen_stems.add(stem)

    if seen_stems != set(records_by_stem):
        missing = sorted(set(records_by_stem) - seen_stems)
        raise ValueError(f"Faltan predicciones para: {', '.join(missing)}.")
    return predictions, categories


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convierte la salida COCO/Cityscapes de EfficientPS oficial al "
            "formato del proyecto y calcula sus métricas de test."
        )
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        required=True,
        help="Carpeta tmp/ y tmp_json/ archivada por la evaluación oficial.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=PROJECT_ROOT / "data_panoptic_stuff/annotations/panoptic.json",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT.parent / "reto-1-cartografa-automatizada",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=PROJECT_ROOT / "data_panoptic_stuff/splits.json",
    )
    parser.add_argument("--split", default="test", choices=("validation", "test"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "training/runs/efficientps_original/test_predictions",
    )
    args = parser.parse_args()

    predictions, categories = load_official_predictions(
        args.prediction_dir.resolve(),
        args.annotations.resolve(),
        args.dataset_root.resolve(),
        args.splits.resolve(),
        args.split,
    )
    prediction_file = save_predictions(
        predictions, categories, args.output_dir.resolve()
    )
    metrics = evaluate_panoptic(
        args.annotations.resolve(),
        prediction_file,
        args.splits.resolve(),
        args.split,
    )
    metrics_path = args.output_dir.resolve().parent / f"{args.split}_metrics.json"
    save_metrics(metrics, metrics_path)
    print(f"Predicciones del proyecto: {prediction_file}")
    print(f"Métricas: {metrics_path}")
    print(
        json.dumps(
            {
                "IoU_macro": metrics["pixel_metrics"]["IoU_macro"],
                "precision_macro": metrics["pixel_metrics"]["precision_macro"],
                "F1_macro": metrics["pixel_metrics"]["F1_macro"],
                "mAP_50_95": metrics["instance_mAP"]["mAP_50_95"],
                "PQ": metrics["panoptic_quality"]["PQ"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
