from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from anotar_instancias import rgb_to_ids


def validate(annotation_file: Path, dataset_root: Path) -> list[str]:
    problems: list[str] = []
    payload = json.loads(annotation_file.read_text(encoding="utf-8"))
    categories = {int(item["id"]): item for item in payload.get("categories", [])}
    annotations = payload.get("annotations", [])
    images = {int(item["id"]): item for item in payload.get("images", [])}
    annotation_by_image = {int(item["image_id"]): item for item in annotations}

    for image_id, image in images.items():
        annotation = annotation_by_image.get(image_id)
        if annotation is None:
            problems.append(f"Sin anotación panóptica: {image['file_name']}")
            continue
        panoptic_path = annotation_file.parent / annotation["file_name"]
        if not panoptic_path.is_file():
            problems.append(f"No existe la máscara: {panoptic_path}")
            continue
        with Image.open(panoptic_path) as source:
            ids = rgb_to_ids(np.asarray(source.convert("RGB")))
        if ids.shape != (int(image["height"]), int(image["width"])):
            problems.append(f"Dimensiones incorrectas en {panoptic_path.name}: {ids.shape}")
            continue

        listed_ids: set[int] = set()
        for segment in annotation.get("segments_info", []):
            segment_id = int(segment["id"])
            category_id = int(segment["category_id"])
            if segment_id == 0 or segment_id in listed_ids:
                problems.append(f"ID de segmento inválido o duplicado en {panoptic_path.name}")
            listed_ids.add(segment_id)
            if category_id not in categories:
                problems.append(
                    f"Categoría {category_id} no declarada en {panoptic_path.name}"
                )
            if not np.any(ids == segment_id):
                problems.append(
                    f"El segmento {segment_id} listado no aparece en {panoptic_path.name}"
                )
        mask_ids = {int(value) for value in np.unique(ids) if value != 0}
        if mask_ids != listed_ids:
            problems.append(
                f"Los IDs de {panoptic_path.name} no coinciden con segments_info "
                f"(PNG={sorted(mask_ids)}, JSON={sorted(listed_ids)})"
            )

        image_path = dataset_root / image["file_name"]
        if not image_path.is_file():
            problems.append(f"No existe la imagen: {image_path}")

    missing_count = len(images) - len(annotation_by_image)
    if missing_count:
        problems.append(f"Faltan anotaciones para {missing_count} imágenes.")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Valida anotaciones COCO Panoptic.")
    parser.add_argument("--dataset", type=Path, required=True, help="Raíz del dataset original.")
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data_panoptic/annotations/panoptic.json"),
        help="Archivo COCO Panoptic generado por anotar_instancias.py.",
    )
    args = parser.parse_args()

    problems = validate(args.annotations.resolve(), args.dataset.resolve())
    if problems:
        print(f"Validación fallida: {len(problems)} problema(s)")
        for problem in problems:
            print(f"- {problem}")
        raise SystemExit(1)

    payload = json.loads(args.annotations.resolve().read_text(encoding="utf-8"))
    print(
        f"Validación correcta: {len(payload['images'])} imágenes y "
        f"{sum(len(item['segments_info']) for item in payload['annotations'])} segmentos."
    )


if __name__ == "__main__":
    main()
