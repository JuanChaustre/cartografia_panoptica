from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def rgb_to_ids(rgb: np.ndarray) -> np.ndarray:
    rgb64 = rgb.astype(np.uint64, copy=False)
    return (
        rgb64[..., 0]
        + 256 * rgb64[..., 1]
        + 65536 * rgb64[..., 2]
    ).astype(np.uint32)


def ids_to_rgb(ids: np.ndarray) -> np.ndarray:
    ids64 = ids.astype(np.uint64, copy=False)
    return np.stack(
        (ids64 % 256, (ids64 // 256) % 256, (ids64 // 65536) % 256),
        axis=-1,
    ).astype(np.uint8)


def _segment_metadata(
    segment_id: int,
    category_id: int,
    mask: np.ndarray,
    score: float | None = None,
) -> dict[str, Any]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError(f"No se puede generar metadata para el segmento vacío {segment_id}.")
    segment: dict[str, Any] = {
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
    }
    if score is not None:
        segment["score"] = float(score)
    return segment


def convert_annotations(
    source_file: Path,
    output_dir: Path,
    *,
    category_id: int,
    category_name: str,
) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"La carpeta de salida no está vacía: {output_dir}. "
            "Usa una nueva ruta para no sobrescribir resultados."
        )

    payload = json.loads(source_file.read_text(encoding="utf-8"))
    categories = {
        int(item["id"]): item for item in payload.get("categories", [])
    }
    if category_id not in categories:
        raise ValueError(f"No existe category_id={category_id} en {source_file}.")
    if categories[category_id]["name"] != category_name:
        raise ValueError(
            f"category_id={category_id} es {categories[category_id]['name']!r}, "
            f"no {category_name!r}."
        )

    source_annotations = source_file.parent
    output_annotations = output_dir / "annotations"
    transformed_payload = json.loads(json.dumps(payload))
    for category in transformed_payload["categories"]:
        if int(category["id"]) == category_id:
            category["isthing"] = 0

    annotations_by_image = {
        int(item["image_id"]): item
        for item in transformed_payload.get("annotations", [])
    }
    image_ids = {int(item["id"]) for item in transformed_payload.get("images", [])}
    if image_ids != set(annotations_by_image):
        raise ValueError("Cada imagen debe tener exactamente una anotación panóptica.")

    for image_id, annotation in annotations_by_image.items():
        source_mask = source_annotations / annotation["file_name"]
        if not source_mask.is_file():
            raise FileNotFoundError(f"No existe la máscara panóptica: {source_mask}")
        with Image.open(source_mask) as image:
            ids = rgb_to_ids(np.asarray(image.convert("RGB")))

        target_segments = [
            segment
            for segment in annotation["segments_info"]
            if int(segment["category_id"]) == category_id
        ]
        target_ids = {int(segment["id"]) for segment in target_segments}
        merged_segment_id = min(target_ids) if target_ids else None
        if merged_segment_id is not None and len(target_ids) > 1:
            ids[np.isin(ids, list(target_ids))] = merged_segment_id

        segment_info: list[dict[str, Any]] = []
        if merged_segment_id is not None:
            merged_mask = ids == merged_segment_id
            segment_info.append(
                _segment_metadata(merged_segment_id, category_id, merged_mask)
            )

        for segment in annotation["segments_info"]:
            segment_id = int(segment["id"])
            segment_category = int(segment["category_id"])
            if segment_category == category_id or segment_id == merged_segment_id:
                continue
            mask = ids == segment_id
            if not mask.any():
                continue
            segment_info.append(
                _segment_metadata(
                    segment_id,
                    segment_category,
                    mask,
                    score=segment.get("score"),
                )
            )

        annotation["segments_info"] = sorted(
            segment_info,
            key=lambda segment: int(segment["id"]),
        )
        destination_mask = output_annotations / annotation["file_name"]
        destination_mask.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(ids_to_rgb(ids), mode="RGB").save(destination_mask)

    output_annotations.mkdir(parents=True, exist_ok=True)
    json_path = output_annotations / "panoptic.json"
    temporary_path = json_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(transformed_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(json_path)

    split_file = source_file.parent.parent / "splits.json"
    if split_file.is_file():
        split_payload = json.loads(split_file.read_text(encoding="utf-8"))
        output_split = output_dir / "splits.json"
        temporary_split = output_split.with_suffix(".json.tmp")
        temporary_split.write_text(
            json.dumps(split_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_split.replace(output_split)
    return json_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convierte una categoría thing a stuff, fusionándola en un segmento "
            "por imagen sin modificar las anotaciones de origen."
        )
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data_panoptic_auto/annotations/panoptic.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data_panoptic_stuff"),
    )
    parser.add_argument("--category-id", type=int, default=1)
    parser.add_argument("--category-name", default="acera")
    args = parser.parse_args()

    output_file = convert_annotations(
        args.annotations.resolve(),
        args.output.resolve(),
        category_id=args.category_id,
        category_name=args.category_name,
    )
    print(f"Anotaciones convertidas: {output_file}")
    print(f"Splits: {args.output.resolve() / 'splits.json'}")


if __name__ == "__main__":
    main()
