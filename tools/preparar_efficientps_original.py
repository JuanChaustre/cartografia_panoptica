from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_DATASET_ROOT = WORKSPACE_ROOT / "reto-1-cartografa-automatizada"
DEFAULT_ANNOTATIONS = PROJECT_ROOT / "data_panoptic_stuff/annotations/panoptic.json"
DEFAULT_SPLITS = PROJECT_ROOT / "data_panoptic_stuff/splits.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data_panoptic_stuff/efficientps_original"

SEMANTIC_LABELS = {
    "acera": 0,
    "carretera": 1,
    "manzana_construida": 2,
    "zona_verde": 3,
}
EFFICIENTPS_CATEGORY_IDS = {
    "acera": 7,
    "carretera": 8,
    "manzana_construida": 11,
    "zona_verde": 12,
}
INSTANCE_CATEGORY_IDS = {
    "manzana_construida": 1,
    "zona_verde": 2,
}


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


def encode_uncompressed_rle(mask: np.ndarray) -> dict[str, Any]:
    pixels = np.asarray(mask, dtype=np.uint8).reshape(-1, order="F")
    counts: list[int] = []
    current_value = 0
    run_length = 0
    for pixel in pixels:
        value = int(pixel)
        if value == current_value:
            run_length += 1
        else:
            counts.append(run_length)
            run_length = 1
            current_value = value
    counts.append(run_length)
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}


def segment_metadata(
    segment_id: int,
    category_id: int,
    mask: np.ndarray,
) -> dict[str, Any]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError(f"El segmento {segment_id} no contiene píxeles.")
    return {
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


def write_json(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"El archivo ya existe: {path}. Usa --overwrite para regenerar "
            "los archivos derivados."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def prepare_dataset(
    annotations_file: Path,
    splits_file: Path,
    dataset_root: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> dict[str, int]:
    panoptic = json.loads(annotations_file.read_text(encoding="utf-8"))
    split_payload = json.loads(splits_file.read_text(encoding="utf-8"))
    categories = {
        int(item["id"]): item for item in panoptic.get("categories", [])
    }
    category_ids_by_name = {
        item["name"]: category_id
        for category_id, item in categories.items()
    }
    required_names = set(SEMANTIC_LABELS)
    if set(category_ids_by_name) != required_names:
        raise ValueError(
            "Se requieren exactamente las categorías "
            f"{sorted(required_names)}; recibidas: {sorted(category_ids_by_name)}."
        )
    for name in required_names:
        expected_isthing = int(name in INSTANCE_CATEGORY_IDS)
        category = categories[category_ids_by_name[name]]
        if int(category["isthing"]) != expected_isthing:
            raise ValueError(
                f"La categoría {name!r} debe tener isthing={expected_isthing}."
            )

    images_by_id = {int(item["id"]): item for item in panoptic["images"]}
    annotations_by_id = {
        int(item["image_id"]): item for item in panoptic["annotations"]
    }
    all_image_ids = set(images_by_id)
    if all_image_ids != set(annotations_by_id):
        raise ValueError("Cada imagen debe tener exactamente una máscara panóptica.")

    split_names = ("train", "validation", "test")
    split_items = split_payload.get("splits", {})
    if set(split_names) - set(split_items):
        raise ValueError("splits.json debe contener train, validation y test.")
    split_ids = {
        name: [int(item["image_id"]) for item in split_items[name]]
        for name in split_names
    }
    flattened_ids = [image_id for name in split_names for image_id in split_ids[name]]
    if set(flattened_ids) != all_image_ids or len(flattened_ids) != len(set(flattened_ids)):
        raise ValueError("Las particiones deben cubrir cada imagen exactamente una vez.")

    segment_category_ids = {
        category_id: SEMANTIC_LABELS[name]
        for name, category_id in category_ids_by_name.items()
    }
    panoptic_category_ids = {
        category_id: EFFICIENTPS_CATEGORY_IDS[name]
        for name, category_id in category_ids_by_name.items()
    }
    thing_category_ids = {
        category_id: INSTANCE_CATEGORY_IDS[name]
        for name, category_id in category_ids_by_name.items()
        if name in INSTANCE_CATEGORY_IDS
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    statistics = {"images": 0, "thing_instances": 0}

    for split_name in split_names:
        coco_images: list[dict[str, Any]] = []
        coco_annotations: list[dict[str, Any]] = []
        panoptic_images: list[dict[str, Any]] = []
        panoptic_annotations: list[dict[str, Any]] = []
        annotation_id = 1
        seen_stems: set[str] = set()

        for image_id in split_ids[split_name]:
            image = images_by_id[image_id]
            annotation = annotations_by_id[image_id]
            file_name = image["file_name"]
            image_path = dataset_root / file_name
            mask_path = annotations_file.parent / annotation["file_name"]
            if not image_path.is_file():
                raise FileNotFoundError(f"No existe la imagen: {image_path}")
            if not mask_path.is_file():
                raise FileNotFoundError(f"No existe la máscara panóptica: {mask_path}")

            stem = Path(file_name).stem
            if stem in seen_stems:
                raise ValueError(f"Nombre de imagen duplicado en {split_name}: {stem}")
            seen_stems.add(stem)
            with Image.open(image_path) as source:
                actual_width, actual_height = source.size
            with Image.open(mask_path) as source:
                panoptic_ids = rgb_to_ids(np.asarray(source.convert("RGB")))
            height, width = panoptic_ids.shape
            if (width, height) != (int(image["width"]), int(image["height"])):
                raise ValueError(f"Dimensiones inconsistentes para {image_path}.")
            semantic_map = np.full((height, width), 255, dtype=np.uint8)
            panoptic_segments: list[dict[str, Any]] = []

            for segment in annotation["segments_info"]:
                segment_id = int(segment["id"])
                category_id = int(segment["category_id"])
                if category_id not in segment_category_ids:
                    raise ValueError(f"Categoría no declarada: {category_id}.")
                mask = panoptic_ids == segment_id
                if not mask.any():
                    raise ValueError(
                        f"El segmento {segment_id} no aparece en {mask_path.name}."
                    )
                semantic_map[mask] = segment_category_ids[category_id]
                panoptic_segments.append(
                    {
                        **segment_metadata(
                            segment_id,
                            panoptic_category_ids[category_id],
                            mask,
                        ),
                        "score": float(segment.get("score", 1.0)),
                    }
                )
                if category_id not in thing_category_ids:
                    continue
                metadata = segment_metadata(
                    segment_id,
                    thing_category_ids[category_id],
                    mask,
                )
                coco_annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": thing_category_ids[category_id],
                        "segmentation": encode_uncompressed_rle(mask),
                        "area": metadata["area"],
                        "bbox": metadata["bbox"],
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1
                statistics["thing_instances"] += 1

            semantic_path = output_dir / "semantic" / split_name / file_name
            semantic_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(semantic_map, mode="L").save(semantic_path)

            coco_images.append(
                {
                    "id": image_id,
                    "file_name": file_name,
                    "width": width,
                    "height": height,
                }
            )
            panoptic_file = f"{stem}_panoptic.png"
            panoptic_images.append(
                {
                    "id": stem,
                    "file_name": Path(file_name).name,
                    "width": width,
                    "height": height,
                }
            )
            panoptic_annotations.append(
                {
                    "image_id": stem,
                    "file_name": panoptic_file,
                    "segments_info": panoptic_segments,
                }
            )

            if split_name in ("validation", "test"):
                panoptic_path = output_dir / f"panoptic_{split_name}" / panoptic_file
                panoptic_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(ids_to_rgb(panoptic_ids), mode="RGB").save(
                    panoptic_path
                )
            statistics["images"] += 1

        instance_payload = {
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": [
                {
                    "id": INSTANCE_CATEGORY_IDS[name],
                    "name": name,
                    "supercategory": "thing",
                }
                for name in ("manzana_construida", "zona_verde")
            ],
        }
        write_json(
            output_dir / "annotations" / f"instances_{split_name}.json",
            instance_payload,
            overwrite=overwrite,
        )

        if split_name in ("validation", "test"):
            panoptic_payload = {
                "images": panoptic_images,
                "annotations": panoptic_annotations,
                "categories": [
                    {
                        "id": EFFICIENTPS_CATEGORY_IDS[name],
                        "name": name,
                        "isthing": int(name in INSTANCE_CATEGORY_IDS),
                        "color": [0, 0, 0],
                    }
                    for name in SEMANTIC_LABELS
                ],
            }
            write_json(
                output_dir / f"panoptic_{split_name}.json",
                panoptic_payload,
                overwrite=overwrite,
            )

    manifest = {
        "source_annotations": str(annotations_file.resolve()),
        "source_splits": str(splits_file.resolve()),
        "dataset_root": str(dataset_root.resolve()),
        "semantic_channels": [
            {"label": SEMANTIC_LABELS[name], "name": name}
            for name in SEMANTIC_LABELS
        ],
        "panoptic_category_ids": EFFICIENTPS_CATEGORY_IDS,
        "instance_category_ids": INSTANCE_CATEGORY_IDS,
        "counts": statistics,
    }
    write_json(output_dir / "manifest.json", manifest, overwrite=overwrite)
    return statistics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepara copias COCO de instancias y mapas semánticos para EfficientPS "
            "oficial sin modificar las anotaciones originales."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite actualizar los archivos derivados ya generados.",
    )
    args = parser.parse_args()
    counts = prepare_dataset(
        args.annotations.resolve(),
        args.splits.resolve(),
        args.dataset_root.resolve(),
        args.output.resolve(),
        overwrite=args.overwrite,
    )
    print(
        f"Dataset EfficientPS preparado: {counts['images']} imágenes, "
        f"{counts['thing_instances']} instancias thing."
    )
    print(f"Salida: {args.output.resolve()}")


if __name__ == "__main__":
    main()
