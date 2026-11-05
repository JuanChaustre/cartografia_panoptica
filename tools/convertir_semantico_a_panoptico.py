from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from anotar_instancias import CATEGORIES, annotation_for_image, ids_to_rgb


CHANNEL_BY_CATEGORY = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
}

# Apply from lowest to highest priority so later classes own shared pixels.
PRIORITY_LOW_TO_HIGH = (4, 1, 2, 3)
THING_CATEGORIES = (1, 2, 3)


def make_panoptic_mask(
    semantic_mask: np.ndarray,
    connectivity: int,
) -> tuple[np.ndarray, dict[int, int], int]:
    if semantic_mask.ndim != 3 or semantic_mask.shape[2] != 4:
        raise ValueError(
            "La máscara semántica debe tener forma (alto, ancho, 4); "
            f"se recibió {semantic_mask.shape}."
        )
    if not np.isin(semantic_mask, (0, 1, False, True)).all():
        raise ValueError("La máscara semántica solo puede contener valores binarios 0/1.")
    if connectivity not in (4, 8):
        raise ValueError("La conectividad debe ser 4 u 8.")

    semantic_mask = semantic_mask.astype(bool, copy=False)
    overlap_pixels = int(np.count_nonzero(semantic_mask.sum(axis=2) > 1))
    owner = np.zeros(semantic_mask.shape[:2], dtype=np.uint8)
    for category_id in PRIORITY_LOW_TO_HIGH:
        channel = CHANNEL_BY_CATEGORY[category_id]
        owner[semantic_mask[:, :, channel]] = category_id

    panoptic_ids = np.zeros(owner.shape, dtype=np.uint32)
    category_by_segment: dict[int, int] = {}
    next_segment_id = 1
    structure = ndimage.generate_binary_structure(2, 1 if connectivity == 4 else 2)

    for category_id in THING_CATEGORIES:
        labels, component_count = ndimage.label(owner == category_id, structure=structure)
        if not component_count:
            continue
        segment_ids = np.arange(
            next_segment_id,
            next_segment_id + component_count,
            dtype=np.uint32,
        )
        lookup = np.zeros(component_count + 1, dtype=np.uint32)
        lookup[1:] = segment_ids
        category_pixels = labels != 0
        panoptic_ids[category_pixels] = lookup[labels[category_pixels]]
        category_by_segment.update(
            (int(segment_id), category_id) for segment_id in segment_ids
        )
        next_segment_id += component_count

    road_pixels = owner == 4
    if road_pixels.any():
        panoptic_ids[road_pixels] = next_segment_id
        category_by_segment[next_segment_id] = 4

    return panoptic_ids, category_by_segment, overlap_pixels


def convert_dataset(
    dataset_dir: Path,
    output_dir: Path,
    connectivity: int = 8,
    overwrite: bool = False,
) -> tuple[int, dict[int, int], int]:
    dataset_dir = dataset_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir == dataset_dir or dataset_dir in output_dir.parents:
        raise ValueError("La carpeta de salida no puede estar dentro del dataset original.")

    image_dir = dataset_dir / "Dataset segmentacion"
    image_paths = sorted(
        image_dir.glob("recorte_*.png"),
        key=lambda path: int(path.stem.removeprefix("recorte_")),
    )
    if not image_paths:
        raise FileNotFoundError(f"No se encontraron imágenes en {image_dir}")

    image_by_id = {
        int(path.stem.removeprefix("recorte_")): path for path in image_paths
    }
    mask_by_id = {
        int(path.stem.removeprefix("ground_truth_recorte_")): path
        for path in image_dir.glob("ground_truth_recorte_*.npy")
    }
    if image_by_id.keys() != mask_by_id.keys():
        missing_masks = sorted(image_by_id.keys() - mask_by_id.keys())
        missing_images = sorted(mask_by_id.keys() - image_by_id.keys())
        raise ValueError(
            f"Los pares imagen/máscara no coinciden. Sin máscara: {missing_masks}; "
            f"sin imagen: {missing_images}."
        )

    annotations_dir = output_dir / "annotations"
    panoptic_dir = annotations_dir / "panoptic"
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"La salida ya existe y no está vacía: {output_dir}. "
            "Usa otra ruta o --overwrite para reemplazarla."
        )
    panoptic_dir.mkdir(parents=True, exist_ok=True)

    images: list[dict] = []
    annotations: list[dict] = []
    instance_counts = {category_id: 0 for category_id in CHANNEL_BY_CATEGORY}
    total_overlap_pixels = 0

    for image_id, image_path in sorted(image_by_id.items()):
        mask_path = mask_by_id[image_id]
        semantic_mask = np.load(mask_path, allow_pickle=False)
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            if semantic_mask.shape != (image.height, image.width, 4):
                raise ValueError(
                    f"Dimensiones incorrectas en {mask_path}: "
                    f"máscara={semantic_mask.shape}, imagen={(image.height, image.width)}."
                )

            panoptic_ids, category_by_segment, overlap_pixels = make_panoptic_mask(
                semantic_mask,
                connectivity,
            )
            total_overlap_pixels += overlap_pixels
            for category_id in category_by_segment.values():
                instance_counts[category_id] += 1

            panoptic_name = f"panoptic/{image_path.stem}.png"
            Image.fromarray(ids_to_rgb(panoptic_ids), mode="RGB").save(
                panoptic_dir / image_path.name
            )
            images.append(
                {
                    "id": image_id,
                    "file_name": f"Dataset segmentacion/{image_path.name}",
                    "width": image.width,
                    "height": image.height,
                }
            )
            annotations.append(
                annotation_for_image(
                    image_id,
                    panoptic_name,
                    panoptic_ids,
                    category_by_segment,
                )
            )

    payload = {
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES,
    }
    json_path = annotations_dir / "panoptic.json"
    temporary_path = json_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(json_path)
    return len(images), instance_counts, total_overlap_pixels


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convierte máscaras semánticas superpuestas a COCO Panoptic "
            "usando prioridad de clases y componentes conectados."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Carpeta raíz reto-1-cartografa-automatizada.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data_panoptic_auto"),
        help="Salida independiente; por defecto ./data_panoptic_auto.",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=(4, 8),
        default=8,
        help="Conectividad para separar instancias thing (por defecto: 8).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite reemplazar una carpeta de salida no vacía.",
    )
    args = parser.parse_args()

    image_count, instance_counts, overlap_pixels = convert_dataset(
        args.dataset,
        args.output,
        connectivity=args.connectivity,
        overwrite=args.overwrite,
    )
    print(f"Conversión completada: {image_count} imágenes.")
    print(f"Píxeles con dos o más clases en el origen: {overlap_pixels}.")
    print(
        "Segmentos generados por categoría: "
        + ", ".join(
            f"{category['name']}={instance_counts[int(category['id'])]}"
            for category in CATEGORIES
        )
    )
    print(f"COCO Panoptic: {(args.output / 'annotations' / 'panoptic.json').resolve()}")
    print("Revisa visualmente las máscaras generadas antes de entrenar.")


if __name__ == "__main__":
    main()
