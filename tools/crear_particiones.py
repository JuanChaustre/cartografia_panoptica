from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def create_splits(
    annotation_file: Path,
    output_file: Path,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, object]:
    test_ratio = 1.0 - train_ratio - validation_ratio
    if train_ratio <= 0 or validation_ratio <= 0 or test_ratio <= 0:
        raise ValueError("Las proporciones de train/validation/test deben ser positivas.")

    payload = json.loads(annotation_file.read_text(encoding="utf-8"))
    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    if not images:
        raise ValueError(f"No hay imágenes en {annotation_file}.")

    image_ids = [int(image["id"]) for image in images]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("El archivo panóptico contiene IDs de imagen duplicados.")
    annotation_image_ids = {int(item["image_id"]) for item in annotations}
    if annotation_image_ids != set(image_ids):
        raise ValueError(
            "Cada imagen debe tener exactamente una anotación panóptica antes "
            "de crear las particiones."
        )

    shuffled_ids = image_ids.copy()
    random.Random(seed).shuffle(shuffled_ids)
    train_count = round(len(shuffled_ids) * train_ratio)
    validation_count = round(len(shuffled_ids) * validation_ratio)
    if train_count == 0 or validation_count == 0 or train_count + validation_count >= len(shuffled_ids):
        raise ValueError("El dataset es demasiado pequeño para estas proporciones.")

    splits = {
        "train": shuffled_ids[:train_count],
        "validation": shuffled_ids[train_count : train_count + validation_count],
        "test": shuffled_ids[train_count + validation_count :],
    }
    all_split_ids = [image_id for ids in splits.values() for image_id in ids]
    if len(all_split_ids) != len(image_ids) or set(all_split_ids) != set(image_ids):
        raise RuntimeError("La partición generada no conserva todas las imágenes exactamente una vez.")

    image_name_by_id = {int(item["id"]): item["file_name"] for item in images}
    result: dict[str, object] = {
        "schema_version": 1,
        "seed": seed,
        "split_method": "seeded_random_by_image",
        "ratios": {
            "train": train_ratio,
            "validation": validation_ratio,
            "test": test_ratio,
        },
        "splits": {
            name: [
                {"image_id": image_id, "file_name": image_name_by_id[image_id]}
                for image_id in ids
            ]
            for name, ids in splits.items()
        },
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_suffix(output_file.suffix + ".tmp")
    temporary_file.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_file.replace(output_file)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crea particiones reproducibles train/validation/test por imagen."
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data_panoptic_auto/annotations/panoptic.json"),
        help="Archivo COCO Panoptic completo.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data_panoptic_auto/splits.json"),
        help="Archivo JSON de salida con IDs y nombres de imagen.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = create_splits(
        args.annotations.resolve(),
        args.output.resolve(),
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )
    splits = result["splits"]
    print(
        "Particiones guardadas: "
        + ", ".join(f"{name}={len(items)}" for name, items in splits.items())
    )
    print(f"Archivo: {args.output.resolve()}")
    print(f"Semilla: {args.seed}")


if __name__ == "__main__":
    main()
