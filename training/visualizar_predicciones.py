from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from PIL import Image

from data import rgb_to_ids


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_DATASET_ROOT = WORKSPACE_ROOT / "reto-1-cartografa-automatizada"
DEFAULT_ANNOTATIONS = PROJECT_ROOT / "data_panoptic_stuff/annotations/panoptic.json"
DEFAULT_SPLITS = PROJECT_ROOT / "data_panoptic_stuff/splits.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "training/runs/visualizations_stuff"
DEFAULT_PREDICTIONS = {
    "mask2former": PROJECT_ROOT
    / "training/runs/mask2former_stuff_reentrenado/test_predictions/panoptic.json",
    "efficientps_original": PROJECT_ROOT
    / "training/runs/efficientps_original_desde_cero/test_predictions/panoptic.json",
}
CLASS_COLORS = {
    1: np.array([255, 140, 0], dtype=np.uint8),
    2: np.array([30, 144, 255], dtype=np.uint8),
    3: np.array([50, 205, 50], dtype=np.uint8),
    4: np.array([255, 215, 0], dtype=np.uint8),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Guarda overlays y una hoja de comparación de las predicciones "
            "panópticas del split test."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument(
        "--model",
        choices=(
            "both",
            "mask2former",
            "efficientps_original",
        ),
        default="both",
    )
    parser.add_argument(
        "--mask2former-predictions",
        type=Path,
        default=DEFAULT_PREDICTIONS["mask2former"],
    )
    parser.add_argument(
        "--efficientps-original-predictions",
        type=Path,
        default=DEFAULT_PREDICTIONS["efficientps_original"],
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-images",
        type=int,
        help="Limita la vista (por defecto procesa todas las imágenes de test).",
    )
    args = parser.parse_args()
    if args.max_images is not None and args.max_images < 1:
        parser.error("--max-images debe ser un entero positivo.")
    return args


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"No existe el archivo requerido: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_panoptic_map(
    annotation_file: Path,
    annotation: dict[str, Any],
) -> np.ndarray:
    mask_path = annotation_file.parent / annotation["file_name"]
    if not mask_path.is_file():
        raise FileNotFoundError(f"No existe la máscara panóptica: {mask_path}")
    with Image.open(mask_path) as source:
        return rgb_to_ids(np.asarray(source.convert("RGB")))


def category_map_for(
    segment_ids: np.ndarray,
    segments_info: list[dict[str, Any]],
    categories: dict[int, dict[str, Any]],
) -> np.ndarray:
    category_map = np.zeros(segment_ids.shape, dtype=np.uint16)
    for segment in segments_info:
        segment_id = int(segment["id"])
        category_id = int(segment["category_id"])
        if category_id not in categories:
            raise ValueError(f"Categoría {category_id} no está declarada.")
        category_map[segment_ids == segment_id] = category_id
    return category_map


def segment_boundaries(segment_ids: np.ndarray) -> np.ndarray:
    edges = np.zeros(segment_ids.shape, dtype=bool)
    different = segment_ids[:, 1:] != segment_ids[:, :-1]
    relevant = different & (
        (segment_ids[:, 1:] != 0) | (segment_ids[:, :-1] != 0)
    )
    edges[:, 1:] |= relevant
    edges[:, :-1] |= relevant

    different = segment_ids[1:, :] != segment_ids[:-1, :]
    relevant = different & (
        (segment_ids[1:, :] != 0) | (segment_ids[:-1, :] != 0)
    )
    edges[1:, :] |= relevant
    edges[:-1, :] |= relevant
    return edges


def make_overlay(
    image: np.ndarray,
    segment_ids: np.ndarray,
    category_map: np.ndarray,
) -> np.ndarray:
    if image.shape[:2] != segment_ids.shape or category_map.shape != segment_ids.shape:
        raise ValueError("La imagen y las máscaras no tienen las mismas dimensiones.")

    overlay = image.astype(np.float32).copy()
    labeled = category_map != 0
    for category_id in np.unique(category_map[labeled]):
        color = CLASS_COLORS.get(int(category_id))
        if color is None:
            raise ValueError(f"No hay color asignado para category_id={category_id}.")
        mask = category_map == category_id
        overlay[mask] = 0.48 * overlay[mask] + 0.52 * color

    overlay[segment_boundaries(segment_ids)] = np.array([20, 20, 20], dtype=np.uint8)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def load_prediction_records(
    prediction_file: Path,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    payload = read_json(prediction_file)
    images = {int(item["id"]): item for item in payload.get("images", [])}
    annotations = {
        int(item["image_id"]): item for item in payload.get("annotations", [])
    }
    if not images or not annotations:
        raise ValueError(f"{prediction_file} no contiene imágenes/anotaciones.")
    if images.keys() != annotations.keys():
        raise ValueError(f"Imágenes y anotaciones no coinciden en {prediction_file}.")
    return images, annotations


def render_comparison(
    panels: list[tuple[str, np.ndarray]],
    image_id: int,
    file_name: str,
    categories: dict[int, dict[str, Any]],
    output_file: Path,
) -> None:
    figure, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5.8))
    if len(panels) == 1:
        axes = [axes]
    for axis, (title, displayed_image) in zip(axes, panels):
        axis.imshow(displayed_image)
        axis.set_title(title, fontsize=11)
        axis.axis("off")

    legend = [
        Patch(
            facecolor=CLASS_COLORS[category_id] / 255,
            edgecolor="black",
            label=str(category["name"]),
        )
        for category_id, category in sorted(categories.items())
        if category_id in CLASS_COLORS
    ]
    figure.suptitle(f"image_id={image_id} — {Path(file_name).name}", fontsize=13)
    figure.legend(
        handles=legend,
        loc="lower center",
        ncol=len(legend),
        frameon=False,
        bbox_to_anchor=(0.5, 0.015),
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.93))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=100, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    ground_truth = read_json(args.annotations)
    split_payload = read_json(args.splits)
    categories = {
        int(item["id"]): item for item in ground_truth.get("categories", [])
    }
    gt_images = {int(item["id"]): item for item in ground_truth.get("images", [])}
    gt_annotations = {
        int(item["image_id"]): item
        for item in ground_truth.get("annotations", [])
    }
    split = split_payload.get("splits", {}).get("test")
    if not split:
        raise ValueError(f"No existe un split test con imágenes en {args.splits}.")
    selected_ids = [int(item["image_id"]) for item in split]
    if args.max_images is not None:
        selected_ids = selected_ids[: args.max_images]

    models = (
        ["mask2former", "efficientps_original"]
        if args.model == "both"
        else [args.model]
    )
    prediction_paths = {
        "mask2former": args.mask2former_predictions,
        "efficientps_original": args.efficientps_original_predictions,
    }
    prediction_labels = {
        "mask2former": "Mask2Former",
        "efficientps_original": "EfficientPS original",
    }
    model_data: dict[str, tuple[Path, dict[int, dict[str, Any]]]] = {}
    for model_name in models:
        prediction_file = prediction_paths[model_name]
        prediction_images, prediction_annotations = load_prediction_records(
            prediction_file
        )
        missing_ids = set(selected_ids) - prediction_annotations.keys()
        if missing_ids:
            raise ValueError(
                f"Faltan predicciones para image_id={sorted(missing_ids)} "
                f"en {prediction_file}."
            )
        if set(prediction_images) != set(prediction_annotations):
            raise ValueError(f"Índices inconsistentes en {prediction_file}.")
        model_data[model_name] = (prediction_file, prediction_annotations)

    panels_by_image: list[tuple[int, str, list[tuple[str, np.ndarray]]]] = []
    for image_id in selected_ids:
        if image_id not in gt_images or image_id not in gt_annotations:
            raise ValueError(f"Falta image_id={image_id} en las anotaciones GT.")
        image_record = gt_images[image_id]
        image_path = args.dataset_root / image_record["file_name"]
        if not image_path.is_file():
            raise FileNotFoundError(f"No existe la imagen original: {image_path}")
        with Image.open(image_path) as source:
            image = np.asarray(source.convert("RGB"))

        gt_annotation = gt_annotations[image_id]
        gt_ids = load_panoptic_map(args.annotations, gt_annotation)
        gt_categories = category_map_for(
            gt_ids,
            gt_annotation["segments_info"],
            categories,
        )
        if image.shape[:2] != gt_ids.shape:
            raise ValueError(f"Dimensiones incompatibles para image_id={image_id}.")

        panels = [
            ("Imagen original", image),
            (
                "Ground truth",
                make_overlay(image, gt_ids, gt_categories),
            ),
        ]
        for model_name in models:
            prediction_file, prediction_annotations = model_data[model_name]
            annotation = prediction_annotations[image_id]
            pred_ids = load_panoptic_map(prediction_file, annotation)
            pred_categories = category_map_for(
                pred_ids,
                annotation["segments_info"],
                categories,
            )
            panels.append(
                (
                    prediction_labels[model_name],
                    make_overlay(image, pred_ids, pred_categories),
                )
            )

        per_image_file = (
            args.output_dir / args.model / f"image_{image_id:03d}_comparison.png"
        )
        render_comparison(
            panels,
            image_id,
            image_record["file_name"],
            categories,
            per_image_file,
        )
        panels_by_image.append((image_id, image_record["file_name"], panels))

    columns = len(panels_by_image[0][2])
    figure, axes = plt.subplots(
        len(panels_by_image),
        columns,
        figsize=(4.2 * columns, 3.45 * len(panels_by_image)),
        squeeze=False,
    )
    for row, (image_id, file_name, panels) in enumerate(panels_by_image):
        for column, (title, displayed_image) in enumerate(panels):
            axis = axes[row, column]
            axis.imshow(displayed_image)
            axis.axis("off")
            if row == 0:
                axis.set_title(title, fontsize=11)
            if column == 0:
                axis.text(
                    0.02,
                    0.98,
                    f"{image_id}: {Path(file_name).name}",
                    transform=axis.transAxes,
                    va="top",
                    color="white",
                    fontsize=9,
                    bbox={"facecolor": "black", "alpha": 0.65, "pad": 3},
                )

    legend = [
        Patch(
            facecolor=CLASS_COLORS[category_id] / 255,
            edgecolor="black",
            label=str(category["name"]),
        )
        for category_id, category in sorted(categories.items())
        if category_id in CLASS_COLORS
    ]
    figure.legend(
        handles=legend,
        loc="lower center",
        ncol=len(legend),
        frameon=False,
        bbox_to_anchor=(0.5, 0.002),
    )
    figure.suptitle(
        f"Predicciones panópticas — split test ({len(panels_by_image)} imágenes)",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0.015, 1, 0.985))
    sheet_path = args.output_dir / args.model / "contact_sheet.png"
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(sheet_path, dpi=75, bbox_inches="tight")
    plt.close(figure)

    print(f"Hoja de comparación: {sheet_path}")
    print(
        f"Comparaciones individuales: "
        f"{args.output_dir / args.model}/image_*_comparison.png"
    )


if __name__ == "__main__":
    main()
