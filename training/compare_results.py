from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = {
    "Mask2Former": PROJECT_ROOT / "training/runs/mask2former_stuff_reentrenado",
    "EfficientPS original": PROJECT_ROOT
    / "training/runs/efficientps_original_desde_cero",
}
MODEL_NOTES = {
    "Mask2Former": (
        "Mejor resultado de esta comparación en IoU, F1, precisión, mAP y PQ.",
        "El ajuste parte de pesos COCO y solo 100 recortes; el backbone Swin "
        "y el decodificador exigen más VRAM que el modelo compacto.",
    ),
    "EfficientPS original": (
        "Implementación oficial con ramas de instancias y semántica y fusión "
        "panóptica del método.",
        "Stack legado adaptado y experimental; en estos resultados la detección "
        "de instancias queda muy por debajo de Mask2Former.",
    ),
}
METRIC_KEYS = {
    "IoU": ("pixel_metrics", "IoU_macro"),
    "mAP": ("instance_mAP", "mAP_50_95"),
    "F1": ("pixel_metrics", "F1_macro"),
    "Precisión": ("pixel_metrics", "precision_macro"),
    "PQ": ("panoptic_quality", "PQ"),
}
CLASS_METRIC_KEYS = {
    "IoU": "IoU",
    "F1": "F1",
    "Precisión": "precision",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera un reporte comparativo con métricas, pérdidas y "
            "predicciones panópticas frente al ground truth."
        )
    )
    parser.add_argument("--mask2former-run", type=Path, default=DEFAULT_RUNS["Mask2Former"])
    parser.add_argument(
        "--efficientps-run",
        type=Path,
        default=DEFAULT_RUNS["EfficientPS original"],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "training/runs/comparison_final",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=PROJECT_ROOT / "data_panoptic_stuff/annotations/panoptic.json",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=PROJECT_ROOT / "data_panoptic_stuff/splits.json",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT.parent / "reto-1-cartografa-automatizada",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        help="Limita las comparaciones visuales; por defecto usa todo el split test.",
    )
    args = parser.parse_args()
    if args.max_images is not None and args.max_images < 1:
        parser.error("--max-images debe ser un entero positivo.")
    return args


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"No existe el archivo requerido: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def nested_value(payload: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"Falta la métrica {'/'.join(keys)}.")
        value = value[key]
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"La métrica {'/'.join(keys)} no es finita.")
    return number


def load_history(path: Path) -> list[dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(f"No existe el historial de pérdidas: {path}")
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        expected = {"epoch", "train_loss", "validation_loss"}
        if not expected.issubset(reader.fieldnames or []):
            raise ValueError(f"Columnas inesperadas en el historial: {path}")
        history = [
            {key: float(row[key]) for key in expected}
            for row in reader
        ]
    if not history:
        raise ValueError(f"El historial no contiene épocas: {path}")
    if any(not math.isfinite(value) for row in history for value in row.values()):
        raise ValueError(f"El historial contiene valores no finitos: {path}")
    return history


def format_percent(value: float | None) -> str:
    return "N/D" if value is None else f"{100 * value:.2f}%"


def render_loss_curves(
    histories: dict[str, list[dict[str, float]]],
    output_file: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
    for column, model_name in enumerate(histories):
        history = histories[model_name]
        epochs = [int(item["epoch"]) for item in history]
        train_axis = axes[0, column]
        validation_axis = axes[1, column]
        train_axis.plot(
            epochs,
            [item["train_loss"] for item in history],
            color="#2878b5",
            marker=".",
        )
        validation_axis.plot(
            epochs,
            [item["validation_loss"] for item in history],
            color="#d9534f",
            marker=".",
        )
        train_axis.set_title(f"{model_name} — entrenamiento")
        train_axis.set_ylabel("Pérdida train")
        train_axis.grid(True, alpha=0.25)
        validation_axis.set_title(f"{model_name} — validación")
        validation_axis.set_xlabel("Época")
        validation_axis.set_ylabel("Pérdida validation")
        validation_axis.grid(True, alpha=0.25)
    figure.suptitle("Curvas de pérdida por modelo")
    figure.text(
        0.5,
        0.01,
        "Cada panel tiene escala propia; las pérdidas entre arquitecturas no son comparables directamente.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.94))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=160, bbox_inches="tight")
    plt.close(figure)


def render_metrics_chart(
    metrics: dict[str, dict[str, float | None]],
    output_file: Path,
) -> None:
    figure, (main_axis, map_axis) = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        gridspec_kw={"height_ratios": [3, 1.4]},
    )
    model_names = list(metrics)
    main_names = ["IoU", "F1", "Precisión", "PQ"]
    x = np.arange(len(main_names))
    width = 0.34
    for index, model_name in enumerate(model_names):
        values = [metrics[model_name][name] for name in main_names]
        bars = main_axis.bar(
            x + (index - (len(model_names) - 1) / 2) * width,
            [value if value is not None else 0.0 for value in values],
            width,
            label=model_name,
        )
        for bar, value in zip(bars, values):
            main_axis.annotate(
                format_percent(value),
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    main_axis.set_xticks(x, main_names)
    main_axis.set_ylim(0, 1.12)
    main_axis.set_ylabel("Puntuación")
    main_axis.set_title("Métricas globales del conjunto test")
    main_axis.grid(axis="y", alpha=0.25)
    main_axis.legend()

    for index, model_name in enumerate(model_names):
        value = metrics[model_name]["mAP"]
        bar = map_axis.bar(
            index,
            value if value is not None else 0.0,
            width=0.52,
            label=model_name,
        )
        map_axis.annotate(
            format_percent(value),
            (bar[0].get_x() + bar[0].get_width() / 2, bar[0].get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
        )
    map_axis.set_xticks(range(len(model_names)), model_names)
    map_axis.set_ylim(0, 1.12)
    map_axis.set_ylabel("mAP")
    map_axis.set_title("mAP de instancias (IoU 0.50–0.95)")
    map_axis.grid(axis="y", alpha=0.25)

    figure.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=160, bbox_inches="tight")
    plt.close(figure)


def run_visual_comparison(args: argparse.Namespace, output_dir: Path) -> Path:
    script = Path(__file__).with_name("visualizar_predicciones.py")
    prediction_output = output_dir / "predictions"
    command = [
        sys.executable,
        str(script),
        "--dataset-root",
        str(args.dataset_root.resolve()),
        "--annotations",
        str(args.annotations.resolve()),
        "--splits",
        str(args.splits.resolve()),
        "--model",
        "both",
        "--mask2former-predictions",
        str((args.mask2former_run / "test_predictions/panoptic.json").resolve()),
        "--efficientps-original-predictions",
        str((args.efficientps_run / "test_predictions/panoptic.json").resolve()),
        "--output-dir",
        str(prediction_output.resolve()),
    ]
    if args.max_images is not None:
        command.extend(["--max-images", str(args.max_images)])
    subprocess.run(command, check=True)
    return prediction_output / "both/contact_sheet.png"


def main() -> None:
    args = parse_args()
    run_dirs = {
        "Mask2Former": args.mask2former_run.resolve(),
        "EfficientPS original": args.efficientps_run.resolve(),
    }
    results: dict[str, dict[str, Any]] = {}
    histories: dict[str, list[dict[str, float]]] = {}
    metrics: dict[str, dict[str, float | None]] = {}

    for model_name, run_dir in run_dirs.items():
        metrics_path = run_dir / "test_metrics.json"
        payload = read_json(metrics_path)
        if payload.get("split") != "test":
            raise ValueError(f"{metrics_path} no contiene métricas del split test.")
        results[model_name] = payload
        histories[model_name] = load_history(run_dir / "loss_history.csv")
        metrics[model_name] = {
            name: nested_value(payload, path)
            for name, path in METRIC_KEYS.items()
        }

    image_counts = {int(payload.get("image_count", 0)) for payload in results.values()}
    if len(image_counts) != 1 or 0 in image_counts:
        raise ValueError("Los modelos no fueron evaluados en el mismo split test.")
    categories_by_model = [
        payload["pixel_metrics"]["per_category"] for payload in results.values()
    ]
    if set(categories_by_model[0]) != set(categories_by_model[1]):
        raise ValueError("Los modelos no tienen las mismas categorías evaluadas.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    loss_chart = args.output_dir / "loss_curves.png"
    metrics_chart = args.output_dir / "metrics_comparison.png"
    render_loss_curves(histories, loss_chart)
    render_metrics_chart(metrics, metrics_chart)
    contact_sheet = run_visual_comparison(args, args.output_dir)

    metric_rows: list[dict[str, str]] = []
    for model_name, values in metrics.items():
        advantage, limitation = MODEL_NOTES[model_name]
        metric_rows.append(
            {
                "Modelo": model_name,
                **{
                    name: "N/D" if value is None else f"{value:.6f}"
                    for name, value in values.items()
                },
                "Ventajas observadas": advantage,
                "Limitaciones": limitation,
            }
        )
    csv_path = args.output_dir / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)

    category_rows = []
    for category_id, mask2former_category in categories_by_model[0].items():
        category_name = mask2former_category["name"]
        category_rows.append(
            {
                "Categoría": category_name,
                **{
                    f"{model_name} {metric_name}": format_percent(
                        results[model_name]["pixel_metrics"]["per_category"][
                            category_id
                        ][metric_key]
                    )
                    for model_name in run_dirs
                    for metric_name, metric_key in CLASS_METRIC_KEYS.items()
                },
            }
        )

    overall_headers = ["Modelo", *METRIC_KEYS.keys(), "Ventajas observadas", "Limitaciones"]
    overall_table = [
        "| " + " | ".join(overall_headers) + " |",
        "| " + " | ".join("---" for _ in overall_headers) + " |",
    ]
    overall_table.extend(
        "| "
        + " | ".join(
            [
                row["Modelo"],
                *[
                    format_percent(metrics[row["Modelo"]][key])
                    for key in METRIC_KEYS
                ],
                row["Ventajas observadas"],
                row["Limitaciones"],
            ]
        )
        + " |"
        for row in metric_rows
    )
    category_headers = list(category_rows[0])
    category_table = [
        "| " + " | ".join(category_headers) + " |",
        "| " + " | ".join("---" for _ in category_headers) + " |",
    ]
    category_table.extend(
        "| " + " | ".join(row[key] for key in category_headers) + " |"
        for row in category_rows
    )

    report_path = args.output_dir / "comparison.md"
    report_path.write_text(
        "# Comparación de Mask2Former y EfficientPS original\n\n"
        f"Evaluación sobre **{next(iter(image_counts))} imágenes del split test**. "
        "IoU, F1 y precisión son promedios macro por píxel; mAP es AP de "
        "máscaras thing en IoU 0.50:0.95; PQ es calidad panóptica. Los valores "
        "se expresan como porcentaje. `N/D` indica que no se pudo calcular.\n\n"
        "## Métricas globales\n\n"
        + "\n".join(overall_table)
        + "\n\n## Métricas por clase\n\n"
        "IoU, F1 y precisión por píxel; se excluyen los píxeles void.\n\n"
        + "\n".join(category_table)
        + "\n\n## Comparación de métricas\n\n"
        f"![Gráfica de métricas](./{metrics_chart.name})\n\n"
        "## Pérdidas de entrenamiento y validación\n\n"
        "Entrenamiento y validación tienen paneles independientes para que los "
        "picos de una curva no oculten la otra; las pérdidas entre arquitecturas "
        "no son directamente comparables.\n\n"
        f"![Curvas de pérdida](./{loss_chart.name})\n\n"
        "## Predicciones frente al ground truth\n\n"
        "La hoja muestra, para cada imagen, imagen original, ground truth y "
        "predicciones de ambos modelos. Los colores identifican clases; los "
        "bordes oscuros separan segmentos.\n\n"
        f"![Comparación visual del split test](./{contact_sheet.relative_to(args.output_dir)})\n\n"
        f"Se guardaron también comparaciones por imagen en "
        f"`{contact_sheet.parent.relative_to(args.output_dir)}/image_*_comparison.png`.\n\n"
        "## Interpretación\n\n"
        "Mask2Former presenta mejores métricas globales en esta ejecución, "
        "incluida la detección de instancias (mAP). EfficientPS original "
        "mantiene valor como comparación de la implementación oficial, pero "
        "sus resultados dependen del port experimental para Ubuntu/GPU y de "
        "una configuración reducida para la memoria disponible. La escala de "
        "pérdida no debe usarse para decidir qué modelo es mejor.\n",
        encoding="utf-8",
    )
    print(f"Reporte: {report_path}")
    print(f"CSV de métricas: {csv_path}")
    print(f"Gráfica de métricas: {metrics_chart}")
    print(f"Gráfica de pérdidas: {loss_chart}")
    print(f"Comparación visual: {contact_sheet}")


if __name__ == "__main__":
    main()
