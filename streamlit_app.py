from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor


PROJECT_ROOT = Path(__file__).resolve().parent
TRAINING_ROOT = PROJECT_ROOT / "training"
sys.path.insert(0, str(TRAINING_ROOT))

MASK2FORMER_RUN = PROJECT_ROOT / "training/runs/mask2former_stuff_reentrenado"
EFFICIENTPS_RUN = PROJECT_ROOT / "training/runs/efficientps_original_desde_cero"
EFFICIENTPS_CHECKPOINT = EFFICIENTPS_RUN / "best_validation_loss.pth"
REPORT_DIR = PROJECT_ROOT / "training/runs/comparison_final"
ANNOTATIONS_FILE = PROJECT_ROOT / "data_panoptic_stuff/annotations/panoptic.json"
EFFICIENTPS_DATASET = PROJECT_ROOT / "data_panoptic_stuff/efficientps_original"
EFFICIENTPS_ENV = Path.home() / ".cache/cartografia-panoptica/efficientps-original"
EFFICIENTPS_CONFIG = PROJECT_ROOT / "training/efficientps_original/efficientps_cartografia.py"
EFFICIENTPS_SCRIPT = PROJECT_ROOT / "tools/predecir_efficientps_original.py"
CLASS_COLORS = {
    1: np.array([255, 140, 0], dtype=np.uint8),
    2: np.array([30, 144, 255], dtype=np.uint8),
    3: np.array([50, 205, 50], dtype=np.uint8),
    4: np.array([255, 215, 0], dtype=np.uint8),
}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 30_000_000
MASK2FORMER_IMAGE_SIZE = 256

st.set_page_config(
    page_title="Cartografía panóptica",
    page_icon="🛰️",
    layout="wide",
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"No existe el archivo requerido: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_categories(path_string: str) -> list[dict[str, Any]]:
    return read_json(Path(path_string))["categories"]


@st.cache_resource
def load_mask2former() -> tuple[Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation]:
    checkpoint_dir = MASK2FORMER_RUN / "best_model"
    if not (checkpoint_dir / "model.safetensors").is_file():
        raise FileNotFoundError(
            f"No encuentro los pesos de Mask2Former: {checkpoint_dir}. "
            "Descárgalos con `bash tools/descargar_modelos.sh`."
        )
    processor = Mask2FormerImageProcessor.from_pretrained(checkpoint_dir)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(checkpoint_dir)
    model.eval()
    return processor, model


def predict_mask2former(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    processor, model = load_mask2former()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # El modelo vuelve a CPU al terminar para liberar VRAM antes de EfficientPS.
    resized = image.resize(
        (MASK2FORMER_IMAGE_SIZE, MASK2FORMER_IMAGE_SIZE),
        resample=Image.Resampling.BILINEAR,
    )
    pixels = processor(
        images=np.asarray(resized),
        do_resize=False,
        return_tensors="pt",
    )["pixel_values"].to(device)
    categories = load_categories(str(ANNOTATIONS_FILE))
    label_ids_to_fuse = {
        index
        for index, category in enumerate(categories)
        if int(category["isthing"]) == 0
    }

    model.to(device)
    try:
        with torch.inference_mode():
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                output = model(pixel_values=pixels)
            result = processor.post_process_panoptic_segmentation(
                output,
                target_sizes=[(image.height, image.width)],
                label_ids_to_fuse=label_ids_to_fuse,
                threshold=0.5,
                mask_threshold=0.5,
                overlap_mask_area_threshold=0.8,
            )[0]
    finally:
        model.to("cpu")
        del pixels
        if device.type == "cuda":
            torch.cuda.empty_cache()

    category_map = np.zeros((image.height, image.width), dtype=np.uint8)
    segment_ids = result["segmentation"].cpu().numpy().astype(np.int32)
    category_ids = [int(category["id"]) for category in categories]
    for segment in result["segments_info"]:
        label_id = int(segment["label_id"])
        if label_id < 0 or label_id >= len(category_ids):
            raise ValueError(f"Label desconocido en salida Mask2Former: {label_id}")
        category_map[segment_ids == int(segment["id"])] = category_ids[label_id]
    return category_map, segment_ids


def predict_efficientps(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    # EfficientPS depende de Python/PyTorch antiguos y se ejecuta en otro proceso
    # para no mezclar sus bibliotecas con el entorno moderno de Streamlit.
    python = EFFICIENTPS_ENV / "bin/python"
    if not python.is_file():
        raise FileNotFoundError(f"No encuentro el entorno EfficientPS: {python}")
    if not EFFICIENTPS_CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"No encuentro el checkpoint de EfficientPS original: "
            f"{EFFICIENTPS_CHECKPOINT}. Descárgalo con "
            "`bash tools/descargar_modelos.sh`."
        )
    upstream_root = PROJECT_ROOT / "third_party/EfficientPS"
    with tempfile.TemporaryDirectory(prefix="cartografia-streamlit-") as directory:
        work_dir = Path(directory)
        image_path = work_dir / "uploaded_image.png"
        result_path = work_dir / "prediction.npz"
        image.save(image_path, format="PNG")
        environment = os.environ.copy()
        environment.update(
            {
                "CARTOGRAFIA_PROJECT_ROOT": str(PROJECT_ROOT),
                "PYTHONPATH": os.pathsep.join(
                    [str(upstream_root), str(TRAINING_ROOT)]
                ),
                "LD_LIBRARY_PATH": os.pathsep.join(
                    [
                        str(EFFICIENTPS_ENV / "cuda/lib64"),
                        str(EFFICIENTPS_ENV / "lib"),
                        str(EFFICIENTPS_ENV / "lib64"),
                        environment.get("LD_LIBRARY_PATH", ""),
                    ]
                ),
                "TORCH_CUDA_ARCH_LIST": "8.6",
            }
        )
        completed = subprocess.run(
            [
                str(python),
                str(EFFICIENTPS_SCRIPT),
                "--image",
                str(image_path),
                "--output",
                str(result_path),
                "--config",
                str(EFFICIENTPS_CONFIG),
                "--checkpoint",
                str(EFFICIENTPS_CHECKPOINT),
            ],
            cwd=upstream_root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=300,
            check=True,
        )
        if not result_path.is_file():
            raise RuntimeError(
                "EfficientPS terminó sin guardar la predicción. "
                f"Salida: {completed.stdout}\n{completed.stderr}"
            )
        with np.load(result_path) as result:
            return result["category_map"].copy(), result["segment_ids"].copy()


def crop_square(
    image: Image.Image,
    horizontal_position: int = 50,
    vertical_position: int = 50,
) -> Image.Image:
    if not 0 <= horizontal_position <= 100 or not 0 <= vertical_position <= 100:
        raise ValueError("La posición del recorte debe estar entre 0 y 100.")
    width, height = image.size
    # El entrenamiento usa recortes cuadrados; se conserva una región movible.
    side = min(width, height)
    left = round((width - side) * horizontal_position / 100)
    top = round((height - side) * vertical_position / 100)
    return image.crop((left, top, left + side, top + side))


def make_overlay(
    image: np.ndarray,
    category_map: np.ndarray,
    segment_ids: np.ndarray,
) -> np.ndarray:
    if image.shape[:2] != category_map.shape or category_map.shape != segment_ids.shape:
        raise ValueError("La imagen y la predicción tienen dimensiones incompatibles.")
    overlay = image.astype(np.float32).copy()
    for category_id in np.unique(category_map):
        if category_id == 0:
            continue
        color = CLASS_COLORS.get(int(category_id))
        if color is None:
            raise ValueError(f"No hay color asignado a la clase {category_id}.")
        mask = category_map == category_id
        overlay[mask] = 0.48 * overlay[mask] + 0.52 * color
    edges = np.zeros(category_map.shape, dtype=bool)
    horizontal = (segment_ids[:, 1:] != segment_ids[:, :-1]) & (
        (segment_ids[:, 1:] != 0) | (segment_ids[:, :-1] != 0)
    )
    edges[:, 1:] |= horizontal
    edges[:, :-1] |= horizontal
    vertical = (segment_ids[1:, :] != segment_ids[:-1, :]) & (
        (segment_ids[1:, :] != 0) | (segment_ids[:-1, :] != 0)
    )
    edges[1:, :] |= vertical
    edges[:-1, :] |= vertical
    overlay[edges] = [20, 20, 20]
    return np.clip(overlay, 0, 255).astype(np.uint8)


def render_markdown_report(report_path: Path) -> None:
    if not report_path.is_file():
        st.error(f"No existe el reporte generado: {report_path}")
        return
    content = report_path.read_text(encoding="utf-8")
    image_pattern = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
    cursor = 0
    # Renderiza las figuras como imágenes locales y el resto como Markdown.
    for match in image_pattern.finditer(content):
        markdown_chunk = content[cursor : match.start()].strip()
        if markdown_chunk:
            st.markdown(markdown_chunk)
        image_path = (report_path.parent / match.group(2)).resolve()
        if not image_path.is_relative_to(report_path.parent.resolve()):
            raise ValueError(f"Ruta de imagen fuera de la carpeta del reporte: {image_path}")
        if not image_path.is_file():
            st.warning(f"No se encontró la imagen del reporte: {image_path}")
        else:
            st.image(str(image_path), caption=match.group(1), use_container_width=True)
        cursor = match.end()
    trailing = content[cursor:].strip()
    if trailing:
        st.markdown(trailing)


def main() -> None:
    st.title("🛰️ Cartografía panóptica")
    st.caption(
        "Prueba Mask2Former y EfficientPS original con imágenes propias y "
        "consulta el reporte comparativo del conjunto de test."
    )
    prediction_tab, report_tab = st.tabs(
        ["Probar modelos", "Reporte comparativo"]
    )

    with prediction_tab:
        st.subheader("Sube una imagen satelital")
        uploaded_file = st.file_uploader(
            "Imagen PNG, JPEG, WEBP o TIFF",
            type=["png", "jpg", "jpeg", "webp", "tif", "tiff"],
        )
        selected_models = st.multiselect(
            "Modelos a ejecutar",
            ["Mask2Former", "EfficientPS original"],
            default=["Mask2Former", "EfficientPS original"],
        )
        st.caption(
            "EfficientPS original usa el checkpoint "
            f"`{EFFICIENTPS_CHECKPOINT}` y los datos "
            f"preparados desde `data_panoptic_stuff` "
            f"(`{EFFICIENTPS_DATASET}`)."
        )
        if not torch.cuda.is_available():
            st.warning(
                "CUDA no está disponible. Mask2Former se ejecutará en CPU; "
                "EfficientPS original requiere GPU y no podrá ejecutarse."
            )
            selected_models = [
                name for name in selected_models if name != "EfficientPS original"
            ]

        if uploaded_file is not None:
            if len(uploaded_file.getvalue()) > MAX_UPLOAD_BYTES:
                st.error("La imagen supera el límite de 20 MB.")
                return
            try:
                with Image.open(uploaded_file) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
            except Exception as error:
                st.error(f"No pude leer la imagen cargada: {error}")
                return
            if image.width * image.height > MAX_IMAGE_PIXELS:
                st.error("La imagen supera el límite de 30 megapíxeles.")
                return
            use_square_crop = st.checkbox(
                "Recortar a la proporción cuadrada del dataset",
                value=True,
                help=(
                    "Los recortes de entrenamiento son cuadrados (500×500). "
                    "El recorte puede reducir la distorsión en imágenes "
                    "rectangulares, pero elimina el área que queda fuera."
                ),
            )
            horizontal_position = 50
            vertical_position = 50
            if use_square_crop and image.width != image.height:
                horizontal_position = st.slider(
                    "Posición horizontal del recorte",
                    min_value=0,
                    max_value=100,
                    value=50,
                    help="0 conserva el borde izquierdo; 100 conserva el derecho.",
                )
                vertical_position = st.slider(
                    "Posición vertical del recorte",
                    min_value=0,
                    max_value=100,
                    value=50,
                    help="0 conserva el borde superior; 100 conserva el inferior.",
                )
            prediction_image = (
                crop_square(image, horizontal_position, vertical_position)
                if use_square_crop
                else image
            )
            if prediction_image.size != image.size:
                st.image(
                    prediction_image,
                    caption=(
                        f"Recorte para predicción — {prediction_image.width} × "
                        f"{prediction_image.height} (de {image.width} × {image.height})"
                    ),
                )
            else:
                st.image(
                    image,
                    caption=f"Imagen para predicción — {image.width} × {image.height}",
                )
            st.caption(
                "La imagen cuadrada se escala internamente a 256×256 para "
                "Mask2Former y 384×384 para EfficientPS original."
            )
            if not selected_models:
                st.info("Selecciona al menos un modelo para obtener predicciones.")
            elif st.button("Generar predicciones", type="primary"):
                image_array = np.asarray(prediction_image, dtype=np.uint8)
                columns = st.columns(len(selected_models))
                for column, model_name in zip(columns, selected_models):
                    with column:
                        st.subheader(model_name)
                        with st.spinner(f"Ejecutando {model_name}..."):
                            try:
                                if model_name == "Mask2Former":
                                    category_map, segment_ids = predict_mask2former(
                                        prediction_image
                                    )
                                else:
                                    category_map, segment_ids = predict_efficientps(
                                        prediction_image
                                    )
                                overlay = make_overlay(
                                    image_array,
                                    category_map,
                                    segment_ids,
                                )
                                st.image(
                                    overlay,
                                    caption=f"Predicción — {model_name}",
                                    use_container_width=True,
                                )
                                categories = load_categories(str(ANNOTATIONS_FILE))
                                detected = [
                                    category["name"]
                                    for category in categories
                                    if np.any(category_map == int(category["id"]))
                                ]
                                st.write(
                                    "Clases encontradas: "
                                    + (", ".join(detected) if detected else "ninguna")
                                )
                            except subprocess.CalledProcessError as error:
                                st.error(
                                    f"{model_name} terminó con error:\n"
                                    f"{error.stderr or error.stdout}"
                                )
                            except Exception as error:
                                st.exception(error)
        else:
            st.info("Carga una imagen para ver las predicciones.")

    with report_tab:
        st.subheader("Reporte comparativo")
        st.caption(
            "Resultados de las ejecuciones finales: "
            "mask2former_stuff_reentrenado y efficientps_original_desde_cero."
        )
        render_markdown_report(REPORT_DIR / "comparison.md")


if __name__ == "__main__":
    main()
