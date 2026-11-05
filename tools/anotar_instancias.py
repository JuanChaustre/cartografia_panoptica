from __future__ import annotations

import argparse
import json
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk


CLASSES = {
    "Acera / área caminable": (1, (255, 220, 0), True),
    "Manzana construida": (2, (220, 45, 45), True),
    "Zona verde": (3, (20, 190, 80), True),
    "Carretera (stuff)": (4, (60, 120, 255), False),
    "Borrar / ignorar": (0, (0, 0, 0), False),
}

CATEGORIES = [
    {"id": 1, "name": "acera", "supercategory": "cartografia", "isthing": 1},
    {"id": 2, "name": "manzana_construida", "supercategory": "cartografia", "isthing": 1},
    {"id": 3, "name": "zona_verde", "supercategory": "cartografia", "isthing": 1},
    {"id": 4, "name": "carretera", "supercategory": "cartografia", "isthing": 0},
]

REFERENCE_CHANNELS = (
    (1, (255, 220, 0)),
    (2, (220, 45, 45)),
    (3, (20, 190, 80)),
    (4, (60, 120, 255)),
)


def ids_to_rgb(ids: np.ndarray) -> np.ndarray:
    """Encode panoptic ids as the RGB representation used by COCO Panoptic."""
    ids64 = ids.astype(np.uint64, copy=False)
    return np.stack(
        (
            ids64 % 256,
            (ids64 // 256) % 256,
            (ids64 // 65536) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)


def rgb_to_ids(rgb: np.ndarray) -> np.ndarray:
    rgb64 = rgb.astype(np.uint64, copy=False)
    return (
        rgb64[..., 0]
        + 256 * rgb64[..., 1]
        + 65536 * rgb64[..., 2]
    ).astype(np.uint32)


def annotation_for_image(
    image_id: int,
    panoptic_name: str,
    panoptic_ids: np.ndarray,
    category_by_segment: dict[int, int],
) -> dict:
    segments_info = []
    for segment_id, category_id in sorted(category_by_segment.items()):
        ys, xs = np.nonzero(panoptic_ids == segment_id)
        if not len(xs):
            continue
        x_min = int(xs.min())
        y_min = int(ys.min())
        x_max = int(xs.max())
        y_max = int(ys.max())
        segments_info.append(
            {
                "id": int(segment_id),
                "category_id": int(category_id),
                "area": int(len(xs)),
                "bbox": [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1],
                "iscrowd": 0,
            }
        )

    return {
        "image_id": image_id,
        "file_name": panoptic_name,
        "segments_info": segments_info,
    }


class PanopticAnnotator:
    def __init__(self, dataset_dir: Path, output_dir: Path) -> None:
        self.dataset_dir = dataset_dir
        self.image_dir = dataset_dir / "Dataset segmentacion"
        self.output_dir = output_dir
        self.panoptic_dir = output_dir / "annotations" / "panoptic"
        self.json_path = output_dir / "annotations" / "panoptic.json"
        self.panoptic_dir.mkdir(parents=True, exist_ok=True)
        self.json_path.parent.mkdir(parents=True, exist_ok=True)

        self.image_paths = sorted(
            self.image_dir.glob("recorte_*.png"),
            key=lambda path: int(path.stem.removeprefix("recorte_")),
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No se encontraron recortes en {self.image_dir}")

        self.annotations: dict[int, dict] = {}
        self.images: dict[int, dict] = {}
        for path in self.image_paths:
            with Image.open(path) as source:
                image_id = self._image_id(path)
                self.images[image_id] = {
                    "id": image_id,
                    "file_name": f"Dataset segmentacion/{path.name}",
                    "width": source.width,
                    "height": source.height,
                }
        if self.json_path.exists():
            saved = json.loads(self.json_path.read_text(encoding="utf-8"))
            self.annotations = {
                int(item["image_id"]): item for item in saved.get("annotations", [])
            }
            self.images = {
                int(item["id"]): item for item in saved.get("images", [])
            }

        self.index = 0
        self.image: Image.Image
        self.reference: np.ndarray | None = None
        self.panoptic_ids: np.ndarray
        self.category_by_segment: dict[int, int] = {}
        self.pending_points: list[tuple[int, int]] = []
        self.scale = 1.0
        self.photo: ImageTk.PhotoImage | None = None
        self.dirty = False

        self.root = tk.Tk()
        self.root.title("Anotación panóptica de cartografía")
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)
        self.selected_class = tk.StringVar(value=next(iter(CLASSES)))
        self.status = tk.StringVar()
        self._build_ui()
        self.load_image(0)

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(fill="x")
        ttk.Label(toolbar, text="Categoría:").pack(side="left")
        ttk.Combobox(
            toolbar,
            textvariable=self.selected_class,
            values=list(CLASSES),
            state="readonly",
            width=28,
        ).pack(side="left", padx=6)
        ttk.Button(toolbar, text="Aplicar polígono (Enter)", command=self.apply_polygon).pack(
            side="left", padx=4
        )
        ttk.Button(toolbar, text="Deshacer puntos", command=self.undo_points).pack(
            side="left", padx=4
        )
        ttk.Button(toolbar, text="Limpiar imagen", command=self.clear_image).pack(
            side="left", padx=4
        )
        ttk.Button(toolbar, text="Guardar (Ctrl+S)", command=self.save_image).pack(
            side="left", padx=4
        )
        ttk.Button(toolbar, text="Anterior", command=lambda: self.load_image(self.index - 1)).pack(
            side="right", padx=4
        )
        ttk.Button(toolbar, text="Siguiente", command=lambda: self.load_image(self.index + 1)).pack(
            side="right", padx=4
        )

        self.canvas = tk.Canvas(self.root, background="#222222", highlightthickness=0)
        self.canvas.pack(padx=8, pady=4)
        self.canvas.bind("<Button-1>", self.add_point)
        self.canvas.bind("<Button-3>", lambda _: self.apply_polygon())
        self.canvas.bind("<Motion>", self.preview_polygon)
        self.root.bind("<Return>", lambda _: self.apply_polygon())
        self.root.bind("<Control-s>", lambda _: self.save_image())
        self.root.bind("<Left>", lambda _: self.load_image(self.index - 1))
        self.root.bind("<Right>", lambda _: self.load_image(self.index + 1))
        ttk.Label(self.root, textvariable=self.status, padding=8).pack(fill="x")

    def _image_id(self, path: Path) -> int:
        return int(path.stem.removeprefix("recorte_"))

    def load_image(self, index: int) -> None:
        if index < 0 or index >= len(self.image_paths):
            return
        if self.dirty or self.pending_points:
            from tkinter import messagebox

            decision = messagebox.askyesnocancel(
                "Cambios sin guardar",
                "¿Guardar la anotación actual antes de cambiar de imagen?",
            )
            if decision is None:
                return
            if decision:
                if self.pending_points:
                    self.apply_polygon()
                self.save_image()
        self.index = index
        path = self.image_paths[index]
        with Image.open(path) as source:
            self.image = source.convert("RGB")
        image_id = self._image_id(path)
        self.reference = None
        reference_path = self.image_dir / f"ground_truth_recorte_{image_id}.npy"
        if reference_path.exists():
            self.reference = np.load(reference_path, allow_pickle=False)
            if self.reference.shape != (self.image.height, self.image.width, 4):
                raise ValueError(f"Dimensiones de máscara inesperadas: {reference_path}")

        self.panoptic_ids = np.zeros((self.image.height, self.image.width), dtype=np.uint32)
        self.category_by_segment = {}
        panoptic_path = self.panoptic_dir / f"{path.stem}.png"
        saved_annotation = self.annotations.get(image_id)
        if panoptic_path.exists() and saved_annotation:
            with Image.open(panoptic_path) as saved_mask:
                self.panoptic_ids = rgb_to_ids(np.asarray(saved_mask.convert("RGB")))
            self.category_by_segment = {
                int(segment["id"]): int(segment["category_id"])
                for segment in saved_annotation["segments_info"]
            }

        self.pending_points.clear()
        self.dirty = False
        self._render()
        saved_count = len(self.annotations)
        self.status.set(
            f"{path.name} ({index + 1}/{len(self.image_paths)}) | "
            f"anotadas: {saved_count}/{len(self.image_paths)} | "
            "clic izquierdo: puntos; Enter o clic derecho: cerrar polígono."
        )

    def _render(self) -> None:
        display = self.image.copy().convert("RGBA")
        if self.reference is not None:
            for channel, color in REFERENCE_CHANNELS:
                layer = self.reference[:, :, channel - 1].astype(bool)
                hint_array = np.zeros((self.image.height, self.image.width, 4), dtype=np.uint8)
                hint_array[layer] = (*color, 48)
                display = Image.alpha_composite(display, Image.fromarray(hint_array))

        colors_by_category = {item[0]: item[1] for item in CLASSES.values() if item[0]}
        overlay = np.zeros((self.image.height, self.image.width, 4), dtype=np.uint8)
        for segment_id, category_id in self.category_by_segment.items():
            color = colors_by_category[category_id]
            overlay[self.panoptic_ids == segment_id] = (*color, 118)
        display = Image.alpha_composite(display, Image.fromarray(overlay))

        self.scale = min(1.0, 760 / self.image.width, 760 / self.image.height)
        width = max(1, round(self.image.width * self.scale))
        height = max(1, round(self.image.height * self.scale))
        display = display.convert("RGB").resize((width, height), Image.Resampling.NEAREST)
        self.photo = ImageTk.PhotoImage(display)
        self.canvas.configure(width=width, height=height)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self._draw_pending()

    def add_point(self, event: tk.Event) -> None:
        x = int(event.x / self.scale)
        y = int(event.y / self.scale)
        if 0 <= x < self.image.width and 0 <= y < self.image.height:
            self.pending_points.append((x, y))
            self._draw_pending()

    def _draw_pending(self, cursor: tuple[int, int] | None = None) -> None:
        self.canvas.delete("pending")
        if not self.pending_points:
            return
        _, color, _ = CLASSES[self.selected_class.get()]
        display_points = [(round(x * self.scale), round(y * self.scale)) for x, y in self.pending_points]
        for x, y in display_points:
            self.canvas.create_oval(
                x - 3, y - 3, x + 3, y + 3, fill="#ffffff", outline="#111111", tags="pending"
            )
        line_points = [coordinate for point in display_points for coordinate in point]
        if cursor is not None:
            line_points.extend((round(cursor[0] * self.scale), round(cursor[1] * self.scale)))
        if len(line_points) >= 4:
            self.canvas.create_line(
                *line_points,
                fill="#%02x%02x%02x" % color,
                width=2,
                tags="pending",
            )

    def preview_polygon(self, event: tk.Event) -> None:
        self._draw_pending((int(event.x / self.scale), int(event.y / self.scale)))

    def apply_polygon(self) -> None:
        if len(self.pending_points) < 3:
            return
        binary = Image.new("1", self.image.size, 0)
        ImageDraw.Draw(binary).polygon(self.pending_points, fill=1)
        selected = np.asarray(binary, dtype=bool)
        category_id, _, isthing = CLASSES[self.selected_class.get()]
        if category_id == 0:
            self.panoptic_ids[selected] = 0
            self.category_by_segment = {
                segment_id: existing_category
                for segment_id, existing_category in self.category_by_segment.items()
                if np.any(self.panoptic_ids == segment_id)
            }
        else:
            if isthing:
                segment_id = max(self.category_by_segment, default=0) + 1
            else:
                segment_id = next(
                    (
                        segment
                        for segment, existing_category in self.category_by_segment.items()
                        if existing_category == category_id
                    ),
                    max(self.category_by_segment, default=0) + 1,
                )
            self.panoptic_ids[selected] = segment_id
            self.category_by_segment[segment_id] = category_id
            self.category_by_segment = {
                segment: category
                for segment, category in self.category_by_segment.items()
                if np.any(self.panoptic_ids == segment)
            }
        self.pending_points.clear()
        self.dirty = True
        self._render()

    def undo_points(self) -> None:
        if self.pending_points:
            self.pending_points.pop()
            self._draw_pending()

    def clear_image(self) -> None:
        self.panoptic_ids.fill(0)
        self.category_by_segment.clear()
        self.pending_points.clear()
        self.dirty = True
        self._render()

    def save_image(self) -> None:
        path = self.image_paths[self.index]
        image_id = self._image_id(path)
        panoptic_name = f"panoptic/{path.stem}.png"
        panoptic_path = self.output_dir / "annotations" / panoptic_name
        Image.fromarray(ids_to_rgb(self.panoptic_ids), mode="RGB").save(panoptic_path)

        self.images[image_id] = {
            "id": image_id,
            "file_name": f"Dataset segmentacion/{path.name}",
            "width": self.image.width,
            "height": self.image.height,
        }
        self.annotations[image_id] = annotation_for_image(
            image_id,
            panoptic_name,
            self.panoptic_ids,
            self.category_by_segment,
        )
        payload = {
            "images": sorted(self.images.values(), key=lambda item: item["id"]),
            "annotations": sorted(self.annotations.values(), key=lambda item: item["image_id"]),
            "categories": CATEGORIES,
        }
        temporary_path = self.json_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary_path.replace(self.json_path)
        self.dirty = False
        self.pending_points.clear()
        self.status.set(
            f"Guardada {path.name} | {len(self.annotations)}/{len(self.image_paths)} imágenes anotadas."
        )

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Anota polígonos y exporta COCO Panoptic con instancias identificadas."
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
        default=Path("data_panoptic"),
        help="Carpeta de salida para annotations/panoptic.json y sus PNG.",
    )
    args = parser.parse_args()
    PanopticAnnotator(args.dataset.resolve(), args.output.resolve()).run()


if __name__ == "__main__":
    main()
