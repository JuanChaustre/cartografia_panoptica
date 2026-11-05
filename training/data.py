from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


def load_records(
    annotation_file: Path,
    dataset_root: Path,
    split_file: Path,
    split_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = json.loads(annotation_file.read_text(encoding="utf-8"))
    split_payload = json.loads(split_file.read_text(encoding="utf-8"))
    images = {int(item["id"]): item for item in payload["images"]}
    annotations = {int(item["image_id"]): item for item in payload["annotations"]}
    split_items = split_payload.get("splits", {}).get(split_name)
    if split_items is None:
        raise ValueError(f"No existe la partición {split_name!r} en {split_file}.")

    records: list[dict[str, Any]] = []
    for split_item in split_items:
        image_id = int(split_item["image_id"])
        if image_id not in images or image_id not in annotations:
            raise ValueError(f"El ID {image_id} de {split_name} no existe en el COCO Panoptic.")
        image = images[image_id]
        annotation = annotations[image_id]
        if split_item["file_name"] != image["file_name"]:
            raise ValueError(f"Nombre de imagen inconsistente para image_id={image_id}.")
        records.append(
            {
                "image_id": image_id,
                "file_name": image["file_name"],
                "image_path": dataset_root / image["file_name"],
                "height": int(image["height"]),
                "width": int(image["width"]),
                "panoptic_file_name": annotation["file_name"],
                "panoptic_path": annotation_file.parent / annotation["file_name"],
                "segments_info": annotation["segments_info"],
            }
        )
    if not records:
        raise ValueError(f"La partición {split_name!r} no contiene imágenes.")
    return records, payload["categories"]


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


class PanopticDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: list[dict[str, Any]],
        categories: list[dict[str, Any]],
        *,
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        self.records = records
        self.categories = {int(item["id"]): item for item in categories}
        self.augment = augment
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        with Image.open(record["image_path"]) as source:
            image = source.convert("RGB")
        with Image.open(record["panoptic_path"]) as source:
            panoptic_ids = rgb_to_ids(np.asarray(source.convert("RGB")))

        if image.size != (record["width"], record["height"]):
            raise ValueError(f"Dimensiones inesperadas en {record['image_path']}.")
        if panoptic_ids.shape != (record["height"], record["width"]):
            raise ValueError(f"Dimensiones inesperadas en {record['panoptic_path']}.")

        if self.augment:
            rng = random.Random(self.seed + self.epoch * len(self.records) + index)
            if rng.random() < 0.5:
                image = ImageOps.mirror(image)
                panoptic_ids = np.fliplr(panoptic_ids).copy()
            if rng.random() < 0.5:
                image = ImageOps.flip(image)
                panoptic_ids = np.flipud(panoptic_ids).copy()

        image_array = np.asarray(image, dtype=np.uint8)
        category_map = np.zeros(panoptic_ids.shape, dtype=np.uint8)
        for segment in record["segments_info"]:
            segment_id = int(segment["id"])
            category_id = int(segment["category_id"])
            if category_id not in self.categories:
                raise ValueError(f"Categoría {category_id} no declarada en el JSON.")
            category_map[panoptic_ids == segment_id] = category_id

        return {
            "image_id": record["image_id"],
            "file_name": record["file_name"],
            "image": image_array,
            "panoptic_ids": panoptic_ids,
            "category_map": category_map,
            "segments_info": record["segments_info"],
            "height": record["height"],
            "width": record["width"],
        }


def make_mask2former_targets(
    sample: dict[str, Any],
    categories: list[dict[str, Any]],
    image_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image = Image.fromarray(sample["image"]).resize(
        (image_size, image_size),
        resample=Image.Resampling.BILINEAR,
    )
    panoptic = Image.fromarray(sample["panoptic_ids"], mode="I").resize(
        (image_size, image_size),
        resample=Image.Resampling.NEAREST,
    )
    image_tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)

    category_to_label = {
        int(item["id"]): index for index, item in enumerate(categories)
    }
    isthing_by_category = {
        int(item["id"]): bool(item["isthing"]) for item in categories
    }
    resized_ids = np.asarray(panoptic, dtype=np.uint32)
    masks: list[np.ndarray] = []
    labels: list[int] = []
    stuff_masks: dict[int, np.ndarray] = {}
    for segment in sample["segments_info"]:
        segment_id = int(segment["id"])
        category_id = int(segment["category_id"])
        mask = resized_ids == segment_id
        if not mask.any():
            continue
        if isthing_by_category[category_id]:
            masks.append(mask)
            labels.append(category_to_label[category_id])
        elif category_id in stuff_masks:
            stuff_masks[category_id] |= mask
        else:
            stuff_masks[category_id] = mask.copy()

    for category_id, mask in stuff_masks.items():
        masks.append(mask)
        labels.append(category_to_label[category_id])

    if masks:
        mask_tensor = torch.from_numpy(np.stack(masks).astype(np.float32))
        label_tensor = torch.tensor(labels, dtype=torch.long)
    else:
        mask_tensor = torch.zeros((0, image_size, image_size), dtype=torch.float32)
        label_tensor = torch.zeros((0,), dtype=torch.long)
    return image_tensor, mask_tensor, label_tensor
