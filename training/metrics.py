from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from data import rgb_to_ids


def _panoptic_to_categories(
    ids: np.ndarray,
    segments: list[dict[str, Any]],
) -> np.ndarray:
    categories = np.zeros(ids.shape, dtype=np.uint8)
    for segment in segments:
        categories[ids == int(segment["id"])] = int(segment["category_id"])
    return categories


def _load_panoptic_mask(
    root: Path,
    file_name: str,
) -> np.ndarray:
    with Image.open(root / file_name) as source:
        return rgb_to_ids(np.asarray(source.convert("RGB")))


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = np.count_nonzero(left & right)
    union = np.count_nonzero(left | right)
    return float(intersection / union) if union else 0.0


def _average_precision_101(
    predictions: list[tuple[float, bool]],
    ground_truth_count: int,
) -> float | None:
    if ground_truth_count == 0:
        return None
    if not predictions:
        return 0.0

    predictions.sort(key=lambda item: item[0], reverse=True)
    true_positives = np.cumsum([int(item[1]) for item in predictions], dtype=np.float64)
    false_positives = np.cumsum([int(not item[1]) for item in predictions], dtype=np.float64)
    recall = true_positives / ground_truth_count
    precision = true_positives / np.maximum(true_positives + false_positives, 1e-12)
    precision_envelope = np.maximum.accumulate(precision[::-1])[::-1]
    return float(
        np.mean(
            [
                precision_envelope[recall >= level].max()
                if np.any(recall >= level)
                else 0.0
                for level in np.linspace(0.0, 1.0, 101)
            ]
        )
    )


def _instance_average_precision(
    ground_truth_by_image: dict[int, list[tuple[int, np.ndarray]]],
    predictions_by_image: dict[int, list[tuple[int, float, np.ndarray]]],
    thing_category_ids: list[int],
) -> dict[str, Any]:
    thresholds = [round(value, 2) for value in np.arange(0.5, 0.951, 0.05)]
    per_category: dict[str, Any] = {}
    category_average_precisions: list[float] = []

    for category_id in thing_category_ids:
        category_gt = {
            image_id: [mask for label, mask in segments if label == category_id]
            for image_id, segments in ground_truth_by_image.items()
        }
        ground_truth_count = sum(len(masks) for masks in category_gt.values())
        category_predictions = [
            (image_id, score, mask)
            for image_id, segments in predictions_by_image.items()
            for label, score, mask in segments
            if label == category_id
        ]
        ap_by_threshold: dict[str, float | None] = {}
        for threshold in thresholds:
            matched = {image_id: set() for image_id in category_gt}
            ranked_results: list[tuple[float, bool]] = []
            for image_id, score, pred_mask in sorted(
                category_predictions,
                key=lambda item: item[1],
                reverse=True,
            ):
                gt_masks = category_gt.get(image_id, [])
                candidates = [
                    (index, _mask_iou(pred_mask, gt_mask))
                    for index, gt_mask in enumerate(gt_masks)
                    if index not in matched.get(image_id, set())
                ]
                if candidates:
                    best_index, best_iou = max(candidates, key=lambda item: item[1])
                    is_true_positive = best_iou >= threshold
                    if is_true_positive:
                        matched.setdefault(image_id, set()).add(best_index)
                else:
                    is_true_positive = False
                ranked_results.append((score, is_true_positive))

            ap_by_threshold[f"{threshold:.2f}"] = _average_precision_101(
                ranked_results,
                ground_truth_count,
            )

        valid_ap = [value for value in ap_by_threshold.values() if value is not None]
        category_ap = float(np.mean(valid_ap)) if valid_ap else None
        per_category[str(category_id)] = {
            "ground_truth_instances": ground_truth_count,
            "predicted_instances": len(category_predictions),
            "AP_by_IoU": ap_by_threshold,
            "mAP_50_95": category_ap,
        }
        if category_ap is not None:
            category_average_precisions.append(category_ap)

    return {
        "definition": (
            "Mask AP over thing categories only; IoU thresholds 0.50:0.05:0.95; "
            "101-point interpolated precision."
        ),
        "mAP_50_95": (
            float(np.mean(category_average_precisions))
            if category_average_precisions
            else None
        ),
        "per_category": per_category,
    }


def _panoptic_quality(
    ground_truth_by_image: dict[int, tuple[np.ndarray, dict[int, int]]],
    predictions_by_image: dict[int, tuple[np.ndarray, dict[int, int]]],
    category_ids: list[int],
) -> dict[str, Any]:
    totals = {
        category_id: {"tp": 0, "fp": 0, "fn": 0, "sum_iou": 0.0}
        for category_id in category_ids
    }
    for image_id, (gt_ids, gt_categories) in ground_truth_by_image.items():
        pred_ids, pred_categories = predictions_by_image[image_id]
        valid = gt_ids != 0
        gt_masks = {
            segment_id: (gt_ids == segment_id)
            for segment_id in gt_categories
        }
        pred_masks = {
            segment_id: ((pred_ids == segment_id) & valid)
            for segment_id in pred_categories
        }
        for category_id in category_ids:
            gt_segments = [
                (segment_id, mask)
                for segment_id, mask in gt_masks.items()
                if gt_categories[segment_id] == category_id and mask.any()
            ]
            pred_segments = [
                (segment_id, mask)
                for segment_id, mask in pred_masks.items()
                if pred_categories[segment_id] == category_id and mask.any()
            ]
            candidates: list[tuple[float, int, int]] = []
            for gt_segment_id, gt_mask in gt_segments:
                gt_area = int(gt_mask.sum())
                for pred_segment_id, pred_mask in pred_segments:
                    intersection = int(np.count_nonzero(gt_mask & pred_mask))
                    union = gt_area + int(pred_mask.sum()) - intersection
                    iou = intersection / union if union else 0.0
                    if iou > 0.5:
                        candidates.append((iou, gt_segment_id, pred_segment_id))

            used_gt: set[int] = set()
            used_pred: set[int] = set()
            for iou, gt_segment_id, pred_segment_id in sorted(candidates, reverse=True):
                if gt_segment_id in used_gt or pred_segment_id in used_pred:
                    continue
                used_gt.add(gt_segment_id)
                used_pred.add(pred_segment_id)
                totals[category_id]["tp"] += 1
                totals[category_id]["sum_iou"] += iou
            totals[category_id]["fn"] += len(gt_segments) - len(used_gt)
            totals[category_id]["fp"] += len(pred_segments) - len(used_pred)

    per_category: dict[str, Any] = {}
    values: list[float] = []
    for category_id, counts in totals.items():
        denominator = counts["tp"] + 0.5 * counts["fp"] + 0.5 * counts["fn"]
        pq = counts["sum_iou"] / denominator if denominator else None
        sq = counts["sum_iou"] / counts["tp"] if counts["tp"] else None
        rq = counts["tp"] / denominator if denominator else None
        per_category[str(category_id)] = {
            **counts,
            "PQ": pq,
            "SQ": sq,
            "RQ": rq,
        }
        if pq is not None:
            values.append(float(pq))
    return {
        "definition": "Panoptic Quality; same-category segment matches with IoU > 0.5.",
        "PQ": float(np.mean(values)) if values else None,
        "per_category": per_category,
    }


def evaluate_panoptic(
    ground_truth_file: Path,
    prediction_file: Path,
    split_file: Path,
    split_name: str,
) -> dict[str, Any]:
    gt_payload = json.loads(ground_truth_file.read_text(encoding="utf-8"))
    pred_payload = json.loads(prediction_file.read_text(encoding="utf-8"))
    split_payload = json.loads(split_file.read_text(encoding="utf-8"))
    selected_ids = {
        int(item["image_id"])
        for item in split_payload["splits"][split_name]
    }
    gt_annotations = {
        int(item["image_id"]): item
        for item in gt_payload["annotations"]
        if int(item["image_id"]) in selected_ids
    }
    pred_annotations = {
        int(item["image_id"]): item
        for item in pred_payload["annotations"]
        if int(item["image_id"]) in selected_ids
    }
    if gt_annotations.keys() != pred_annotations.keys():
        raise ValueError("Predicciones y ground truth no contienen los mismos image_id.")

    gt_root = ground_truth_file.parent
    pred_root = prediction_file.parent
    categories = {int(item["id"]): item for item in gt_payload["categories"]}
    category_ids = sorted(categories)
    thing_category_ids = [
        category_id
        for category_id in category_ids
        if int(categories[category_id]["isthing"]) == 1
    ]

    counts = {
        category_id: {"tp": 0, "fp": 0, "fn": 0}
        for category_id in category_ids
    }
    gt_instances: dict[int, list[tuple[int, np.ndarray]]] = {}
    pred_instances: dict[int, list[tuple[int, float, np.ndarray]]] = {}
    gt_panoptic: dict[int, tuple[np.ndarray, dict[int, int]]] = {}
    pred_panoptic: dict[int, tuple[np.ndarray, dict[int, int]]] = {}

    for image_id in sorted(selected_ids):
        gt_annotation = gt_annotations[image_id]
        pred_annotation = pred_annotations[image_id]
        gt_ids = _load_panoptic_mask(gt_root, gt_annotation["file_name"])
        pred_ids = _load_panoptic_mask(pred_root, pred_annotation["file_name"])
        if gt_ids.shape != pred_ids.shape:
            raise ValueError(
                f"Dimensiones distintas en image_id={image_id}: "
                f"GT={gt_ids.shape}, predicción={pred_ids.shape}."
            )
        gt_categories = {
            int(item["id"]): int(item["category_id"])
            for item in gt_annotation["segments_info"]
        }
        pred_categories = {
            int(item["id"]): int(item["category_id"])
            for item in pred_annotation["segments_info"]
        }
        gt_panoptic[image_id] = (gt_ids, gt_categories)
        pred_panoptic[image_id] = (pred_ids, pred_categories)

        gt_category_map = _panoptic_to_categories(gt_ids, gt_annotation["segments_info"])
        pred_category_map = _panoptic_to_categories(pred_ids, pred_annotation["segments_info"])
        valid = gt_ids != 0
        for category_id in category_ids:
            gt_mask = (gt_category_map == category_id) & valid
            pred_mask = (pred_category_map == category_id) & valid
            counts[category_id]["tp"] += int(np.count_nonzero(gt_mask & pred_mask))
            counts[category_id]["fp"] += int(np.count_nonzero(~gt_mask & pred_mask))
            counts[category_id]["fn"] += int(np.count_nonzero(gt_mask & ~pred_mask))

        gt_instances[image_id] = []
        for segment in gt_annotation["segments_info"]:
            category_id = int(segment["category_id"])
            if category_id in thing_category_ids:
                mask = gt_ids == int(segment["id"])
                if mask.any():
                    gt_instances[image_id].append((category_id, mask))

        pred_instances[image_id] = []
        for segment in pred_annotation["segments_info"]:
            category_id = int(segment["category_id"])
            if category_id in thing_category_ids:
                mask = pred_ids == int(segment["id"])
                if mask.any():
                    pred_instances[image_id].append(
                        (category_id, float(segment.get("score", 1.0)), mask)
                    )

    per_category: dict[str, Any] = {}
    iou_values: list[float] = []
    precision_values: list[float] = []
    f1_values: list[float] = []
    for category_id, item in counts.items():
        tp, fp, fn = item["tp"], item["fp"], item["fn"]
        union = tp + fp + fn
        iou = tp / union if union else None
        precision = tp / (tp + fp) if tp + fp else (0.0 if fn else None)
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
        per_category[str(category_id)] = {
            "name": categories[category_id]["name"],
            "tp_pixels": tp,
            "fp_pixels": fp,
            "fn_pixels": fn,
            "IoU": iou,
            "precision": precision,
            "F1": f1,
        }
        if iou is not None:
            iou_values.append(float(iou))
        if precision is not None:
            precision_values.append(float(precision))
        if f1 is not None:
            f1_values.append(float(f1))

    result = {
        "split": split_name,
        "image_count": len(selected_ids),
        "pixel_metrics": {
            "definition": "Per-class pixel metrics; GT void pixels are excluded; reported values are macro means.",
            "IoU_macro": float(np.mean(iou_values)) if iou_values else None,
            "precision_macro": float(np.mean(precision_values)) if precision_values else None,
            "F1_macro": float(np.mean(f1_values)) if f1_values else None,
            "per_category": per_category,
        },
        "instance_mAP": _instance_average_precision(
            gt_instances,
            pred_instances,
            thing_category_ids,
        ),
        "panoptic_quality": _panoptic_quality(
            gt_panoptic,
            pred_panoptic,
            category_ids,
        ),
    }
    return result
