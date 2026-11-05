from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from data import PanopticDataset, load_records, make_mask2former_targets
from metrics import evaluate_panoptic
from utils import save_history, save_metrics, save_predictions, seed_everything


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_DATASET_ROOT = WORKSPACE_ROOT / "reto-1-cartografa-automatizada"
DEFAULT_ANNOTATIONS = PROJECT_ROOT / "data_panoptic_stuff/annotations/panoptic.json"
DEFAULT_SPLITS = PROJECT_ROOT / "data_panoptic_stuff/splits.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "training/runs/mask2former_stuff"
CHECKPOINT = "facebook/mask2former-swin-tiny-coco-panoptic"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ajusta Mask2Former para segmentación panóptica."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="Épocas sin mejora de validación antes de detener.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="Mejora mínima de validation_loss para reiniciar patience.",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.image_size < 128 or args.patience < 1:
        parser.error("--epochs y --patience deben ser positivos; --image-size mínimo 128.")
    if args.min_delta < 0:
        parser.error("--min-delta no puede ser negativo.")
    return args


def prepare_pixels(
    processor: Mask2FormerImageProcessor,
    image: np.ndarray,
    image_size: int,
    device: torch.device,
) -> torch.Tensor:
    resized = Image.fromarray(image).resize(
        (image_size, image_size),
        resample=Image.Resampling.BILINEAR,
    )
    encoded = processor(
        images=np.asarray(resized),
        do_resize=False,
        return_tensors="pt",
    )
    return encoded["pixel_values"].to(device)


def loss_for_sample(
    model: Mask2FormerForUniversalSegmentation,
    processor: Mask2FormerImageProcessor,
    sample: dict[str, Any],
    categories: list[dict[str, Any]],
    image_size: int,
    device: torch.device,
) -> torch.Tensor:
    _, masks, labels = make_mask2former_targets(sample, categories, image_size)
    pixels = prepare_pixels(processor, sample["image"], image_size, device)
    output = model(
        pixel_values=pixels,
        mask_labels=[masks.to(device)],
        class_labels=[labels.to(device)],
    )
    if output.loss is None:
        raise RuntimeError("Mask2Former no devolvió loss con las etiquetas de entrenamiento.")
    return output.loss


def mean_epoch_loss(
    model: Mask2FormerForUniversalSegmentation,
    processor: Mask2FormerImageProcessor,
    loader: DataLoader[list[dict[str, Any]]],
    dataset: PanopticDataset,
    categories: list[dict[str, Any]],
    image_size: int,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    amp: bool = False,
) -> float:
    training = optimizer is not None
    model.train(training)
    dataset.set_epoch(getattr(dataset, "epoch", 0) + 1)
    total_loss = 0.0
    sample_count = 0

    for batch in loader:
        sample = batch[0]
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp,
            ):
                loss = loss_for_sample(
                    model,
                    processor,
                    sample,
                    categories,
                    image_size,
                    device,
                )
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

        total_loss += float(loss.detach().item())
        sample_count += 1
        if sample_count % 10 == 0:
            print(
                f"  {'train' if training else 'validation'} "
                f"{sample_count}/{len(dataset)} loss={total_loss / sample_count:.4f}",
                flush=True,
            )
    if sample_count == 0:
        raise ValueError("No hay muestras para calcular la pérdida.")
    return total_loss / sample_count


@torch.inference_mode()
def predict_split(
    model: Mask2FormerForUniversalSegmentation,
    processor: Mask2FormerImageProcessor,
    dataset: PanopticDataset,
    categories: list[dict[str, Any]],
    image_size: int,
    device: torch.device,
    amp: bool,
) -> list[dict[str, Any]]:
    model.eval()
    category_ids = [int(item["id"]) for item in categories]
    label_ids_to_fuse = {
        index
        for index, item in enumerate(categories)
        if int(item["isthing"]) == 0
    }
    predictions: list[dict[str, Any]] = []

    for index in range(len(dataset)):
        sample = dataset[index]
        pixels = prepare_pixels(processor, sample["image"], image_size, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp,
        ):
            outputs = model(pixel_values=pixels)
        result = processor.post_process_panoptic_segmentation(
            outputs,
            target_sizes=[(sample["height"], sample["width"])],
            label_ids_to_fuse=label_ids_to_fuse,
            threshold=0.5,
            mask_threshold=0.5,
            overlap_mask_area_threshold=0.8,
        )[0]
        unique_segments: dict[int, dict[str, Any]] = {}
        for segment in result["segments_info"]:
            segment_id = int(segment["id"])
            category_id = category_ids[int(segment["label_id"])]
            existing = unique_segments.get(segment_id)
            if existing is not None and existing["category_id"] != category_id:
                raise ValueError(
                    f"El postprocesador asignó dos categorías al segmento {segment_id}."
                )
            unique_segments[segment_id] = {
                "id": segment_id,
                "category_id": category_id,
                "score": max(
                    float(segment["score"]),
                    float(existing["score"]) if existing else 0.0,
                ),
            }
        segment_info = list(unique_segments.values())
        predictions.append(
            {
                "image_id": sample["image_id"],
                "file_name": sample["file_name"],
                "height": sample["height"],
                "width": sample["width"],
                "segmentation": result["segmentation"].cpu().numpy(),
                "segments_info": segment_info,
            }
        )
    return predictions


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    print(f"Dispositivo: {device}; AMP: {amp}; checkpoint: {args.checkpoint}")

    train_records, categories = load_records(
        args.annotations, args.dataset_root, args.splits, "train"
    )
    validation_records, validation_categories = load_records(
        args.annotations, args.dataset_root, args.splits, "validation"
    )
    test_records, test_categories = load_records(
        args.annotations, args.dataset_root, args.splits, "test"
    )
    if categories != validation_categories or categories != test_categories:
        raise ValueError("Las categorías no coinciden entre las particiones.")

    train_dataset = PanopticDataset(train_records, categories, augment=True, seed=args.seed)
    validation_dataset = PanopticDataset(validation_records, categories)
    test_dataset = PanopticDataset(test_records, categories)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: batch,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: batch,
    )

    processor = Mask2FormerImageProcessor.from_pretrained(args.checkpoint)
    labels = {index: item["name"] for index, item in enumerate(categories)}
    label2id = {name: index for index, name in labels.items()}
    if args.no_pretrained:
        from transformers import Mask2FormerConfig

        config = Mask2FormerConfig.from_pretrained(args.checkpoint)
        config.num_labels = len(categories)
        config.id2label = labels
        config.label2id = label2id
        config.ignore_value = 255
        model = Mask2FormerForUniversalSegmentation(config)
    else:
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            args.checkpoint,
            num_labels=len(categories),
            id2label=labels,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
        model.config.ignore_value = 255
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start_epoch = 0
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    resumed_history: list[dict[str, float]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"])
        best_validation_loss = float(checkpoint["best_validation_loss"])
        epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )
        resumed_history = checkpoint.get("history", [])
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = args.learning_rate
            parameter_group["initial_lr"] = args.learning_rate
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            last_epoch=start_epoch - 1,
        )
        print(
            f"Reanudando desde época {start_epoch}; "
            f"tasa de aprendizaje ajustada al plan de {args.epochs} épocas."
        )

    history = resumed_history

    for epoch in range(start_epoch, args.epochs):
        print(f"Época {epoch + 1}/{args.epochs}", flush=True)
        train_loss = mean_epoch_loss(
            model,
            processor,
            train_loader,
            train_dataset,
            categories,
            args.image_size,
            device,
            optimizer=optimizer,
            scaler=scaler,
            amp=amp,
        )
        validation_loss = mean_epoch_loss(
            model,
            processor,
            validation_loader,
            validation_dataset,
            categories,
            args.image_size,
            device,
            amp=amp,
        )
        scheduler.step()
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        save_history(history, args.output_dir)
        improved = validation_loss < best_validation_loss - args.min_delta
        if improved:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
            model.save_pretrained(args.output_dir / "best_model")
            processor.save_pretrained(args.output_dir / "best_model")
        else:
            epochs_without_improvement += 1
        torch.save(
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_validation_loss": best_validation_loss,
                "epochs_without_improvement": epochs_without_improvement,
                "history": history,
            },
            args.output_dir / "last_training.pt",
        )
        print(
            f"train_loss={train_loss:.4f} validation_loss={validation_loss:.4f} "
            f"sin_mejora={epochs_without_improvement}/{args.patience}",
            flush=True,
        )
        if epochs_without_improvement >= args.patience:
            print(
                f"Early stopping: validation_loss no mejoró durante "
                f"{args.patience} épocas consecutivas."
            )
            break

    best_path = args.output_dir / "best_model"
    if best_path.is_dir():
        model = Mask2FormerForUniversalSegmentation.from_pretrained(best_path).to(device)
    predictions = predict_split(
        model,
        processor,
        test_dataset,
        categories,
        args.image_size,
        device,
        amp,
    )
    prediction_file = save_predictions(
        predictions,
        categories,
        args.output_dir / "test_predictions",
    )
    metrics = evaluate_panoptic(
        args.annotations,
        prediction_file,
        args.splits,
        "test",
    )
    save_metrics(metrics, args.output_dir / "test_metrics.json")
    print(json.dumps(metrics["pixel_metrics"], ensure_ascii=False, indent=2))
    print(f"Predicciones de test: {prediction_file}")
    print(f"Métricas: {args.output_dir / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
