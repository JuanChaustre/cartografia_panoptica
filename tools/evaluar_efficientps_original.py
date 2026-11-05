from __future__ import print_function

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.apis import single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evalúa EfficientPS original sobre validation o test y archiva "
            "sus salidas panópticas."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("EfficientPS original requiere CUDA.")
    if not 0 <= args.score_threshold < 1:
        raise ValueError("El umbral debe estar en el intervalo [0, 1).")

    config = Config.fromfile(args.config)
    config.model.pretrained = None
    dataset_config = config.data.val if args.split == "validation" else config.data.test
    dataset = build_dataset(dataset_config, dict(test_mode=True))
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
    )
    model = build_detector(
        config.model,
        train_cfg=None,
        test_cfg=config.test_cfg,
    )
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.eval()

    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            "El directorio de salida no está vacío: {}".format(output_root)
        )
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="efficientps-evaluation-") as work_dir:
        previous_directory = os.getcwd()
        try:
            os.chdir(work_dir)
            model.module.test_cfg.rcnn.score_thr = args.score_threshold
            os.environ["EFFICIENTPS_PANOPTIC_OUTPUT_DIR"] = str(output_root)
            single_gpu_test(model, data_loader, show=False, eval=["panoptic"])
            metrics = dataset.evaluate([], metric=["panoptic"])
            summary = {
                "split": args.split,
                "score_threshold": args.score_threshold,
                "All_pq": float(metrics["All_pq"]),
                "Things_pq": float(metrics["Things_pq"]),
            }
            print(
                "score_threshold={:.3f} All_pq={:.4f} Things_pq={:.4f}".format(
                    args.score_threshold,
                    metrics["All_pq"],
                    metrics["Things_pq"],
                )
            )
        finally:
            os.chdir(previous_directory)
            os.environ.pop("EFFICIENTPS_PANOPTIC_OUTPUT_DIR", None)

    summary_path = output_root / "official_pq.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("Official PQ summary: {}".format(summary_path))


if __name__ == "__main__":
    main()
