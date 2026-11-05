from __future__ import print_function

import argparse
import json
import os

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import load_checkpoint
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector


CITYSCAPES_TRAIN_ID_TO_PROJECT = {
    0: 1,
    1: 4,
    2: 2,
    3: 3,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run official EfficientPS panoptic inference for one image."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("EfficientPS original requiere CUDA para esta aplicación.")

    config = Config.fromfile(args.config)
    config.model.pretrained = None
    model = build_detector(
        config.model,
        train_cfg=None,
        test_cfg=config.test_cfg,
    )
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.eval()

    image_path = os.path.abspath(args.image)
    pipeline = Compose(config.data.test.pipeline)
    data = pipeline(
        {
            "img_info": {"filename": os.path.basename(image_path)},
            "img_prefix": os.path.dirname(image_path),
        }
    )
    batch = collate([data], samples_per_gpu=1)
    with torch.no_grad():
        predictions = model(
            return_loss=False,
            rescale=True,
            eval="panoptic",
            **batch
        )

    panoptic_ids, category_ids, _, scores = predictions[0]
    panoptic_ids = panoptic_ids.numpy()
    category_ids = category_ids.numpy()
    scores = scores.numpy()
    category_map = np.zeros(panoptic_ids.shape, dtype=np.uint8)
    segment_scores = {}
    for segment_id in np.unique(panoptic_ids):
        if segment_id >= len(category_ids):
            raise ValueError(
                "EfficientPS devolvió un ID de segmento sin categoría: {}".format(
                    segment_id
                )
            )
        cityscapes_id = int(category_ids[segment_id])
        project_id = CITYSCAPES_TRAIN_ID_TO_PROJECT.get(cityscapes_id)
        if project_id is None:
            if cityscapes_id == 255:
                continue
            raise ValueError(
                "No existe mapeo para category_id Cityscapes {}.".format(
                    cityscapes_id
                )
            )
        category_map[panoptic_ids == segment_id] = project_id
        segment_scores[int(segment_id)] = float(scores[segment_id])

    np.savez_compressed(
        args.output,
        category_map=category_map,
        segment_ids=panoptic_ids.astype(np.int32),
    )
    print(
        json.dumps(
            {
                "height": int(category_map.shape[0]),
                "width": int(category_map.shape[1]),
                "segments": len(segment_scores),
            }
        )
    )


if __name__ == "__main__":
    main()
