import os

from mmdet.datasets import CityscapesDataset


CityscapesDataset.CLASSES = ("manzana_construida", "zona_verde")
CityscapesDataset.STUFF_CLASSES = ("acera", "carretera")

_base_ = "../../third_party/EfficientPS/configs/efficientPS_singlegpu_sample.py"

PROJECT_ROOT = os.environ.get("CARTOGRAFIA_PROJECT_ROOT")
if PROJECT_ROOT is None:
    raise RuntimeError("Define CARTOGRAFIA_PROJECT_ROOT antes de cargar esta configuración.")
PROJECT_ROOT = os.path.abspath(PROJECT_ROOT)
WORKSPACE_ROOT = os.path.dirname(PROJECT_ROOT)
DATASET_ROOT = os.path.join(WORKSPACE_ROOT, "reto-1-cartografa-automatizada")
PREPARED_ROOT = os.path.join(
    PROJECT_ROOT, "data_panoptic_stuff/efficientps_original"
)
RUN_ROOT = os.path.join(
    PROJECT_ROOT, "training/runs/efficientps_original"
)

model = dict(
    pretrained=True,
    backbone=dict(
        type="tf_efficientnet_b0",
        act_cfg=dict(type="Identity"),
    ),
    neck=dict(in_channels=[24, 40, 112, 1280]),
    bbox_head=dict(num_classes=3),
    mask_head=dict(num_classes=3),
    semantic_head=dict(num_classes=4, ignore_label=255),
)

test_cfg = dict(panoptic=dict(min_stuff_area=1))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True, with_seg=True),
    dict(
        type="Resize",
        img_scale=(384, 384),
        ratio_range=(0.85, 1.15),
        keep_ratio=True,
    ),
    dict(type="RandomCrop", crop_size=(384, 384)),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels", "gt_masks", "gt_semantic_seg"],
    ),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(384, 384),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=False),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

validation_loss_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True, with_seg=True),
    dict(type="Resize", img_scale=(384, 384), keep_ratio=False),
    dict(type="RandomFlip", flip_ratio=0.0),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(
        type="DefaultFormatBundle",
    ),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels", "gt_masks", "gt_semantic_seg"],
    ),
]

dataset_type = "CityscapesDataset"
data = dict(
    imgs_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file=os.path.join(PREPARED_ROOT, "annotations/instances_train.json"),
        img_prefix=DATASET_ROOT,
        seg_prefix=os.path.join(PREPARED_ROOT, "semantic/train"),
        filter_empty_gt=False,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        ann_file=os.path.join(
            PREPARED_ROOT, "annotations/instances_validation.json"
        ),
        img_prefix=DATASET_ROOT,
        seg_prefix=os.path.join(PREPARED_ROOT, "semantic/validation"),
        panoptic_gt=os.path.join(PREPARED_ROOT, "panoptic_validation"),
        filter_empty_gt=False,
        pipeline=test_pipeline,
    ),
    val_loss=dict(
        type=dataset_type,
        ann_file=os.path.join(
            PREPARED_ROOT, "annotations/instances_validation.json"
        ),
        img_prefix=DATASET_ROOT,
        seg_prefix=os.path.join(PREPARED_ROOT, "semantic/validation"),
        filter_empty_gt=False,
        pipeline=validation_loss_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file=os.path.join(PREPARED_ROOT, "annotations/instances_test.json"),
        img_prefix=DATASET_ROOT,
        seg_prefix=os.path.join(PREPARED_ROOT, "semantic/test"),
        panoptic_gt=os.path.join(PREPARED_ROOT, "panoptic_test"),
        filter_empty_gt=False,
        test_mode=True,
        pipeline=test_pipeline,
    ),
)

evaluation = dict(interval=1, metric=["panoptic"])
validation_loss = dict(patience=8, min_delta=0.0)
optimizer = dict(type="SGD", lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy="step",
    warmup="linear",
    warmup_iters=100,
    warmup_ratio=1.0 / 3,
    step=[28, 36],
)
checkpoint_config = dict(interval=1, max_keep_ckpts=2)
log_config = dict(
    interval=10,
    hooks=[dict(type="TextLoggerHook")],
)
total_epochs = 40
work_dir = RUN_ROOT
load_from = None
resume_from = None
workflow = [("train", 1)]
