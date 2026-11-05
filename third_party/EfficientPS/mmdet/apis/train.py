import csv
import os
import random
import shutil
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import DistSamplerSeedHook, Hook, Runner

from mmdet.core import (DistEvalHook, DistOptimizerHook, EvalHook,
                        Fp16OptimizerHook, build_optimizer)
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.utils import get_root_logger


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    log_vars['loss'] = loss
    for loss_name, loss_value in log_vars.items():
        # reduce loss when distributed training
        if dist.is_available() and dist.is_initialized():
            loss_value = loss_value.data.clone()
            dist.all_reduce(loss_value.div_(dist.get_world_size()))
        log_vars[loss_name] = loss_value.item()

    return loss, log_vars


class ValidationLossHook(Hook):

    def __init__(
        self,
        dataloader,
        patience=8,
        min_delta=0.0,
        history_path=None,
        resume=False,
    ):
        self.dataloader = dataloader
        self.patience = patience
        self.min_delta = min_delta
        self.train_loss_total = 0.0
        self.train_sample_count = 0
        self.history = []
        self.best_validation_loss = float('inf')
        self.epochs_without_improvement = 0
        self.resume = resume
        self.history_path = Path(history_path) if history_path else None
        if resume:
            if self.history_path is None or not self.history_path.is_file():
                raise RuntimeError(
                    'Cannot resume early stopping without loss_history.csv.')
            with self.history_path.open(newline='') as history_file:
                self.history = [
                    {
                        'epoch': float(row['epoch']),
                        'train_loss': float(row['train_loss']),
                        'validation_loss': float(row['validation_loss']),
                    }
                    for row in csv.DictReader(history_file)
                ]
            if not self.history:
                raise RuntimeError(
                    'Cannot resume early stopping from an empty loss history.')

    def before_run(self, runner):
        if not self.resume:
            return

        completed_epoch = runner.epoch
        self.history = [
            item for item in self.history
            if item['epoch'] <= completed_epoch
        ]
        self.best_validation_loss = float('inf')
        self.epochs_without_improvement = 0
        for item in self.history:
            validation_loss = item['validation_loss']
            if validation_loss < self.best_validation_loss - self.min_delta:
                self.best_validation_loss = validation_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
        from utils import save_history
        save_history(self.history, Path(runner.work_dir))

    def after_train_iter(self, runner):
        log_vars = runner.outputs.get('log_vars', {})
        if 'loss' in log_vars:
            self.train_loss_total += float(log_vars['loss'])
            self.train_sample_count += int(runner.outputs.get('num_samples', 1))

    def after_train_epoch(self, runner):
        model = runner.model
        model.eval()
        validation_loss_total = 0.0
        validation_sample_count = 0
        try:
            with torch.no_grad():
                for data in self.dataloader:
                    losses = model(return_loss=True, **data)
                    loss, _ = parse_losses(losses)
                    validation_loss_total += float(loss.item())
                    validation_sample_count += 1
        finally:
            model.train()

        if not validation_sample_count:
            raise RuntimeError('Validation-loss dataset contains no samples.')
        if not self.train_sample_count:
            raise RuntimeError('No training losses were recorded this epoch.')

        validation_loss = validation_loss_total / validation_sample_count
        train_loss = self.train_loss_total / self.train_sample_count
        self.history.append(
            dict(
                epoch=float(runner.epoch + 1),
                train_loss=train_loss,
                validation_loss=validation_loss,
            ))
        runner.log_buffer.output['validation_loss'] = validation_loss
        runner.log_buffer.ready = True

        from utils import save_history
        save_history(self.history, Path(runner.work_dir))
        runner.logger.info(
            'Epoch %d: train_loss=%.6f validation_loss=%.6f',
            runner.epoch + 1, train_loss, validation_loss)
        if validation_loss < self.best_validation_loss - self.min_delta:
            self.best_validation_loss = validation_loss
            self.epochs_without_improvement = 0
            checkpoint = Path(runner.work_dir) / 'epoch_{}.pth'.format(
                runner.epoch + 1)
            if not checkpoint.is_file():
                raise RuntimeError(
                    'Cannot save best checkpoint; epoch checkpoint is missing: '
                    '{}'.format(checkpoint))
            shutil.copy2(
                str(checkpoint),
                str(Path(runner.work_dir) / 'best_validation_loss.pth'),
            )
        else:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.patience:
                runner._max_epochs = runner.epoch + 1
                runner.logger.info(
                    'Early stopping after %d epochs without validation-loss improvement.',
                    self.patience)
        self.train_loss_total = 0.0
        self.train_sample_count = 0


class EarlyStoppingRunner(Runner):

    def run(self, data_loaders, workflow, max_epochs, **kwargs):
        assert isinstance(data_loaders, list)
        assert len(data_loaders) == len(workflow)
        self._max_epochs = max_epochs
        for index, flow in enumerate(workflow):
            mode, _ = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[index])
                break

        self.logger.info(
            'Start running workflow %s, maximum %d epochs',
            workflow, max_epochs)
        self.call_hook('before_run')
        while self.epoch < self._max_epochs:
            for index, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):
                    if not hasattr(self, mode):
                        raise ValueError(
                            'runner has no method named "{}"'.format(mode))
                    epoch_runner = getattr(self, mode)
                elif callable(mode):
                    epoch_runner = mode
                else:
                    raise TypeError(
                        'workflow mode must be a method name or callable.')

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    epoch_runner(data_loaders[index], **kwargs)
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break

        time.sleep(1)
        self.call_hook('after_run')


def batch_processor(model, data, train_mode):
    """Process a data batch.

    This method is required as an argument of Runner, which defines how to
    process a data batch and obtain proper outputs. The first 3 arguments of
    batch_processor are fixed.

    Args:
        model (nn.Module): A PyTorch model.
        data (dict): The data batch in a dict.
        train_mode (bool): Training mode or not. It may be useless for some
            models.

    Returns:
        dict: A dict containing losses and log vars.
    """
    losses = model(**data)
    loss, log_vars = parse_losses(losses)

    outputs = dict(
        loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

    return outputs


def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):
    logger = get_root_logger(cfg.log_level)

    # start training
    if distributed:
        _dist_train(
            model,
            dataset,
            cfg,
            validate=validate,
            logger=logger,
            timestamp=timestamp,
            meta=meta)
    else:
        _non_dist_train(
            model,
            dataset,
            cfg,
            validate=validate,
            logger=logger,
            timestamp=timestamp,
            meta=meta)


def _dist_train(model,
                dataset,
                cfg,
                validate=False,
                logger=None,
                timestamp=None,
                meta=None):
    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            dist=True,
            seed=cfg.seed) for ds in dataset
    ]
    # put model on gpus
    find_unused_parameters = cfg.get('find_unused_parameters', False)
    # Sets the `find_unused_parameters` parameter in
    # torch.nn.parallel.DistributedDataParallel
    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False,
        find_unused_parameters=find_unused_parameters)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    runner_class = EarlyStoppingRunner if cfg.get('validation_loss') else Runner
    runner = runner_class(
        model,
        batch_processor,
        optimizer,
        cfg.work_dir,
        logger=logger,
        meta=meta)
    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(**cfg.optimizer_config,
                                             **fp16_cfg)
    else:
        optimizer_config = DistOptimizerHook(**cfg.optimizer_config)

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)
    runner.register_hook(DistSamplerSeedHook())
    # register eval hooks
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=True,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        runner.register_hook(DistEvalHook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)


def _non_dist_train(model,
                    dataset,
                    cfg,
                    validate=False,
                    logger=None,
                    timestamp=None,
                    meta=None):
    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            cfg.gpus,
            dist=False,
            seed=cfg.seed) for ds in dataset
    ]
    # put model on gpus
    model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    runner_class = EarlyStoppingRunner if cfg.get('validation_loss') else Runner
    runner = runner_class(
        model,
        batch_processor,
        optimizer,
        cfg.work_dir,
        logger=logger,
        meta=meta)
    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp
    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=False)
    else:
        optimizer_config = cfg.optimizer_config
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)

    # register eval hooks
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        runner.register_hook(EvalHook(val_dataloader, **eval_cfg))

    if cfg.get('validation_loss', False):
        val_loss_dataset = build_dataset(cfg.data.val_loss)
        val_loss_dataloader = build_dataloader(
            val_loss_dataset,
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)
        validation_loss_cfg = cfg.get('validation_loss', {})
        runner.register_hook(
            ValidationLossHook(
                val_loss_dataloader,
                patience=validation_loss_cfg.get('patience', 8),
                min_delta=validation_loss_cfg.get('min_delta', 0.0),
                history_path=os.path.join(cfg.work_dir, 'loss_history.csv'),
                resume=bool(cfg.resume_from),
            ),
            priority='LOW')

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
