import logging
import numpy as np
import os
import json
import operator
import itertools
import visdom
import torch
from fvcore.common.file_io import PathManager
from detectron2.data import (
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.data.dataset_mapper import DatasetMapper
from torch.utils.data.sampler import Sampler
from detectron2.data.common import (
    AspectRatioGroupedDataset,
    DatasetFromList,
    MapDataset
)
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.utils import comm
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    EventWriter,
    JSONWriter,
    get_event_storage,
)
from detectron2.utils.logger import setup_logger
setup_logger()
logger = logging.getLogger("detectron2")


class VISWriter(EventWriter):
    """
    Draw training and validation loss in visdom
    http://ec2-100-205-232-116.compute-1.amazonaws.com:8097/
    """

    def __init__(self, OUTPUT_DIR, window_size):
        """
        Args:
            window_size (int): the window size of median smoothing for the
            scalars whose `smoothing_hint` are True.
        """
        self._file_handle_loss = PathManager.open(
            os.path.join(OUTPUT_DIR, "loss.json"), "a")
        self._window_size = window_size
        self._win = None

    def write(self):

        storage = get_event_storage()
        to_save = {"iteration": storage.iter}
        to_save.update(storage.latest_with_smoothing_hint(self._window_size))

        self._file_handle_loss.write(json.dumps(
            {"iteration": storage.iter, "train_loss": to_save["total_loss"],
             "validation_loss": to_save["validation_loss"]}) + "\n")
        self._file_handle_loss.flush()

        viz = visdom.Visdom()

        opts = dict(xlabel='iter', ylabel='Loss', title='loss', legend=[
            "train", "validation"])
        if self._win is None:
            self._win = viz.line(
                X=np.column_stack((storage.iter, storage.iter)),
                Y=np.column_stack(
                    ([to_save["total_loss"], to_save["validation_loss"]])),
                opts=opts
            )

        else:
            viz.line(
                X=np.column_stack((storage.iter, storage.iter)),
                Y=np.column_stack(
                    ([to_save["total_loss"], to_save["validation_loss"]])),
                win=self._win,
                update='append',
                opts=opts
            )
        try:
            os.fsync(self._file_handle.fileno())
        except AttributeError:
            pass

    def close(self):
        self._file_handle.close()


class ValidationSampler(Sampler):

    def __init__(self, size: int):

        self._size = size
        assert size > 0
        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(
            self._indices(), start, None, self._world_size
        )

    def _indices(self):
        yield from torch.arange(self._size)


def build_detection_validation_loader(cfg, dataset_name, mapper=None):
    """
    Similar to `build_detection_test_loader`.
    uses batch size 1.
    """
    images_per_worker = 1

    dataset_dicts = get_detection_dataset_dicts(
        [dataset_name],
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
        if cfg.MODEL.LOAD_PROPOSALS else None,
    )
    dataset = DatasetFromList(dataset_dicts, copy=False)
    data_len = len(dataset)
    if mapper is None:
        mapper = DatasetMapper(cfg, True)
    dataset = MapDataset(dataset, mapper)

    sampler_name = "ValidationSampler"
    sampler = ValidationSampler(len(dataset))
    logger = logging.getLogger(__name__)
    logger.info("Using validation sampler {}".format(sampler_name))

    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, 1, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        # don't batch, but yield individual elements
        collate_fn=operator.itemgetter(0)
    )  # yield individual mapped dict
    data_loader = AspectRatioGroupedDataset(data_loader, images_per_worker)

    return data_loader, data_len


def do_validation(cfg, model, storage):
    val_data, len_val_data = build_detection_validation_loader(
        cfg, "table_test"
    )
    sum_losses_val = 0
    for i, batch in enumerate(val_data):
        print("i: ", i)
        print("file name: ", batch[0]["file_name"])
        losses = model(batch)
        total_loss_smpl = 0
        for loss in losses.values():
            total_loss_smpl += loss.item()
        print("total_loss_smpl: ", total_loss_smpl)
        print(70 * "*")
        sum_losses_val += total_loss_smpl

    print("AVERAGE: ", sum_losses_val / len_val_data)
    return sum_losses_val / len_val_data


def do_train(cfg, model, resume=True, patience=3):
    """
    Our own training loop:
        - computes validation loss every cfg.TEST.EVAL_PERIOD
        - saves iteration, train and validation loss in loss.json
        - training stops with early stopping if no improvement of validation
        loss for 'patience' iterations
    """
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)
    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get(
            "iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER
    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            VISWriter(cfg.OUTPUT_DIR, cfg.TEST.EVAL_PERIOD)
        ]
        if comm.is_main_process()
        else []
    )

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement
    data_loader = build_detection_train_loader(cfg)
    logger.info("Starting training from iteration {}".format(start_iter))

    best_val_loss = None
    waiting = 0
    stop = False  # when stop training

    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            iteration = iteration + 1
            print("ITERATION: ", iteration)
            storage.step()

            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item()
                                 for k, v in
                                 comm.reduce_dict(loss_dict).items()
                                 }
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(
                    total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar(
                "lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if cfg.TEST.EVAL_PERIOD <= 0:
                continue
            last_iteration = iteration == max_iter
            periodic_iteration = (iteration % cfg.TEST.EVAL_PERIOD) == 0

            if periodic_iteration:
                validation_loss = do_validation(cfg, model, storage)
                storage.put_scalar(
                    "validation_loss",
                    validation_loss,
                    smoothing_hint=False)
                comm.synchronize()
                print("validation loss: ", validation_loss)
                print("best_val_loss: ", best_val_loss)
                print("waiting: ", waiting)
                print("patience: ", patience)
                # Early stopping
                if best_val_loss is None:
                    best_val_loss = validation_loss
                    periodic_checkpointer.save(
                        "model_{:07d}".format(iteration), iteration=iteration
                    )
                    print("best_val_loss: ", best_val_loss)
                elif validation_loss < best_val_loss:
                    # save model
                    periodic_checkpointer.save(
                        "model_{:07d}".format(iteration), iteration=iteration
                    )
                    best_val_loss = validation_loss
                    waiting = 0
                else:
                    waiting += 1

                if waiting == patience:
                    stop = True

            if periodic_iteration or last_iteration:
                for writer in writers:
                    writer.write()
                if stop:
                    break
