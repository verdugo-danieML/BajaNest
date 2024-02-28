from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.config import CfgNode
from detectron2.solver.build import get_default_optimizer_params, maybe_add_gradient_clipping
from detectron2.engine.hooks import HookBase
import torch
import logging
import os
from data_aug import *
from detectron2.data import build_detection_train_loader
from detectron2.data import DatasetMapper

class EagleNestTrainer(DefaultTrainer):
    """
    This trainer evaluate data on the `cfg.DATASETS.TEST` validation dataset every `cfg.TEST.EVAL_PERIOD` iterations.
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder = None):
        if output_folder == None:
            output_folder = cfg.OUTPUT_DIR
        else:
            output_folder = os.path.join(cfg.OUTPUT_DIR, output_folder)
            os.makedirs(output_folder)

        return COCOEvaluator(dataset_name, distributed = False, output_dir = output_folder)

    """@classmethod
    def build_optimizer(cls, cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
        
        Build an optimizer from config.
        
        params = get_default_optimizer_params(
            model,
            base_lr=cfg.SOLVER.BASE_LR,
            weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
            bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
            weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
        )
        return maybe_add_gradient_clipping(cfg, torch.optim.Adam)(
            params,
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )"""


class EagleNestAugTrainer(DefaultTrainer):
    """
    This trainer evaluate data on the `cfg.DATASETS.TEST` validation dataset every `cfg.TEST.EVAL_PERIOD` iterations.
    It also changes the list of augmentations
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder = None):
        if output_folder == None:
            output_folder = cfg.OUTPUT_DIR
        else:
            output_folder = os.path.join(cfg.OUTPUT_DIR, output_folder)
            os.makedirs(output_folder)

        return COCOEvaluator(dataset_name, distributed = False, output_dir = output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        augs = []
        
        # Aug 0: Add RandomBrightness with 50% chance
        augs.append(
            T.RandomApply(
                T.RandomBrightness(
                    intensity_min = 0.5,
                    intensity_max = 1.5),
                prob = 0.5
                ))
              
        # Aug 1: Add rotation with 50% chance
        augs.append(
            T.RandomApply(
                T.RandomRotation(
                      angle         = [-30, 30],
                      sample_style  = "range",
                      center        = [[0.4, 0.6], [0.4, 0.6]],
                      expand        = False
                        ), prob=0.5))
        
        # Aug 2: Add mixup with 50% chance
        # augs.append(T.RandomApply(MixUpAug(cfg), prob=0.5))
        # Aug 3: Add mosaic with 50% chance
        # augs.append(T.RandomApply(MosaicAug(cfg), prob=0.5))

        # Aug 4: Add ResizeShortestEdge
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        augs.append(T.ResizeShortestEdge(
                min_size, 
                max_size, 
                sample_style)
            )

        # Aug 5: Add RandomFlipping
        if cfg.INPUT.RANDOM_FLIP != "none":
            augs.append(T.RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
                )
            )

        #mapper = ExtendedDatasetMapper(cfg, 
                                       #is_train = True, 
                                       #augmentations = augs)
        mapper = DatasetMapper(cfg,
                               is_train = True,
                               augmentations = augs)
        
        return build_detection_train_loader(cfg, mapper=mapper)


class BestModelHook(HookBase):
    def __init__(self, cfg,
                metric = 'segm/AP50',
                min_max = 'max'):
        self._period = cfg.TEST.EVAL_PERIOD
        self.metric = metric
        self.min_max = min_max
        self.best_value = float('-inf') if min_max == 'max' else float('inf') 
        logger = logging.getLogger('detectron2')
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        self._logger = logger 

    def _take_latest_metrics(self):
        with torch.no_grad():
            latest_metrics = self.trainer.storage.latest()   
            return latest_metrics

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            latest_metrics = self._take_latest_metrics()
            for (key, (value, iter)) in latest_metrics.items():
                if key == self.metric:
                    if (self.min_max == "min" and value < self.best_value) or (self.min_max == "max" and value > self.best_value):
                        self._logger.info("Updating best model at iteration {} with {} = {}".format(iter, self.metric, value))
                        self.best_value = value
                        self.trainer.checkpointer.save("model_best")
        




