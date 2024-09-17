#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import cv2
import logging
import os
from collections import OrderedDict
import torch
import matplotlib.pyplot as plt
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)

from detectron2.utils.visualizer import Visualizer
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.solver.build import maybe_add_gradient_clipping, get_default_optimizer_params

# from swint import add_swint_config
import torch
import torch.nn as nn
import torchvision.models as models
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.modeling import Backbone
from detectron2.modeling.backbone import FPN
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode
# from segment_anything import SamAutomaticMaskGenerator
#import torch
import torch.nn as nn
import torchvision.models as models
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.modeling import Backbone
from detectron2.modeling.backbone import FPN
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode
from build_sam import build_sam_vit_h, _build_sam
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
# --------------------------------------------------------
# Swin Transformer
# modified from https://github.com/SwinTransformer/Swin-Transformer-Object-Detection/blob/master/mmdet/models/backbones/swin_transformer.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool, LastLevelP6P7
from detectron2.layers import ShapeSpec
# t = torch.cuda.get_device_properties(0).total_memory
# r = torch.cuda.memory_reserved(0)
# a = torch.cuda.memory_allocated(0)
# # Define your custom CNN network as a PyTorch module.
class SAM(nn.Module):
    def __init__(self):
        super().__init__()
        #self.conv1 = sam_model_registry["vit_h"]()
        checkpoint_ = '/home/fariborz_taherkhani/Backup_sam/sam_vit_h_4b8939.pth'
        checkpoint_=None
        self.conv1=build_sam_vit_h(checkpoint=checkpoint_)
        f = '/home/fariborz_taherkhani/Backup_sam/sam_vit_h_4b8939.pth'
        state_dict = torch.load(f)
        for name, param in self.conv1.named_parameters():
            if 'pos_embed' not in name and 'necks' not in name:
                pretrained_param = state_dict[name]
                param.data = pretrained_param.data
                print(f"Loaded {name} successfully")
        #self.conv1.load_state_dict(state_dict)
        for name, param in state_dict.items():
            if 'decoder' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        # for name, param in state_dict.items():
        #     param.requires_grad = False

        self.conv1.to(device='cuda')


    def forward(self, x):
        # self.predictor.set_image(x)
        x = self.conv1(x)
        # for x_ in x:
        #     print(x_.shape)
        return x
# @BACKBONE_REGISTRY.register()
class MyBackbone(Backbone):
    def __init__(self, cfg, input_shape):
        #print(input_shape)
        super().__init__()
        self.sam = SAM()
        self.ratio = 1
        self.depth = 32
        self.names=[]
        for i in range(self.depth):
            if (i+1)%8==0:
                name = f'stage{(i+1)//8}'
                self.names.append(name)
    def forward(self, x):
        multimask_output=False
        #print(x.shape)

        #import torchvision.transforms as transforms
        # img = x[0]
        # # permute to match the desired memory format
        # img = img.permute(1, 2, 0).cpu().numpy()
        # plt.imshow(img)
        # plt.savefig("mygraph.png")
        x = self.sam(x)
        outs = {}
        for i in range(len(x)):
            name = f'stage{(i+1)}'
            outs[name] = x[(i)]
        # print('this is x shape', x.shape)
        #self.ratio= (480 // x.shape[2])
        return outs

    def output_shape(self):
        outs = {}
        i = 1
        for name in self.names:
            outs[name] = ShapeSpec(
                channels=256, stride=2**i
            )
            i += 1
        return outs


@BACKBONE_REGISTRY.register()
def build_swint_fpn_backbone(cfg, input_shape):
#     """
#     Args:
#         cfg: a detectron2 CfgNode
#
#     Returns:
#         backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
#     """
    bottom_up = MyBackbone(cfg, input_shape)
    input_shapes = bottom_up.output_shape()
    print(input_shapes)

    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    # print(in_features )
    # print(out_channels)
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    #print(backbone._out_features)
    return backbone
class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    @classmethod
    def build_optimizer(cls, cfg, model):
        params = get_default_optimizer_params(
            model,
            base_lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
            bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
            weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
            overrides={
                "absolute_pos_embed": {"lr": cfg.SOLVER.BASE_LR, "weight_decay": 0.0},
                "relative_position_bias_table": {"lr": cfg.SOLVER.BASE_LR, "weight_decay": 0.0},
            }
        )

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM,
                nesterov=cfg.SOLVER.NESTEROV,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        elif optimizer_type == "AdamW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR, betas=(0.9, 0.999),
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        return optimizer


def setup(args):
    """
    Create configs and perform basic setups.
    """
    from detectron2.data.datasets import register_coco_instances
    from detectron2.data import MetadataCatalog, DatasetCatalog
    address_ = '/home/fariborz_taherkhani/'
    register_coco_instances("experiment", {}, os.path.join(address_, "train/trainSwinC.json"),
                            os.path.join(address_, "train/images"))

    sample_metadata = MetadataCatalog.get("experiment")
    dataset_dicts = DatasetCatalog.get("experiment")
    #####################################################################################################
    register_coco_instances("experiment_test", {}, os.path.join(address_, "test/testSwinC.json"),
                            os.path.join(address_, "test/images"))
    sample_metadata = MetadataCatalog.get("experiment_test")
    dataset_dicts = DatasetCatalog.get("experiment_test")
    cfg = get_cfg()
    #cfg.MODEL.BACKBONE.FREEZE_AT = -1
    # default_setup(cfg, args)
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES=7
    cfg.DATALOADER.NUM_WORKERS = 1

    #cfg.MODEL.BACKBONE.NAME = "MyBackbone"
    #cfg.MODEL.BACKBONE.SAM = SAM()
    # cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 2000
    # cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 1000
    # cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 1000
    # cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7
    cfg.SOLVER.IMS_PER_BATCH = 128
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.MAX_ITER = 30000
    # cfg.ROI_BOX_HEAD.FED_LOSS_NUM_CLASSES=7
    # cfg.SOLVER.STEPS = []
    # cfg.TEST.EVAL_PERIOD = 500
    # cfg.MODEL.FPN.TOP_LEVELS = 2
    cfg.SOLVER.OPTIMIZER = "AdamW"
    # add_swint_config(cfg)
    # print(cfg)
    cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    # cfg.MODEL.BACKBONE.FREEZE_AT = 0
    cfg.SOLVER.IMS_PER_BATCH = 1
    #
    # cfg.freeze()
    # default_setup(cfg, args)
    default_setup(cfg, args)
    cfg.merge_from_file(
        "configs/SwinT/mask_rcnn_swint_T_FPN_3x.yaml")  # replace with your model's config path
    cfg.MODEL.WEIGHTS = "/home/fariborz_taherkhani/Updated_Sam_Model/output/model_final.pth"  # replace with your model's weights path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
    # cfg.freeze()
    predictor = DefaultPredictor(cfg)
    predictor = DefaultPredictor(cfg)
    # evaluator = COCOEvaluator("experiment_test", cfg, False, output_dir="./output/")
    # val_loader = build_detection_test_loader(cfg, "experiment_test")
    # inference_on_dataset(predictor.model, val_loader, evaluator)

    folder_path = '/home/fariborz_taherkhani/test/images'
    folder_path = '/home/fariborz_taherkhani/Updated_Sam_Model/multi'
    for filename in os.listdir(folder_path):
        # Check if the file is an image
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Read the image using OpenCV
            image_path = os.path.join(folder_path, filename)
            im = cv2.imread(image_path)
            # dim=(640,480)
            # im = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)
            # plt.imshow(im)
            # plt.show()
            outputs = predictor(im)
            v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2
                           )
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            # cv2_imshow(v.get_image()[:, :, ::-1])
            plt.imshow(v.get_image())
            plt.show()


    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)

    trainer.resume_or_load(resume=args.resume)
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    torch.cuda.empty_cache()

    args = default_argument_parser().parse_args()

    args.config_file = 'configs/SwinT/mask_rcnn_swint_T_FPN_3x.yaml'


    # args.nums_gpus=2
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

