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
#from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
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
from segment_anything import SamPredictor, sam_model_registry
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

# Define your custom CNN network as a PyTorch module.
class SAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = sam_model_registry["vit_h"](checkpoint=None)

    def forward(self, x):
        x=self.conv1(x)
        return x
@BACKBONE_REGISTRY.register()
class MyBackbone(Backbone):
    def __init__(self, sam):
        super().__init__()
        self.sam = sam

    def forward(self, x):
        x = self.sam(x)
        return {"stage2": x}

@BACKBONE_REGISTRY.register()
def build_swint_fpn_backbone(cfg, input_shape):
    bottom_up = MyBackbone(sam=SAM())

    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone
