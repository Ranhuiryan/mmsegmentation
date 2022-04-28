# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class IRFissure(CustomDataset):

    CLASSES = ('ground', 'fissure')

    PALETTE = [[128, 128, 128], [38, 127, 129]]

    def __init__(self, split, **kwargs):
        super().__init__(seg_map_suffix='.png', split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
