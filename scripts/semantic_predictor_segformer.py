import mmcv
import torch
# from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
# from mmcv.runner import get_dist_info, init_dist, load_checkpoint
# from mmcv.utils import DictAction

from mmseg.apis import init_model, inference_model, show_result_pyplot
# from mmseg.core.evaluation import get_palette
# from mmseg.datasets import build_dataloader, build_dataset
# from mmseg.models import build_segmentor
from mmseg.utils import ade_classes
from mmseg.apis import MMSegInferencer

import numpy as np
import cv2
from PIL import Image


class SemanticPredictor():
    def __init__(self):
        self.objgoal_to_cat = {0: 'chair',     1: 'bed',     2: 'plant',           3: 'toilet',           4: 'tv_monitor',
                               5: 'sofa'}
        self.crossover = {'chair': ['chair', 'armchair', 'swivel chair'],
                          'bed': ['bed '],
                          'plant': ['tree', 'plant', 'flower'],
                          'toilet': ['toilet'],
                          'tv_monitor': ['computer', 'monitor', 'television receiver', 'crt screen', 'screen'],
                          'sofa': ['sofa']}

        self.PALETTE = [[120, 120, 0], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                        [4, 200, 3], [120, 120, 80], [
                            140, 140, 140], [204, 5, 255],
                        [230, 230, 230], [4, 250, 7], [
                            224, 5, 255], [235, 255, 7],
                        [150, 5, 61], [120, 120, 70], [
                            8, 255, 51], [255, 6, 82],
                        [143, 255, 140], [204, 255, 4], [
                            255, 51, 7], [204, 70, 3],
                        [0, 102, 200], [61, 230, 250], [
                            255, 6, 51], [11, 102, 255],
                        [255, 7, 71], [255, 9, 224], [
                            9, 7, 230], [220, 220, 220],
                        [255, 9, 92], [112, 9, 255], [
                            8, 255, 214], [7, 255, 224],
                        [255, 184, 6], [10, 255, 71], [
                            255, 41, 10], [7, 255, 255],
                        [224, 255, 8], [102, 8, 255], [
                            255, 61, 6], [255, 194, 7],
                        [255, 122, 8], [0, 255, 20], [
                            255, 8, 41], [255, 5, 153],
                        [6, 51, 255], [235, 12, 255], [
                            160, 150, 20], [0, 163, 255],
                        [140, 140, 140], [250, 10, 15], [
                            20, 255, 0], [31, 255, 0],
                        [255, 31, 0], [255, 224, 0], [
                            153, 255, 0], [0, 0, 255],
                        [255, 71, 0], [0, 235, 255], [
                            0, 173, 255], [31, 0, 255],
                        [11, 200, 200], [255, 82, 0], [
                            0, 255, 245], [0, 61, 255],
                        [0, 255, 112], [0, 255, 133], [
                            255, 0, 0], [255, 163, 0],
                        [255, 102, 0], [194, 255, 0], [
                            0, 143, 255], [51, 255, 0],
                        [0, 82, 255], [0, 255, 41], [
                            0, 255, 173], [10, 0, 255],
                        [173, 255, 0], [0, 255, 153], [
                            255, 92, 0], [255, 0, 255],
                        [255, 0, 245], [255, 0, 102], [
                            255, 173, 0], [255, 0, 20],
                        [255, 184, 184], [0, 31, 255], [
                            0, 255, 61], [0, 71, 255],
                        [255, 0, 204], [0, 255, 194], [
                            0, 255, 82], [0, 10, 255],
                        [0, 112, 255], [51, 0, 255], [
                            0, 194, 255], [0, 122, 255],
                        [0, 255, 163], [255, 153, 0], [
                            0, 255, 10], [255, 112, 0],
                        [143, 255, 0], [82, 0, 255], [
                            163, 255, 0], [255, 235, 0],
                        [8, 184, 170], [133, 0, 255], [
                            0, 255, 92], [184, 0, 255],
                        [255, 0, 31], [0, 184, 255], [
                            0, 214, 255], [255, 0, 112],
                        [92, 255, 0], [0, 224, 255], [
                            112, 224, 255], [70, 184, 160],
                        [163, 0, 255], [153, 0, 255], [
                            71, 255, 0], [255, 0, 163],
                        [255, 204, 0], [255, 0, 143], [
                            0, 255, 235], [133, 255, 0],
                        [255, 0, 235], [245, 0, 255], [
                            255, 0, 122], [255, 245, 0],
                        [10, 190, 212], [214, 255, 0], [
                            0, 204, 255], [20, 0, 255],
                        [255, 255, 0], [0, 153, 255], [
                            0, 41, 255], [0, 255, 204],
                        [41, 0, 255], [41, 255, 0], [
                            173, 0, 255], [0, 245, 255],
                        [71, 0, 255], [122, 0, 255], [
                            0, 255, 184], [0, 92, 255],
                        [184, 255, 0], [0, 133, 255], [
                            255, 214, 0], [25, 194, 194],
                        [102, 255, 0], [92, 0, 255]]

        config_file = '/home/ladanova_sv/mmsegmentation/configs/segformer/segformer_mit-b1_8xb1-160k_cityscapes-1024x1024.py'
        checkpoint_file = '/home/ladanova_sv/mmsegmentation/checkpoints/segformer_mit-b1_512x512_160k_ade20k_20220620_112037-c3f39e00.pth'
        #config_file = 'SegFormer/local_configs/segformer/B4/segformer.b4.512x512.ade.160k.py'
        # self.model = init_model(config_file, checkpoint_file, device='cuda:0')
        #
        self.model = MMSegInferencer(
            model='segformer_mit-b0_8xb2-160k_ade20k-512x512', device='cpu')
        self.clss = ade_classes()

    def __call__(self, image, cats_to_disp):
        # result = inference_model(self.model, image)

        # im = Image.fromarray(image)
        # im.save(
        #     "/home/ladanova_sv/semantic_mapper_ws/src/semantic_mapper/scripts/image.png")
        # cv2.imwrite('image/demo.png', image)
        result = self.model(image)['predictions']

        # vis_result = show_result_pyplot(self.model, image, result)
        # print(vis_result)
        seg_image = np.zeros(
            (result.shape[0], result.shape[1], 3), dtype=np.uint8)
        for i, cat in enumerate(self.clss):
            if cat in cats_to_disp:
                seg_image[result == i] = self.PALETTE[i]
        return seg_image
