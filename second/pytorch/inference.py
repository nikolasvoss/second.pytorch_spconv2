from pathlib import Path

import numpy as np
import torch

import torchplus
from second.core import box_np_ops
from second.core.inference import InferenceContext
from second.builder import target_assigner_builder, voxel_builder
from second.pytorch.builder import box_coder_builder, second_builder
from second.pytorch.models.voxelnet import VoxelNet
from second.pytorch.train import predict_to_kitti_label, example_convert_to_torch


class TorchInferenceContext(InferenceContext):
    def __init__(self):
        super().__init__()
        self.net = None
        self.anchor_cache = None

    def _build(self):
        config = self.config
        input_cfg = config.eval_input_reader
        model_cfg = config.model.second
        train_cfg = config.train_config
        batch_size = 1
        voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
        bv_range = [
            voxel_generator.coors_range[0],
            voxel_generator.coors_range[1],
            voxel_generator.coors_range[3],
            voxel_generator.coors_range[4]
        ]
        grid_size = voxel_generator.grid_size
        self.voxel_generator = voxel_generator
        vfe_num_filters = list(model_cfg.voxel_feature_extractor.num_filters)

        box_coder = box_coder_builder.build(model_cfg.box_coder)
        target_assigner_cfg = model_cfg.target_assigner
        target_assigner = target_assigner_builder.build(
            target_assigner_cfg, bv_range, box_coder)
        self.target_assigner = target_assigner
        out_size_factor = model_cfg.rpn.layer_strides[0] / model_cfg.rpn.upsample_strides[0]
        out_size_factor *= model_cfg.middle_feature_extractor.downsample_factor
        out_size_factor = int(out_size_factor)
        self.net = second_builder.build(model_cfg, voxel_generator,
                                          target_assigner)
        self.net.cuda().eval()
        if train_cfg.enable_mixed_precision:
            self.net.half()
            self.net.metrics_to_float()
            self.net.convert_norm_to_float(self.net)
        feature_map_size = grid_size[:2] // out_size_factor
        feature_map_size = [*feature_map_size, 1][::-1]
        ret = target_assigner.generate_anchors(feature_map_size)
        anchors_dict = target_assigner.generate_anchors_dict(feature_map_size)
        anchors = ret["anchors"]
        anchors = anchors.reshape([-1, 7])
        matched_thresholds = ret["matched_thresholds"]
        unmatched_thresholds = ret["unmatched_thresholds"]
        anchors_bv = box_np_ops.rbbox2d_to_near_bbox(
            anchors[:, [0, 1, 3, 4, 6]])
        anchor_cache = {
            "anchors": anchors,
            "anchors_bv": anchors_bv,
            "matched_thresholds": matched_thresholds,
            "unmatched_thresholds": unmatched_thresholds,
            "anchors_dict": anchors_dict,
        }
        self.anchor_cache = anchor_cache

    def _restore(self, ckpt_path):
        ckpt_path = Path(ckpt_path)
        assert ckpt_path.suffix == ".tckpt"
        torchplus.train.restore(str(ckpt_path), self.net)

    def _inference(self, example):
        train_cfg = self.config.train_config
        input_cfg = self.config.eval_input_reader
        model_cfg = self.config.model.second
        example_torch = example_convert_to_torch(example)
        result_annos = predict_to_kitti_label(
            self.net, example_torch, list(
                self.target_assigner.classes),
            model_cfg.post_center_limit_range, model_cfg.lidar_input)
        return result_annos

    def _ctx(self):
        return None
