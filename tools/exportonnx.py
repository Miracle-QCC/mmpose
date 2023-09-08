# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import torch
import torch.nn as nn
import onnx
from onnxsim import simplify

from mmpose.apis import init_model


class RTMPoseWithDecode(nn.Module):
    def __init__(self, config, checkpoint):
        super().__init__()
        self.detector = init_model(config, checkpoint, 'cpu')

    def forward(self, x):
        simcc_x, simcc_y = self.detector.forward(x, None)
        # max_val_x, x_locs = torch.max(simcc_x, dim=2)
        # max_val_y, y_locs = torch.max(simcc_y, dim=2)
        # scores = torch.maximum(max_val_x, max_val_y)
        # keypoints = torch.stack([x_locs, y_locs], dim=-1)
        # keypoints = keypoints.float() / self.detector.cfg.codec.simcc_split_ratio

        return simcc_x, simcc_y


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Config file', default='/home/qcj/workcode/mmpose/configs/face_2d_keypoint/rtmpose/face6/rtmpose-t_8xb256-120e_face6_68kpt-256x256.py')
    parser.add_argument('--checkpoint', help='Checkpoint file',default='/home/qcj/workcode/mmpose/work_dirs/rtmpose-t_8xb256-120e_face6_68kpt-256x256/epoch_110.pth')
    parser.add_argument('--save_path', help='onnx save path', default="../onnx/rtmpose_t_68.onnx")
    parser.add_argument(
        '--input_size',
        type=int,
        nargs=2,
        default=[256, 256],
        help='network input size')
    parser.add_argument('--opset', type=int, default=11, help='opset version')
    args = parser.parse_args()
    return args


def export(args):
    model = RTMPoseWithDecode(args.config, args.checkpoint)
    model.eval()
    dummy_image = torch.zeros((1, 3, 256,256), device='cpu')

    torch.onnx.export(
        model,
        dummy_image,
        args.save_path,
        keep_initializers_as_inputs=False,
        verbose=False,
        input_names=['input'],
        dynamic_axes=None,
        output_names=['simcc_x', 'simcc_y'],
        opset_version=11
        )

    # 使用onnx simplify简化模型，当前没用
    # onnx_model = onnx.load(args.save_path)
    # onnx_model_simp, check = simplify(onnx_model)
    # assert check, 'Simplified ONNX model could not be validated'
    # onnx.save(onnx_model_simp, args.save_path)


if __name__ == '__main__':
    args = parse_args()
    export(args)