import torch
from thop import profile

from mmpose.models import build_pose_estimator
from mmpose.models import PoseDataPreprocessor
from mmpose.structures import PoseDataSample

_base_ = ['mmpose::_base_/default_runtime.py']

# common setting
num_keypoints = 68
input_size = (256, 256)

# runtime
max_epochs = 120
stage2_num_epochs = 10
base_lr = 4e-3
train_batch_size = 256
val_batch_size = 32

train_cfg = dict(max_epochs=max_epochs, val_interval=1)
randomness = dict(seed=21)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    clip_grad=dict(max_norm=35, norm_type=2),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.005,
        begin=30,
        end=max_epochs,
        T_max=max_epochs - 30,
        by_epoch=True,
        convert_to_iter_based=True),
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# codec settings
codec = dict(
    type='SimCCLabel',
    input_size=input_size,
    sigma=(5.66, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    backbone=dict(
        _scope_='mmdet',
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.167,
        widen_factor=0.25,
        out_indices=(4,),
        channel_attention=True,
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmdetection/v3.0/'
                       'rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e-3a2dd350.pth'  # noqa
        )),
    # backbone=dict(
    #     type='MobileNetV2',
    #     widen_factor=1.,
    #     out_indices=(7,),
    #     init_cfg=dict(
    #         type='Pretrained',
    #         checkpoint='mmcls://mobilenet_v2',
    #     )),
    head=dict(
        type='RTMCCHead',
        in_channels=256,
        out_channels=num_keypoints,
        input_size=codec['input_size'],
        in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=0.5,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=dict(
            type='KLDiscretLoss',
            use_target_weight=True,
            beta=10.,
            label_softmax=True),
        decoder=codec),
    # test_cfg=dict(flip_test=True,
)


data_samples = []
data_sample = PoseDataSample()
data_sample.set_metainfo(
    dict(target_img_path='tests/data/h36m/S7/'
         'S7_Greeting.55011271/S7_Greeting.55011271_000396.jpg'))

net = build_pose_estimator(model)
net.eval()
data = torch.rand((1,3,256,256))
with torch.no_grad():
    out = net.forward(data, data_samples)
    flops, params = profile(net, (data, data_samples))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.3f G, params: %.3f M\n' % (flops / 1000000000.0, params / 1000000.0))
