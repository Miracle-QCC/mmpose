_base_ = ['../../../_base_/default_runtime.py']

# lapa coco wflw 300w cofw halpe

# runtime
max_epochs = 120
stage2_num_epochs = 10
base_lr = 4e-3

train_cfg = dict(max_epochs=max_epochs, val_interval=1)
randomness = dict(seed=21)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.),
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
        T_max=90,
        by_epoch=True,
        convert_to_iter_based=True),
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# codec settings
codec = dict(
    type='SimCCLabel',
    input_size=(256, 256),
    sigma=(5.66, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    # backbone=dict(
    #     _scope_='mmdet',
    #     type='CSPNeXt',
    #     arch='P5',
    #     expand_ratio=0.5,
    #     deepen_factor=0.167,
    #     widen_factor=0.375,
    #     out_indices=(4, ),
    #     channel_attention=True,
    #     norm_cfg=dict(type='SyncBN'),
    #     act_cfg=dict(type='SiLU'),
    #     init_cfg=dict(
    #         type='Pretrained',
    #         prefix='backbone.',
    #         checkpoint='https://download.openmmlab.com/mmdetection/v3.0/'
    #         'rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e-3a2dd350.pth'  # noqa
    #     )),
    backbone=dict(
        _scope_='mmdet',
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.167,
        widen_factor=0.25,
        out_indices=(4,),
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmdetection/v3.0/'
                       'rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e-3a2dd350.pth'  # noqa
        )),
    head=dict(
        type='RTMCCHead',
        in_channels=256,
        out_channels=68,
        input_size=codec['input_size'],
        in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
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
    test_cfg=dict(flip_test=True, ))

# base dataset settings
dataset_type = 'Face300WDataset'
data_mode = 'topdown'
data_root = '/opt/data/DMS/face6_data/data/'

backend_args = dict(backend='local')

# pipelines
train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform', scale_factor=[0.5, 1.5], rotate_factor=80),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.2),
            dict(type='MedianBlur', p=0.2),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=1.0),
        ]),
    dict(
        type='GenerateTarget',
        encoder=codec,
        use_dataset_keypoint_weights=True),
    dict(type='PackPoseInputs')
]



train_pipeline_stage2 = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.,
        scale_factor=[0.75, 1.25],
        rotate_factor=60),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=0.5),
        ]),
    dict(
        type='GenerateTarget',
        encoder=codec,
        use_dataset_keypoint_weights=True),
    dict(type='PackPoseInputs')
]

kpt_106_to_68 = [
    # cheek
    (0, 0),
    (2, 1),
    (4, 2),
    (6, 3),
    (8, 4),
    (10, 5),
    (12, 6),
    (14, 7),
    (16, 8),
    (18, 9),
    (20, 10),
    (22, 11),
    (24, 12),
    (26, 13),
    (28, 14),
    (30, 15),
    (32, 16),
    # Left eyebrow
    (33, 17),
    (34, 18),
    (35, 19),
    (36, 20),
    (37, 21),
    # Right eyebrow
    (42, 22),
    (43, 23),
    (44, 24),
    (45, 25),
    (46, 26),
    # nose
    (51, 27),
    (52, 28),
    (53, 29),
    (54, 30),
    (57, 31),
    (59, 32),
    (60, 33),
    (62, 34),
    (63, 35),
    # left eye
    (66, 36),
    ((67,68), 37),
    ((68,69), 38),
    (70, 39),
    ((71,72), 40),
    ((72,73), 41),
    # right eye
    (75, 42),
    ((76,77), 43),
    ((77,78), 44),
    (79, 45),
    ((80,81), 46),
    ((81,82), 47),
    # mouse
    (84, 48),
    (85, 49),
    (86, 50),
    (87, 51),
    (88, 52),
    (89, 53),
    (90, 54),
    (91, 55),
    (92, 56),
    (93, 57),
    (94, 58),
    (95, 59),
    (96, 60),
    (97, 61),
    (98, 62),
    (99, 63),
    (100, 64),
    (101, 65),
    (102, 66),
    (103, 67),
]

mapping_halpe = [
    # CHEEK
    (26, 0),
    (27, 1),
    (28, 2),
    (29, 3),
    (30, 4),
    (31, 5),
    (32, 6),
    (33, 7),
    (34, 8),
    (35, 9),
    (36, 10),
    (37, 11),
    (38, 12),
    (39, 13),
    (40, 14),
    (41, 15),
    (42, 16),
    #Left eyebrow
    (43, 17),
    (44, 18),
    (45, 19),
    (46, 20),
    (47, 21),
    #right eyebrow
    (48, 22),
    (49, 23),
    (50, 24),
    (51, 25),
    (52, 26),
    #nose
    (53, 27),
    (54, 28),
    (55, 29),
    (56, 30),
    (57, 31),
    (58, 32),
    (59, 33),
    (60, 34),
    (61, 35),
    #left eye
    (62, 36),
    (63, 37),
    (64, 38),
    (65, 39),
    (66, 40),
    (67, 41),
    # right eye
    (68, 42),
    (69, 43),
    (70, 44),
    (71, 45),
    (72, 46),
    (73, 47),
    # mouth
    (74, 48),
    (75, 49),
    (76, 50),
    (77, 51),
    (78, 52),
    (79, 53),
    (80, 54),
    (81, 55),
    (82, 56),
    (83, 57),
    (84, 58),
    (85, 59),
    (86, 60),
    (87, 61),
    (88, 62),
    (89, 63),
    (90, 64),
    (91, 65),
    (92, 66),
    (93, 67),
]

mapping_wflw = [
    # cheek
    (0, 0),
    (2, 1),
    (4, 2),
    (6, 3),
    (8, 4),
    (10, 5),
    (12, 6),
    (14, 7),
    (16, 8),
    (18, 9),
    (20, 10),
    (22, 11),
    (24, 12),
    (26, 13),
    (28, 14),
    (30, 15),
    (32, 16),
    # Left eyebrow
    (33, 17),
    (34, 18),
    (35, 19),
    (36, 20),
    (37, 21),
    #Right eyebrow
    (42, 22),
    (43, 23),
    (44, 24),
    (45, 25),
    (46, 26),
    #Nose
    (51, 27),
    (52, 28),
    (53, 29),
    (54, 30),
    (55, 31),
    (56, 32),
    (57, 33),
    (58, 34),
    (59, 35),
    # left eye
    (60, 36),
    (61, 37),
    (63, 38),
    (64, 39),
    (65, 40),
    (67, 41),
    # right eye
    (68, 42),
    (69, 43),
    (71, 44),
    (72, 45),
    (73, 46),
    (75, 47),
    # mouse
    (76, 48),
    (77, 49),
    (78, 50),
    (79, 51),
    (80, 52),
    (81, 53),
    (82, 54),
    (83, 55),
    (84, 56),
    (85, 57),
    (86, 58),
    (87, 59),
    (88, 60),
    (89, 61),
    (90, 62),
    (91, 63),
    (92, 64),
    (93, 65),
    (94, 66),
    (95, 67),
]

mapping_cofw = [
    #
    (0, 17),
    (4, 19),
    #
    (1, 26),
    (6, 24),

    #
    (8, 36),
    (10, 39),

    #
    (9, 45),
    (11, 42),

    #
    (20, 30),
    #
    (22, 48),
    (23, 54),
    (24, 51),
    (25, 62),
    (26, 66),
    (27, 57),
    #
    (28, 8)
]
val_pipeline = [
    dict(type='KeypointConverter', num_keypoints=68, mapping=kpt_106_to_68),
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

# train dataset
dataset_lapa = dict(
    type=dataset_type,
    data_root=data_root,
    data_mode=data_mode,
    ann_file='LaPa/annotations/lapa_trainval.json',
    data_prefix=dict(img='/opt/data/DMS/face6_data/data/LaPa/'),
    pipeline=[dict(
            type='KeypointConverter', num_keypoints=68, mapping=kpt_106_to_68)
    ],
)

dataset_coco = dict(
    type='CocoWholeBodyFaceDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='coco/annotations/coco_wholebody_train_v1.0.json',
    data_prefix=dict(img='/opt/data/DMS/face6_data/data/coco/train2017/'),
)


dataset_wflw = dict(
    type='WFLWDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='WFLW/annotations/WFLW_train.json',
    data_prefix=dict(img='/opt/data/DMS/face6_data/data/WFLW/train/'),
    pipeline=[
        dict(
            type='KeypointConverter', num_keypoints=68, mapping=mapping_wflw)
    ],
)

dataset_300w = dict(
    type='Face300WDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='300W/annotations/300W_train.json',
    data_prefix=dict(img='/opt/data/DMS/face6_data/data/'),
)

dataset_cofw = dict(
    type='COFWDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='cofw/annotations/cofw_train.json',
    data_prefix=dict(img='/opt/data/DMS/face6_data/data/cofw/images/'),
    pipeline=[
        dict(
            type='KeypointConverter', num_keypoints=68, mapping=mapping_cofw)
    ],
)

dataset_halpe = dict(
    type='HalpeDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='halpe/halpe_train_v1.json',
    data_prefix=dict(img='/opt/data/DMS/face6_data/data/halpe/images/train2015/'),
    pipeline=[
        dict(
            type='KeypointConverter', num_keypoints=68, mapping=mapping_halpe)
    ],
)

# data loaders
train_dataloader = dict(
    batch_size=256,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CombinedDataset',
        metainfo=dict(from_file='configs/_base_/datasets/300w.py'),
        datasets=[
            dataset_lapa, dataset_coco, dataset_wflw, dataset_300w,
            dataset_cofw, dataset_halpe
        ],
        pipeline=train_pipeline,
        test_mode=False,
    ))
val_dataloader = dict(
    batch_size=32,
    num_workers=10,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='LaPa/annotations/lapa_test.json',
        data_prefix=dict(img='/opt/data/DMS/face6_data/data/LaPa/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))

# test dataset
val_lapa = dict(
    type=dataset_type,
    data_root=data_root,
    data_mode=data_mode,
    ann_file='LaPa/annotations/lapa_test.json',
    data_prefix=dict(img='/opt/data/DMS/face6_data/data/LaPa/'),
    pipeline=[dict(type='KeypointConverter', num_keypoints=68, mapping=kpt_106_to_68)
    ],
)

val_coco = dict(
    type='CocoWholeBodyFaceDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='coco/annotations/coco_wholebody_val_v1.0.json',
    data_prefix=dict(img='/opt/data/DMS/face6_data/data/coco/val2017/'),
    pipeline=[
        dict(
            type='KeypointConverter', num_keypoints=68, mapping=kpt_106_to_68)
    ],
)

val_wflw = dict(
    type='WFLWDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='WFLW/annotations/WFLW_test.json',
    data_prefix=dict(img='/opt/data/DMS/face6_data/data/WFLW/test/'),
    pipeline=[
        dict(
            type='KeypointConverter', num_keypoints=68, mapping=mapping_wflw)
    ],
)

val_300w = dict(
    type='Face300WDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='300w/annotations/face_landmarks_300w_test.json',
    data_prefix=dict(img='pose/300w/images/'),
    pipeline=[
    ],
)

val_cofw = dict(
    type='COFWDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='cofw/annotations/cofw_test.json',
    data_prefix=dict(img='/opt/data/DMS/face6_data/data/cofw/images/'),
    pipeline=[
        dict(
            type='KeypointConverter', num_keypoints=68, mapping=mapping_cofw)
    ],
)

val_halpe = dict(
    type='HalpeDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='halpe/halpe_val_v1.json',
    data_prefix=dict(img='/opt/data/DMS/face6_data/data/halpe/images/test2015/'),
    pipeline=[
        dict(
            type='KeypointConverter', num_keypoints=68, mapping=mapping_halpe)
    ],
)

test_dataloader = dict(
    batch_size=32,
    num_workers=10,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CombinedDataset',
        metainfo=dict(from_file='configs/_base_/datasets/300w.py'),
        datasets=[val_lapa, val_coco, val_wflw, val_300w, val_cofw, val_halpe],
        pipeline=val_pipeline,
        test_mode=True,
    ))

# hooks
default_hooks = dict(
    checkpoint=dict(
        save_best='NME', rule='less', max_keep_ckpts=1, interval=1))

custom_hooks = [
    # dict(
    #     type='EMAHook',
    #     ema_type='ExpMomentumEMA',
    #     momentum=0.0002,
    #     update_buffers=True,
    #     priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

# evaluators
val_evaluator = dict(
    type='NME',
    norm_mode='keypoint_distance',
)
test_evaluator = val_evaluator
