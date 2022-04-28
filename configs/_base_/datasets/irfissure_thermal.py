# dataset settings
dataset_type = 'IRFissure'
data_root = 'data/custom_dataset/IR_fissure/thermal'
img_suffix = '.png'
image_load_cfg = dict(to_float32=True, color_type='unchanged')
crop_size = (512, 512) # reduce crop size to avoid cuda memory issue, have to match with model input (if fine-tune from a pretrained model)
img_scale = (640, 512)
train_pipeline = [
    dict(type='LoadImageFromFile', **image_load_cfg),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=img_scale, ratio_range=(1., 1.5)), # imagine scale better larger than crop size after resized
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile', **image_load_cfg),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2, # batch size
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_suffix=img_suffix,
        img_dir='images',
        ann_dir='annotes',
        split='splits/train.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_suffix=img_suffix,
        img_dir='images',
        ann_dir='annotes',
        split='splits/val.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_suffix=img_suffix,
        img_dir='images',
        ann_dir='annotes',
        split='splits/test.txt',
        pipeline=test_pipeline))
