_base_ = [
    '../_base_/models/pspnet_r50-d8.py',
    '../_base_/datasets/irfissure_thumb.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]

norm_cfg = dict(type='BN', requires_grad=True)
crop_size = (256, 256)
img_scale = (320, 256)
model = dict(
    backbone=dict(norm_cfg=norm_cfg),
    decode_head=dict(sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000), num_classes=2, norm_cfg=norm_cfg),
    auxiliary_head=dict(num_classes=2, norm_cfg=norm_cfg),
    test_cfg=dict(mode='whole'))
optimizer = dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001)

data = dict(samples_per_gpu=6, workers_per_gpu=1)