_base_ = ['mmcls::mobilenet_v2/mobilenet-v2_8xb32_in1k.py']
model = dict(
    _scope_='mmcls',
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1280,
        loss=dict(
            type='LabelSmoothLoss',
            loss_weight=1.0,
            label_smooth_val=0.1,
            num_classes=1000),
    ))

# _base_.train_dataloader.dataset.pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='RandomResizedCrop', scale=224, backend='pillow'),
#     dict(type='RandomFlip', prob=0.5, direction='horizontal'),
#     dict(type='ColorJitter', brightness=0.2, contrast=0.2, saturation=0.2),
#     dict(type='PackClsInputs'),
# ]
# _base_.train_dataloader.batch_size=128
# optim_wrapper = dict(
#     optimizer=dict(
#         type='SGD', lr=0.4, momentum=0.9, weight_decay=4e-5, nesterov=True),
#     paramwise_cfg=dict(bias_decay_mult=0., norm_decay_mult=0.)
# )

# # learning policy
# param_scheduler = [
#     # warm up learning rate scheduler
#     dict(
#         type='LinearLR',
#         start_factor=0.25,
#         by_epoch=True,
#         begin=0,
#         # about 2500 iterations for ImageNet-1k
#         end=5,
#         # update by iter
#         convert_to_iter_based=True),
#     # main learning rate scheduler
#     dict(
#         type='CosineAnnealingLR',
#         T_max=245,
#         by_epoch=True,
#         begin=5,
#         end=250,
#     )
# ]

# # train, val, test setting
# train_cfg = dict(by_epoch=True, max_epochs=250, val_interval=5)
# val_cfg = dict()
# test_cfg = dict()

# # NOTE: `auto_scale_lr` is for automatically scaling LR,
# # based on the actual training batch size.
# auto_scale_lr = dict(base_batch_size=1024)

# # custom_hooks = [dict(type='EMAHook', momentum=5e-4, priority='ABOVE_NORMAL', update_buffers=True)]
default_hooks = dict(checkpoint=dict(save_best=None, max_keep_ckpts=1))