_base_ = ['mmcls::resnet/resnet18_8xb32_in1k.py']
model = dict(
    _scope_='mmcls',
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=512,
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
# _base_.train_dataloader.batch_size = 256
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(edge='short', scale=256, type='ResizeEdge', backend='pillow'),
#     dict(crop_size=224, type='CenterCrop'),
#     dict(type='PackClsInputs')
# ]
# _base_.val_dataloader.dataset.pipeline = test_pipeline
# _base_.test_dataloader.dataset.pipeline = test_pipeline
# optim_wrapper = dict(
#     _delete_=True,
#     optimizer=dict(type='SGD', lr=0.8, momentum=0.9, weight_decay=0.0001, nesterov=True))
default_hooks = dict(checkpoint=dict(save_best=None, max_keep_ckpts=1))
