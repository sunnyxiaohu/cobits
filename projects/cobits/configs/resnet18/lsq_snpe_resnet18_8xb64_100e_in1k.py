_base_ = ['./resnet18_8xb256-warmup-lbs-coslr_in1k.py']

resnet = _base_.model
float_checkpoint = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth'  # noqa: E501

global_qconfig = dict(
    w_observer=dict(type='mmrazor.LSQObserver'),
    a_observer=dict(type='mmrazor.LSQObserver'),
    w_fake_quant=dict(type='mmrazor.LearnableFakeQuantize'),
    a_fake_quant=dict(type='mmrazor.LearnableFakeQuantize'),
    w_qscheme=dict(qdtype='qint8', bit=4, is_symmetry=False, zero_point_trainable=True),
    a_qscheme=dict(qdtype='qint8', bit=4, is_symmetry=False, zero_point_trainable=True),
)

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='MMArchitectureQuant',
    data_preprocessor=dict(
        type='mmcls.ClsDataPreprocessor',
        num_classes=1000,
        # RGB format normalization parameters
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        # convert image from BGR to RGB
        to_rgb=True),
    architecture=resnet,
    float_checkpoint=float_checkpoint,
    quantizer=dict(
        type='mmrazor.SNPEQuantizer',
        quant_bits_skipped_module_names=[
            'backbone.conv1',
            'head.fc'
        ],
        global_qconfig=global_qconfig,
        tracer=dict(
            type='mmrazor.CustomTracer',
            skipped_methods=[
                'mmcls.models.heads.ClsHead._get_loss',
                'mmcls.models.heads.ClsHead._get_predictions'
            ])))

train_dataloader = dict(batch_size=64)

optim_wrapper = dict(
    _delete_=True,
    optimizer=dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001, nesterov=True))

# learning policy
max_epochs = 100
warm_epochs = 5
# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=0.025,
        by_epoch=True,
        begin=0,
        # about 2500 iterations for ImageNet-1k
        end=warm_epochs,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=max_epochs-warm_epochs,
        by_epoch=True,
        begin=warm_epochs,
        end=max_epochs,
    ),
]

model_wrapper_cfg = dict(
    type='mmrazor.MMArchitectureQuantDDP',
    broadcast_buffers=False,
    find_unused_parameters=True)

# train, val, test setting
train_cfg = dict(
    _delete_=True,
    type='mmrazor.LSQEpochBasedLoop',
    max_epochs=max_epochs,
    val_interval=5,
    freeze_bn_begin=-1)
val_cfg = dict(_delete_=True, type='mmrazor.QATValLoop')
test_cfg = val_cfg

# Make sure the buffer such as min_val/max_val in saved checkpoint is the same
# among different rank.
default_hooks = dict(sync=dict(type='SyncBuffersHook'))
