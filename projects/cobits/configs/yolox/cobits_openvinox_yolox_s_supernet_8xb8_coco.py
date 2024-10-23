_base_ = [
    'mmdet::yolox/yolox_s_8xb8-300e_coco.py',
]

# _base_.data_preprocessor.type = 'mmdet.DetDataPreprocessor'
_base_.model.backbone.conv_cfg = dict(type='mmrazor.BigNasConv2d')
_base_.model.backbone.norm_cfg = dict(type='mmrazor.DynamicBatchNorm2d', momentum=0.03, eps=0.001)
_base_.model.neck.conv_cfg = dict(type='mmrazor.BigNasConv2d')
_base_.model.neck.norm_cfg = dict(type='mmrazor.DynamicBatchNorm2d', momentum=0.03, eps=0.001)
_base_.model.bbox_head.conv_cfg = dict(type='mmrazor.BigNasConv2d')
_base_.model.bbox_head.norm_cfg = dict(type='mmrazor.DynamicBatchNorm2d', momentum=0.03, eps=0.001)
# _base_.model.head.type = 'mmrazor.DynamicLinearClsHead'
_base_.model.init_cfg = dict(
    type='Pretrained',
    checkpoint=  # noqa: E251
'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth')

architecture = _base_.model

_base_.train_dataloader.dataset.pipeline = [
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    # Resize and Pad are for the last 15 epochs when Mosaic,
    # RandomAffine, and MixUp are closed by YOLOXModeSwitchHook.
    dict(type='Resize', scale=_base_.img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='PackDetInputs')
]

global_qconfig = dict(
    w_observer=dict(type='mmrazor.PerChannelBatchLSQObserver'),
    a_observer=dict(type='mmrazor.BatchLSQObserver'),
    w_fake_quant=dict(type='mmrazor.DynamicBatchLearnableFakeQuantize'),
    a_fake_quant=dict(type='mmrazor.DynamicBatchLearnableFakeQuantize'),
    w_qscheme=dict(qdtype='qint8', bit=4, is_symmetry=True, is_symmetric_range=False, extreme_estimator=1, param_share_mode=4),
    a_qscheme=dict(qdtype='quint8', bit=4, is_symmetry=True, extreme_estimator=1, param_share_mode=4)
)
# Make sure that the architecture and qmodels have the same data_preprocessor.
qmodel = dict(
    _scope_='mmdet',
    type='mmrazor.MMArchitectureQuant',
    data_preprocessor=_base_.model.data_preprocessor,
    architecture=architecture,
    float_checkpoint=None,
    forward_modes=('tensor', 'predict', 'loss'),
    quantizer=dict(
        type='mmrazor.OpenVINOXQuantizer',
        quant_bits_skipped_module_names=[
            'backbone.stem.conv.conv',
            'bbox_head.multi_level_conv_cls.2',
            'bbox_head.multi_level_conv_reg.2',
            'bbox_head.multi_level_conv_obj.2'
        ],
        w_bits=[4,5,6,7,8],
        a_bits=[4,5,6,7,8],
        global_qconfig=global_qconfig,
        tracer=dict(
            type='mmrazor.CustomTracer',
            skipped_module_classes=[
                'mmrazor.models.architectures.dynamic_ops.bricks.dynamic_conv.BigNasConv2d',
                'mmrazor.models.architectures.dynamic_ops.bricks.dynamic_function.DynamicInputResizer',
                'mmrazor.models.architectures.dynamic_ops.bricks.dynamic_linear.DynamicLinear',
                'mmrazor.models.architectures.dynamic_ops.bricks.dynamic_norm.DynamicBatchNorm2d',
            ],
            skipped_methods=[
                'mmdet.models.dense_heads.yolox_head.YOLOXHead.predict_by_feat',  # noqa: E501
                'mmdet.models.dense_heads.yolox_head.YOLOXHead.loss_by_feat',
            ])))

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='QNAS',
    num_random_samples=2,
    architecture=qmodel,
    mutator=dict(type='mmrazor.NasMutator'))

# train_dataloader = dict(batch_size=64)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=0.0005, momentum=0.9, weight_decay=5e-4,
        nesterov=True),
    paramwise_cfg=dict(bias_decay_mult=0., norm_decay_mult=0., bypass_duplicate=True),
)

model_wrapper_cfg = dict(
    type='mmrazor.QNASDDP',
    broadcast_buffers=False,
    find_unused_parameters=True)

# learning policy
max_epochs = 3
warm_epochs = 1
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
train_cfg = dict(
    _delete_=True,
    type='mmrazor.QNASEpochBasedLoop',
    max_epochs=max_epochs,
    val_interval=1,
    qat_begin=1,
    calibrate_steps=100,
    freeze_bn_begin=-1)

# total calibrate_sample_num = 256 * 8 * 2
val_cfg = dict(_delete_=True, type='mmrazor.QNASValLoop', calibrate_sample_num=200,
               quant_bits=[4,5,6,7,8], only_quantized=True)
# Make sure the buffer such as min_val/max_val in saved checkpoint is the same
# among different rank.
default_hooks = dict(sync=dict(type='SyncBuffersHook'), checkpoint=dict(interval=1))
custom_hooks = []
