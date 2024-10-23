_base_ = [
    '../../../cobits/configs/resnet18/resnet18_8xb256-warmup-lbs-coslr_in1k.py',
]

_base_.data_preprocessor.type = 'mmcls.ClsDataPreprocessor'
_base_.model.backbone.conv_cfg = dict(type='mmrazor.BigNasConv2d')
_base_.model.backbone.norm_cfg = dict(type='mmrazor.DynamicBatchNorm2d')
_base_.model.head.type = 'mmrazor.DynamicLinearClsHead'
_base_.model.init_cfg = dict(
    type='Pretrained',
    checkpoint=  # noqa: E251
'https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth')

architecture = _base_.model

global_qconfig = dict(
    w_observer=dict(type='mmrazor.LSQObserver'),
    a_observer=dict(type='mmrazor.LSQObserver'),
    w_fake_quant=dict(type='mmrazor.DynamicLearnableFakeQuantize'),
    a_fake_quant=dict(type='mmrazor.DynamicLearnableFakeQuantize'),
    w_qscheme=dict(qdtype='qint8', bit=4, is_symmetry=True, param_share_mode=0),
    a_qscheme=dict(qdtype='quint8', bit=4, is_symmetry=True, param_share_mode=0)
)
# Make sure that the architecture and qmodels have the same data_preprocessor.
qmodel = dict(
    _scope_='mmcls',
    type='mmrazor.MMArchitectureQuant',
    data_preprocessor=_base_.data_preprocessor,
    architecture=architecture,
    float_checkpoint=None,
    forward_modes=('tensor', 'predict', 'loss'),
    quantizer=dict(
        type='mmrazor.WeightOnlyQuantizer',
        quant_bits_skipped_module_names=[
            'backbone.conv1',
            'head.fc'
        ],
        w_bits=[2,3,4,5,6],
        a_bits=[2,3,4,5,6],
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
                'mmcls.models.heads.ClsHead._get_loss',
                'mmcls.models.heads.ClsHead._get_predictions'
            ])))

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='QNAS',
    num_random_samples=2,
    architecture=qmodel,
    use_spos=False,
    mutator=dict(type='mmrazor.NasMutator'))

train_dataloader = dict(batch_size=64)
optim_wrapper = dict(
    _delete_=True,
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001, nesterov=True),
    paramwise_cfg=dict(
        bypass_duplicate=True
    ),)

model_wrapper_cfg = dict(
    type='mmrazor.QNASDDP',
    broadcast_buffers=False,
    find_unused_parameters=True)

# learning policy
max_epochs = 5
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
    val_interval=5,
    qat_begin=1,
    freeze_bn_begin=1)

# total calibrate_sample_num = 256 * 8 * 2
val_cfg = dict(_delete_=True, type='mmrazor.QNASValLoop', calibrate_sample_num=0,
               quant_bits=[2,3,4,5,6], only_quantized=True)
# Make sure the buffer such as min_val/max_val in saved checkpoint is the same
# among different rank.
default_hooks = dict(sync=dict(type='SyncBuffersHook'))
