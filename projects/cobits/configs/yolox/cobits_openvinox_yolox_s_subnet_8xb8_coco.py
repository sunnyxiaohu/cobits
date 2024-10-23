_base_ = './cobits_openvinox_yolox_s_supernet_8xb8_coco.py'

global_qconfig = dict(
    w_observer=dict(type='mmrazor.LSQPerChannelObserver'),
    a_observer=dict(type='mmrazor.LSQObserver'),
    w_fake_quant=dict(type='mmrazor.LearnableFakeQuantize'),
    a_fake_quant=dict(type='mmrazor.LearnableFakeQuantize'),
    w_qscheme=dict(qdtype='qint8', bit=8, is_symmetry=True, is_symmetric_range=False),
    a_qscheme=dict(qdtype='quint8', bit=8, is_symmetry=True),
)

qmodel = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='sub_model',
    cfg=_base_.architecture,
    # NOTE: You can replace the yaml with the mutable_cfg searched by yourself
    fix_subnet='work_dirs/cobits_openvinox_yolox_s_search_8xb8_coco/best_fix_subnet.yaml',
    # You can load the checkpoint of supernet instead of the specific
    # subnet by modifying the `checkpoint`(path) in the following `init_cfg`
    # with `init_weight_from_supernet = True`.
    init_weight_from_supernet=False,
    init_cfg=None)

model = dict(
    _delete_=True,
    type='mmrazor.MMArchitectureQuant',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        pad_size_divisor=32,
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        batch_augments=[
            dict(
                type='BatchSyncRandomResize',
                random_size_range=(480, 800),
                size_divisor=32,
                interval=10)],
    ),
    architecture=qmodel,  # architecture,
    float_checkpoint=None,
    input_shapes =(1, 3, 416, 416),
    quantizer=dict(
        type='mmrazor.OpenVINOXQuantizer',
        quant_bits_skipped_module_names=[
            'backbone.stem.conv.conv',
            'bbox_head.multi_level_conv_cls.2',
            'bbox_head.multi_level_conv_reg.2',
            'bbox_head.multi_level_conv_obj.2'
        ],
        global_qconfig=global_qconfig,
        tracer=dict(
            type='mmrazor.CustomTracer',
            skipped_methods=[
                'mmdet.models.dense_heads.yolox_head.YOLOXHead.predict_by_feat',  # noqa: E501
                'mmdet.models.dense_heads.yolox_head.YOLOXHead.loss_by_feat',
            ])))

optim_wrapper = dict(optimizer=dict(lr=1e-6))

# learning policy
max_epochs = 45
warm_epochs = 1
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
    _delete_=True,
    type='mmrazor.MMArchitectureQuantDDP',
    broadcast_buffers=False,
    find_unused_parameters=True)

# train, val, test setting
train_cfg = dict(
    _delete_=True,
    type='mmrazor.LSQEpochBasedLoop',
    max_epochs=max_epochs,
    val_interval=1,
    is_first_batch=False,
    freeze_bn_begin=-1)
val_cfg = dict(_delete_=True, type='mmrazor.QATValLoop', calibrate_sample_num=200)
test_cfg = val_cfg

# Make sure the buffer such as min_val/max_val in saved checkpoint is the same
# among different rank.
default_hooks = dict(sync=dict(type='SyncBuffersHook'))
