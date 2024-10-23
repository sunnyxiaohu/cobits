_base_ = [
    './cobits_superacme_mbv2_supernet_8xb64_in1k.py'
]

train_cfg = dict(
    _delete_=True,
    type='mmrazor.QNASEvolutionSearchLoop',
    solve_mode='evo_org',
    dataloader=_base_.val_dataloader,
    evaluator=_base_.val_evaluator,
    max_epochs=10,
    num_candidates=20,
    # w_act_alphas=[(1.0, 1.0)], # (0.5, 1.0), (1.0, 1.5), (1.0, 2.0), (1.0, 3.0)],
    num_init_candidates=1,
    num_mutation=10,
    num_crossover=10,
    calibrate_dataloader=_base_.train_dataloader,
    calibrate_sample_num=65536,
    # w4a4: Flops: 5394.053 Params: 19.011 fps: 1872
    # w4a8: Flops: 10012.553 Params: 19.011 fps: 1514
    # w8a8: Flops: 19249.553 Params: 27.766 fps: 1106
    # constraints_range=dict(flops=(0, 10012)),
    constraints_range=dict(fps=(1500, 2000)),
    # constraints_range=dict(fps=(1500, 2000), params=(0, 30.0), flops=(0, 7000)),
    estimator_cfg=dict(
        type='mmrazor.HERONResourceEstimator',
        input_shape=(1, 3, 224, 224),
        heronmodel_cfg=dict(
            type='mmrazor.HERONModelWrapper',
            is_quantized=True,
            work_dir='work_dirs/cobits_superacme_mbv2_search_8xb64_in1k',
            mnn_quant_json='projects/commons/heron_files/config_qat.json',
            # Uncomment and adjust `num_infer` for QoR
            num_infer=1000,
            infer_metric=None,)),
    score_key='accuracy/top1')

val_cfg = dict(_delete_=True)
_base_.model.architecture.quantizer.nested_quant_bits_in_layer = True
