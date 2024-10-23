_base_ = [
    './hawq_weightonly_resnet18_supernet_8xb64_in1k.py'
]

train_cfg = dict(
    _delete_=True,
    type='mmrazor.QNASEvolutionSearchLoop',
    solve_mode='ilp_hawq_eigen',
    dataloader=_base_.val_dataloader,
    evaluator=_base_.val_evaluator,
    max_epochs=1,
    num_candidates=1,
    num_mutation=0,
    num_crossover=0,
    w_act_alphas=[(1.0, 1.5)],  #[(1.0, 1.0), (0.5, 1.0), (1.0, 1.5), (1.0, 2.0), (1.0, 3.0)],
    calibrate_dataloader=_base_.train_dataloader,
    calibrate_sample_num=65536,
    # w4a4: Flops: 34714.419 Params: 48.809
    # w3a4: Flops: 22845.587 Params: 37.652
    # w3a3: 23070
    constraints_range=dict(flops=(0., 33050.)),
    score_key='accuracy/top1')

val_cfg = dict(_delete_=True)
_base_.model.architecture.quantizer.nested_quant_bits_in_layer = True
