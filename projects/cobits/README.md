# Env setup
- `mmengine` == 0.8.4
- `mmcv` == 2.0.0rc4
- `mmcls` == 1.0.0rc6
- `mmdet` == 3.2.0

# About the source code
1. This code is modified and based on the project: [mmrazor](https://github.com/open-mmlab/mmrazor)
2. `projects/cobits/configs` shows an example configuration files for Cobits.
3. The main logic of the proposed Cobits is located in:
    - `mmrazor/engine/runner/qnas_loops.py`
    - `mmrazor/models/algorithms/nas/qnas.py`
    - `mmrazor/models/quantizers`
    - `mmrazor/models/observers`

# Commands
### step 1. Train mixed-precision supernet with co-learning scale factor
```bash
bash tools/dist_train.sh projects/cobits/configs/mobilenetv2/cobits_snpe_mbv2_supernet_8xb64_in1k.py 8
```

### step2. Search Bit-width configurations in One-time
```bash
bash tools/dist_train.sh projects/cobits/configs/mobilenetv2/cobits_snpe_mbv2_search_8xb64_in1k.py 8 --cfg-options load_from=$STEP1_CKPT
```

### step3: Finetune the searched subnet
```bash
bash tools/dist_train.sh projects/cobits/configs/mobilenetv2/cobits_snpe_mbv2_subnet_8xb64_in1k.py 8 --cfg-options randomness.seed=777 model.architecture.fix_subnet=$STEP2/best_fix_subnet.yaml model.fix_subnet=$STEP2/best_fix_subnet_by_module_name.yaml load_from=$STEP2/subnet_ckpt.pth
```
### step4: Evaluate the searched subnet
```bash
bash tools/dist_test.sh projects/cobits/configs/mobilenetv2/cobits_snpe_mbv2_subnet_8xb64_in1k.py $STEP2/subnet_ckpt.pth 8 --cfg-options randomness.seed=777 model.architecture.fix_subnet=$STEP2/best_fix_subnet.yaml model.fix_subnet=$STEP2/best_fix_subnet_by_module_name.yaml
```