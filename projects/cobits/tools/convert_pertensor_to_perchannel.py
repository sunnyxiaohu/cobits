import os
import sys
import torch
import copy

if __name__ == '__main__':
    org_pth = sys.argv[1]
    new_pth = org_pth.replace('.pth', '_new.pth')
    pth1 = torch.load(org_pth)
    pth2 = copy.deepcopy(pth1)
    for k, v in pth1['state_dict'].items():
        if 'weight_fake_quant' in k:
            lk = k.split('.')[-1]
            if lk in ['min_val', 'max_val', 'scale', 'zero_point']:
                ref_k = k.split('weight_fake_quant')[0] + 'weight'
                channels = len(pth1['state_dict'][ref_k])
                pth2['state_dict'][k] = v.repeat(channels)
    # import pdb; pdb.set_trace()                
    print(f'Save new_pth to: {new_pth}')
    torch.save(pth2, new_pth)
