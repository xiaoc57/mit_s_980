import json

import torch
from einops import repeat
from jaxtyping import install_import_hook


with open("D:\\document\\mit_s_980\\homework\\6S980-hw1\\abf149\\metadata.json", 'r') as f:
        metadata = json.load(f)
    
c2w = torch.Tensor(metadata["extrinsics"])
k = torch.Tensor(metadata["intrinsics"])
NUM_STEPS = c2w.shape[0]


k1 = torch.Tensor(
    [
        [-1,0,0,0],[0, -1,0,0],[0,0,-1,0],[0,0,0,1]
    ]
)
k1 = repeat(k1, "i j -> b i j", b=NUM_STEPS)
k2 = torch.Tensor(
    [
        [0,0,1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]
    ]
)
k2 = repeat(k2, "i j -> b i j", b=NUM_STEPS)


c2w = c2w @ k1 @ k2

for i in [+1, -1]:
    for j in range(3):
        
        v = torch.zeros([NUM_STEPS, 4, 1])
        v[:,j,:] = i
        
        cnt = 0
        res = c2w @ v
        
        orig = c2w[:,:,-1:]
        # orig[:,-1,:] = 0
        
        for k in range(NUM_STEPS):
            # if res[k,:,0].dot(torch.Tensor([0,1,0,0])) > 0:
            #     cnt += 1
            print(orig[k,:3, 0] + 2 * res[k,:3,0])
            if torch.allclose(orig[k,:3, 0] + 2 * res[k,:3,0], torch.Tensor([0,0,0])):
                
            # if orig[k,1,0] >= 0:
                cnt += 1
        print(cnt)

