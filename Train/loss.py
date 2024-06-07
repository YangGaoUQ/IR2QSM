import torch
import torch.nn as nn
import torch.nn.functional as F

def handle_input_target_mismatch(input, target):
    # confirm input and target have same size
    n, c, d, h, w = input.size()
    nt, nc, dt, ht, wt = target.size()

    # 处理input和target尺寸不一致的情况
    if d > dt and h > ht and w > wt:  # upsample target
        target = target.unsqueeze(1)
        target = F.interpolate(target, size=(d, h, w), mode="nearest")
        target = target.squeeze(1)
    elif d < dt and h < ht and w < wt:  # upsample input
        input = F.interpolate(input, size=(dt, ht, wt), mode="trilinear", align_corners=True)
    elif (d != dt or h != ht or w != wt):  
        raise Exception("Only support upsampling")

    return input, target

def MseLoss(input, target, reduction='sum', bkargs=None):
    input, target = handle_input_target_mismatch(input, target) 
    mse_loss = nn.MSELoss(reduction='sum')
    loss = mse_loss(input,target)
    return loss

class Multi_Step_Loss(nn.Module):
    def __init__(self, scale_weight=0.5, n_inp=None, weight=None, reduction='sum'):
        super(Multi_Step_Loss, self).__init__()
        # [1.0, 0.5, 0.25]
        scale = 1.0 if scale_weight is None else scale_weight
        print('my_multi_step_loss scale weight is {}'.format(scale))
        self.scale_weight = torch.pow(scale * torch.ones(n_inp), torch.arange(n_inp-1, -1, -1, out=torch.FloatTensor()))
        print(scale_weight)
        self.weight = weight
        self.reduction = reduction

    def forward(self, myinput, target):
        loss = 0
        for i, inp in enumerate(myinput):
            loss = loss + self.scale_weight[i] * MseLoss(
                input=inp, target=target, reduction=self.reduction
            )
        return loss

    
