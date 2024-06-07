##################### IR2UNet convolutions  ########################
# import defaults packages
import torch
import torch.nn as nn
import torch.nn.functional as F

# basic blocks
class Conv3D_Block(nn.Module):

    def __init__(self, inp_feat, out_feat, kernel=3, stride=1, padding=1):

        super(Conv3D_Block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(inp_feat, out_feat, kernel_size=kernel,stride=stride, padding=padding, bias=True),
            nn.BatchNorm3d(out_feat),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)

class Deconv3D_Block(nn.Module):

    def __init__(self, inp_feat, out_feat, kernel=2, stride=2):

        super(Deconv3D_Block, self).__init__()

        self.deconv = nn.Sequential(
                        nn.ConvTranspose3d(inp_feat, out_feat, kernel_size=2,stride=2),
                        nn.BatchNorm3d(out_feat),
                        nn.ReLU(),
                        )

    def forward(self, x):
        return self.deconv(x)


class ConvSRU2(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 ):
        super(ConvSRU2, self).__init__()

        self.update_gate = Conv3D_Block(input_size, hidden_size)
        self.out_gate = Conv3D_Block(input_size, hidden_size)

    def forward(self, input, h):
        update = torch.sigmoid(self.update_gate(input))
        # print('input_, reset, h, shape ', input_.shape, reset.shape, h.shape)
        # stacked_inputs_ = torch.cat([input_, h * reset], dim=1)

        out_inputs = self.out_gate(input)
        h_new = h * (1 - update) + out_inputs * update
        return h_new

    def __repr__(self):
        return 'ConvDRU: \n' + \
               '\t reset_gate: \n {}\n'.format(self.reset_gate.__repr__()) + \
               '\t update_gate: \n {}\n'.format(self.update_gate.__repr__()) + \
               '\t out_gate:  \n {}\n'.format(self.out_gate.__repr__())

def AddNoise(ins, SNR):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ins = ins.to("cpu")
    sigPower = SigPower(ins)
    noisePower = sigPower / SNR
    noise = torch.sqrt(noisePower) * torch.randn(ins.size())
    noise = noise.to("cuda:0")
    ins = ins.to("cuda:0")
    return ins + noise

def SigPower(ins):
    ll = torch.numel(ins)
    tmp1 = torch.sum(ins ** 2)
    return torch.div(tmp1, ll)