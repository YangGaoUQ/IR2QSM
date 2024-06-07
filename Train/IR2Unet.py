from torch.nn import Module, Sequential, Conv3d, BatchNorm3d, ConvTranspose3d, ReLU, MaxPool3d, Sigmoid, Parameter
from torch import tensor, cat
import torch
import torch.nn as nn
from IR2UnetBlock import *
import numpy as np

class IR2Unet(Module):

    def __init__(self,
                 iterations: int = 4,  #iteration number
                 multiplier: float = 1.0,  #scaling factor
                 depth: int = 4,  #the depth or network
                 integrate: bool = True):  # default iteration

        super(IR2Unet, self).__init__()
        #  参数
        self.iterations = iterations
        self.multiplier = multiplier
        self.depth = depth
        self.integrate = integrate
        self.Prob = 0.3
        self.SNRs = torch.tensor([20, 10, 5, 2])
        #  channel number【16 32 64 128】
        self.filters_list = [int(16 * (2 ** i) * self.multiplier) for i in range(self.depth)]

        self.pre_transform = []
        self.encoders = []  
        self.middles = []
        self.middle_dropout = []
        self.decoders = []  
        self.post_transform = []

        for iteration in range(self.iterations):  
            
            pre_transform_conv_block = Sequential(
                Conv3D_Block(1, self.filters_list[0]),
                nn.Dropout(0.05),
                Conv3D_Block(self.filters_list[0], self.filters_list[0])
            )
            self.add_module("iteration{0}_pre_transform".format(iteration), pre_transform_conv_block)
            self.pre_transform.append(pre_transform_conv_block)

            #  encoders
            for layer in range(self.depth - 1): 
                in_channel = self.filters_list[layer] * 2  
                mid_channel = self.filters_list[layer]
                out_channel = self.filters_list[layer] * 2
                Conv1 = Conv3D_Block(in_channel, mid_channel)
                Conv2 = Sequential(
                    nn.Dropout(0.05),
                    Conv3D_Block(mid_channel, mid_channel)
                ) 
                down = Sequential(
                    Conv3D_Block(mid_channel, out_channel),
                    MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
                )
                self.add_module("iteration{0}_layer{1}_encoder_conv1".format(iteration, layer), Conv1)
                self.add_module("iteration{0}_layer{1}_encoder_conv2".format(iteration, layer), Conv2)
                self.add_module("iteration{0}_layer{1}_encoder_down".format(iteration, layer), down)
                self.encoders.append((Conv1, Conv2, down))

            #  middle_SRU
            middle_drop = nn.Dropout(0.05)
            Conv_middle = ConvSRU2(self.filters_list[-1], self.filters_list[-1])

            self.add_module("iteration{0}_middle_drop".format(iteration), middle_drop)
            self.add_module("iteration{0}_middle".format(iteration), Conv_middle)
            
            self.middle_dropout.append(middle_drop)
            self.middles.append(Conv_middle)

            #  decoders
            for layer in range(self.depth - 1):  
                in_channel = self.filters_list[self.depth - 1 - layer]
                out_channel = self.filters_list[self.depth - 1 - layer] // 2
                up = Deconv3D_Block(in_channel, out_channel)
                convs = Sequential(
                    Conv3D_Block(in_channel, in_channel),
                    nn.Dropout(0.05),
                    Conv3D_Block(in_channel, out_channel)
                )

                self.add_module("iteration{0}_layer{1}_decoder_up".format(iteration, layer), up)
                self.add_module("iteration{0}_layer{1}_decoder_convs".format(iteration, layer), convs)
                self.decoders.append((up, convs))

            #  Output
            single_output_conv_block = Sequential(
                Conv3D_Block(self.filters_list[0], self.filters_list[0]),
                Conv3d(self.filters_list[0], 1, kernel_size=1, stride=1, padding=0),
            )
            self.add_module("iteration{0}_post_transform".format(iteration), single_output_conv_block)
            self.post_transform.append(single_output_conv_block)

        #concatenation
        self.post_transform_conv_block2 = Sequential(
            Conv3d(self.iterations, 1, kernel_size=1, stride=1, padding=0)
        )

        self.post_transform_conv_block1 = Sequential(
            nn.Conv3d(self.filters_list[0] * self.iterations, self.filters_list[0], kernel_size=(3, 3, 3),
                      padding=(1, 1, 1),
                      stride=(1, 1, 1)) if self.integrate else nn.Conv3d(self.filters_list[0],
                                                                         self.filters_list[0], kernel_size=(3, 3, 3),
                                                                         padding=(1, 1, 1), stride=(1, 1, 1)),
            ReLU(),
            BatchNorm3d(self.filters_list[0]),

            Conv3D_Block(self.filters_list[0], self.filters_list[0]),

            Conv3d(self.filters_list[0], 1, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        enc = [None for i in range(self.depth)]
        dec = [None for i in range(self.depth)]
        all_output = [None for i in range(self.iterations)]
        e_i = 0
        d_i = 0
        for iteration in range(self.iterations):
            # --------pre-------
            if iteration == 0:
                x_in = self.pre_transform[iteration](x)
            else:
                x_in = self.pre_transform[iteration](x_in)

            ### add noise; 
            if self.training:
                tmp = torch.rand(1)
                if tmp > self.Prob:
                    #print('noise')
                    tmp_mask = x_in != 0
                    tmp_idx = torch.randint(4, (1,1))
                    tmp_SNR = self.SNRs[tmp_idx]
                    x_in = AddNoise(x_in, tmp_SNR)

            # --------encoder-------
            if iteration == 0:  
                for layer in range(self.depth - 1):
                    x_in = self.encoders[e_i][1](x_in)
                    enc[layer] = x_in
                    x_in = self.encoders[e_i][2](x_in)
                    e_i = e_i + 1
            else:
                for layer in range(self.depth - 1):
                    x_in1 = x_in
                    x_in2 = dec[self.depth - layer - 2]
                    x_in = self.encoders[e_i][0](cat([x_in1, x_in2], dim=1))
                    x_in = self.encoders[e_i][1](x_in)
                    enc[layer] = x_in
                    x_in = self.encoders[e_i][2](x_in)
                    e_i = e_i + 1
            
            # add noise
            if self.training:
                tmp = torch.rand(1)
                if tmp > self.Prob:
                    tmp_mask = x_in != 0
                    tmp_idx = torch.randint(4, (1,1))
                    tmp_SNR = self.SNRs[tmp_idx]
                    x_in = AddNoise(x_in, tmp_SNR)
            
            
            # --------middle-------
            x_in = self.middle_dropout[iteration](x_in)
            if iteration == 0:
                h = x_in
            h = self.middles[iteration](x_in, h)
            x_in = h
            # add noise
            if self.training:
                tmp = torch.rand(1)
                if tmp > self.Prob:
                    tmp_mask = x_in != 0
                    tmp_idx = torch.randint(4, (1,1))
                    tmp_SNR = self.SNRs[tmp_idx]
                    x_in = AddNoise(x_in, tmp_SNR)

            # --------decoder-------
            for layer in range(self.depth - 1):
                x_in = self.decoders[d_i][0](x_in)
                x_in1 = x_in
                x_in2 = enc[self.depth - layer - 2]
                x_in = self.decoders[d_i][1](cat([x_in1, x_in2], dim=1))
                dec[layer] = x_in
                d_i = d_i + 1

            # add noise
            tmp = torch.rand(1)
            if tmp > self.Prob:
                tmp_mask = x_in != 0
                tmp_idx = torch.randint(4, (1,1))
                tmp_SNR = self.SNRs[tmp_idx]
                x_in = AddNoise(x_in, tmp_SNR)

            x_in = self.post_transform[iteration](x_in)
            x_in = x_in + x
            all_output[iteration] = x_in

        if self.integrate:
            x_in = cat(all_output, dim=1) 
        x_out = self.post_transform_conv_block2(x_in)
        x_out = x_out + x
        return x_out, all_output

def weights_init(m):
    if isinstance(m, nn.Conv3d):
        nn.init.normal_(m.weight, mean=0.0, std=1e-2)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, nn.ConvTranspose3d):
        nn.init.normal_(m.weight, mean=0, std=1e-2)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, nn.BatchNorm3d):
        nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


#################### For Code Test ##################################
if __name__ == '__main__':
    Net = IR2Unet()
    Net.apply(weights_init)
    print(get_parameter_number(Net))
    x = torch.randn(1, 1, 64, 64, 64, dtype=torch.float)
    Net = Net.to("cuda:0")
    x = x.to("cuda:0")
    print('input ' + str(x.size()))
    print(x.dtype)
    y, all_out = Net(x)
    print('output ' + str(y.size()))