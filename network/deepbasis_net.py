
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from network.base_net import LayerNorm2d


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
    
class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class RDANBasis_2018(nn.Module):
    def __init__(self, en_channels, de_channels,basis_channels,basis_num,out_channel=7):
        super().__init__()
        de_channels[-1] = basis_num
        basis_channels[-1] = int(basis_num*10)
        self.out_channel = out_channel
        #  encoder layers
        self.en_convs = []
        self.en_fcs = []
        self.en_g2e_fcs = []
        self.en_ins_norms=[]
        self.basis_num = basis_num
        in_channels = en_channels[0:-1]
        out_channels = en_channels[1:]
        fc_out_channels = en_channels[2:]
        fc_out_channels.append(out_channels[-1])
        for i, (inC, outC, fc_outC) in enumerate(zip(in_channels, out_channels, fc_out_channels)):
            if i == 0:
                self.en_convs.append(nn.Conv2d(inC, outC, 4, 2, padding=[1,1]))
                self.en_fcs.append(nn.Sequential(
                    nn.Linear(inC, fc_outC),
                    nn.SELU(inplace=True)
                ))
            elif i == len(in_channels)-1:
                self.en_convs.append(nn.Sequential(
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(inC, outC, 4, 2, padding=[1,1])
                ))
                self.en_fcs.append(nn.Sequential(
                    nn.Linear(outC+last_outC, fc_outC),
                    nn.SELU(inplace=True)
                ))
                self.en_g2e_fcs.append(nn.Linear(outC, outC, False))
            else:
                self.en_convs.append(nn.Sequential(
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(inC, outC, 4, 2, padding=[1,1])
                ))
                self.en_ins_norms.append(nn.InstanceNorm2d(outC, affine=True))
                self.en_fcs.append(nn.Sequential(
                    nn.Linear(outC+last_outC, fc_outC),
                    nn.SELU(inplace=True)
                ))
                self.en_g2e_fcs.append(nn.Linear(outC, outC, False))
            
            last_outC = fc_outC
        self.en_convs = nn.Sequential(*self.en_convs)
        self.en_fcs = nn.Sequential(*self.en_fcs)
        self.en_g2e_fcs = nn.Sequential(*self.en_g2e_fcs)
        self.en_ins_norms = nn.Sequential(*self.en_ins_norms)

        # decoder layers
        in_channels = de_channels[0:-1]
        out_channels = de_channels[1:]
        fc_out_channels = de_channels[1:]
        self.de_convs = []
        self.de_fcs = []
        self.de_g2e_fcs = []
        self.de_ins_norms=[]
        for i, (inC, outC, fc_outC) in enumerate(zip(in_channels, out_channels, fc_out_channels)):
            if i != 0:
                inC = inC *2
            if i != len(in_channels)-1:
                self.de_ins_norms.append(nn.InstanceNorm2d(outC, affine=True))
                self.de_fcs.append(nn.Sequential(
                    nn.Linear(outC+last_outC, fc_outC),
                    nn.SELU(inplace=True)
                ))
            self.de_convs.append(nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(inC, outC, 3, 1, padding=1),
                nn.Conv2d(outC, outC, 3, 1, padding=1)
            ))
            self.de_g2e_fcs.append(nn.Linear(last_outC, outC, False))
            last_outC = fc_outC
        # weight net
        self.weight_net = nn.ModuleDict()
        self.weight_net.add_module('deconv',nn.Sequential(*self.de_convs))
        self.weight_net.add_module('fcs',nn.Sequential(*self.de_fcs))
        self.weight_net.add_module('f2c',nn.Sequential(*self.de_g2e_fcs))
        self.weight_net.add_module('ins',nn.Sequential(*self.de_ins_norms))

        self.drop = nn.Dropout(inplace=True)
        self.norm = 'torch_normalization'
        self.clip = nn.Tanh()
        # basis network
        self.basis_net = nn.ModuleList()
        for i in range(len(basis_channels)-2):
            self.basis_net.append(nn.Sequential(
                nn.Linear(basis_channels[i],basis_channels[i+1]),
                nn.LeakyReLU(0.2, True),
            ))
        self.basis_net.append(nn.Sequential(
            nn.Linear(basis_channels[-2],basis_channels[-1]),
            nn.Tanh()
        ))
        self.init_modules()

    def init_modules(self):

        module_list = [self.en_convs, self.en_fcs, self.en_g2e_fcs, self.en_ins_norms, self.weight_net]

        for module in module_list:
            
            multiply = 0.01

            for m in module.modules():
                if m == module:
                    continue
                if isinstance(m, nn.Sequential):
                    for msub in m.modules():
                        if msub == m:
                            continue
                        self.init_weights(msub)
                else:
                    self.init_weights(m, multiply) # 改动
    def init_weights(self, m, multiply = 1.0): # 改动
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0, 0.02)
        if isinstance(m, nn.InstanceNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, np.sqrt(1.0/m.weight.data.shape[1])*multiply) # g2e_fc的全连接层方差要乘一个0.01
            if m.bias is not None:
                nn.init.normal_(m.bias, 0,0.002)
            
    def forward(self, x):
        layers = []
        mean = x.mean(dim=(2,3))
        for i, (conv, fc) in enumerate(zip(self.en_convs, self.en_fcs)):
            output = conv(x)
            if i != 0:
                mean = output.mean(dim=(2,3))
                mean = torch.cat([globalOutput,mean], dim=-1)
                if i != len(self.en_convs)-1:
                    output = self.en_ins_norms[i-1](output)
                b, c = globalOutput.shape
                output = output + self.en_g2e_fcs[i-1](globalOutput).view(b,c, 1, 1)
            globalOutput = fc(mean)
            x = output
            layers.append(x)
        # weight net
        for i, (deconv, g2e_fc) in enumerate(zip(self.weight_net['deconv'], self.weight_net['f2c'])):
            if i != 0:
                x = torch.cat([x,layers[-(i+1)]], dim=1)
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            output = deconv(x)
            if i != len(self.weight_net['deconv'])-1:
                mean = output.mean(dim=(2,3))
                mean = torch.cat([globalOutput,mean], dim=-1)
                output = self.weight_net['ins'][i](output)
            b, c = output.shape[:2]
            output = output + g2e_fc(globalOutput).view(b,c, 1, 1)
            # if i < 3:
            #     output = self.drop(output)
            if i != len(self.weight_net['deconv'])-1:
                globalOutput = self.weight_net['fcs'][i](mean)
            x = output
        weight = self.clip(x)
        #basis
        feat = layers[-1].view(layers[-1].shape[0],-1)
        for b_net in self.basis_net:
            feat = b_net(feat)
        basis = feat.view(feat.shape[0],self.basis_num,self.out_channel)
        return weight,basis


class GlobalNet(nn.Module):
    def __init__(self):
        super().__init__()
        en_channels = [9, 64, 128, 256, 512, 512, 512, 512, 512]
        de_channels = [512, 512, 512, 512, 512, 256, 128, 64, 10]
        basis_channels = [512,512,512,512,100]
        basis_num = 10
        self.outChannel  = 10
        self.basisNet = RDANBasis_2018(en_channels,de_channels,basis_channels,basis_num,self.outChannel)
    def forward(self,x):
        weight,basis = self.basisNet(x)
        if self.outChannel == 10:
            n,d,r,s = torch.split(basis,[3,3,1,3],2)
            new_n = torch_norm(n,2)
            new_basis = torch.concat([new_n,d,r,s],2)
        elif self.outChannel == 7:
            new_basis = basis
        return weight,new_basis


class LocalNet(nn.Module):

    def __init__(self, basis_num=10, width=16, middle_blk_num=12, enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2]):
        super().__init__()
        self.basis_num = basis_num
        img_channel=int(12+basis_num*11)
        out_channel=int(basis_num*10)
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=out_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.basisNet = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.clip = nn.Tanh()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp,basis,weights):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        basis_map = torch.tile(basis.unsqueeze(-1).unsqueeze(-1),(1,1,1,256,256))
        inp = torch.cat([inp,weights,basis_map.view(basis_map.shape[0],-1,256,256)],1)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        multi_basis_details =  self.clip(x[:, :, :H, :W])
        multi_basis_details = multi_basis_details.view(B,self.basis_num,10,H,W)
        return multi_basis_details

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
def torch_norm(arr, dim=1):
    length = torch.sqrt(torch.sum(arr * arr, dim = dim, keepdims=True))
    return arr / (length + 1e-12)