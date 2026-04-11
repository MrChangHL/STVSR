''' network architecture for Sakuya '''
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.module_util as mutil
from models.convlstm import ConvLSTM, ConvLSTMCell
from models.base_networks import *
from models.Spatial_Temporal_Transformer import MSTT as MSST_former
from models.mambaIR import ResidualGroup, PatchEmbed, PatchUnEmbed
# from models.fremamba import ResidualGroup, PatchEmbed, PatchUnEmbed
# from models.mambaIR_VFI import ResidualGroup, PatchEmbed, PatchUnEmbed
from basicsr.archs.spynet_arch import SpyNet
from basicsr.archs.arch_util import ResidualBlockNoBN, flow_warp, make_layer
from basicsr.archs.channel_diversity import MultiSpectralAttentionLayer

try:
    from models.DCNv2.dcn_v2 import DCN_sep
except ImportError:
    raise ImportError('Failed to import DCNv2 module.')

class TDM_L(nn.Module):

    def __init__(self, nframes, apha=0.5, belta=0.5):
        super(TDM_L, self).__init__()

        self.nframes = nframes
        self.apha = apha
        self.belta = belta
        base_filter = 64

        self.compress_3_1 = ConvBlock(self.nframes*64, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.compress_3_2 = ConvBlock(self.nframes*64, base_filter, 3, 1, 1, activation='prelu', norm=None)

        self.conv1_1 = ConvBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.conv1_2 = ConvBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.conv2_1 = ConvBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.conv2_2 = ConvBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.conv3_1 = ConvBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.conv3_2 = ConvBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)

        # self.conv4 = ConvBlock(base_filter, self.nframes*64, 3, 1, 1, activation='prelu', norm=None)
        self.conv4_1 = ConvBlock(base_filter, self.nframes*64, 3, 1, 1, activation='prelu', norm=None)
        self.conv4_2 = ConvBlock(base_filter, self.nframes*64, 3, 1, 1, activation='prelu', norm=None)
        self.conv5_1 = ConvBlock(base_filter*3, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.conv5_2 = ConvBlock(base_filter*3, base_filter, 3, 1, 1, activation='prelu', norm=None)


        self.avg_diff = nn.AvgPool2d(kernel_size=2, stride=2)
        self.sigmoid = nn.Sigmoid()

        self.fus = nn.Conv2d(base_filter*2, base_filter, 3, 1, 1)

        self.CAB = ChannelAttention(64)

    def forward(self, frame_fea_list):
        frame_fea = torch.cat(frame_fea_list, 1)  # [b nframe*64 h w]

        frame_list_reverse = frame_fea_list.copy()
        frame_list_reverse.reverse()  # [[B,64,h,w], ..., ]
        # multi-scale: 3*3
        # forward
        forward_fea3 = self.conv1_1(self.compress_3_1(torch.cat(frame_fea_list, 1)))
        # backward
        backward_fea3 = self.conv1_2(self.compress_3_2(torch.cat(frame_list_reverse, 1)))

        forward_diff_fea3 = forward_fea3 - backward_fea3

        backward_diff_fea3 = backward_fea3 - forward_fea3


        id_f3 = forward_diff_fea3  # [b 96 h w]
        id_b3 = backward_diff_fea3
        pool_f3 = self.conv3_1(self.avg_diff(forward_fea3))  # [b 96 h/2, w/2]
        up_f3 = F.interpolate(pool_f3, scale_factor=2, mode='bilinear', align_corners=True)

        pool_b3 = self.conv3_2(self.avg_diff(backward_fea3))
        up_b3 = F.interpolate(pool_b3, scale_factor=2, mode='bilinear', align_corners=True)

        enhance_f3 = self.conv2_1(forward_fea3)
        enhance_b3 = self.conv2_2(backward_fea3)

        # f3 = self.sigmoid(self.conv4(id_f3 + enhance_f3 + up_f3))
        # b3 = self.sigmoid(self.conv4(id_b3 + enhance_b3 + up_b3))
        f3 = self.sigmoid(self.conv4_1(self.CAB(self.conv5_1(torch.cat((id_f3, enhance_f3, up_f3), dim=1)))))
        b3 = self.sigmoid(self.conv4_2(self.CAB(self.conv5_2(torch.cat((id_b3, enhance_b3, up_b3), dim=1)))))
        # f3 = self.sigmoid(self.conv4(self.CAB(self.conv5(id_f3 + enhance_f3 + up_f3))))
        # b3 = self.sigmoid(self.conv4(self.CAB(self.conv5(id_b3 + enhance_b3 + up_b3))))

        att3 = f3 + b3
        module_fea3 = att3 * frame_fea + frame_fea
        return module_fea3

class MSTTr(nn.Module):
    def __init__(self, nf=64, nframes=3, groups=8, front_RBs=5, back_RBs=10):
        super(MSTTr, self).__init__()
        # self.nf = nf
        # self.in_frames = 1 + nframes // 2
        # self.ot_frames = nframes
        p_size = 48 # a place holder, not so useful
        patch_size = (p_size, p_size)
        n_layers = 1
        hidden_dim = []
        for i in range(n_layers):
            hidden_dim.append(nf)

        ResidualBlock_noBN_f = functools.partial(mutil.ResidualBlock_noBN, nf=nf)
        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.feature_extraction = mutil.make_layer(ResidualBlock_noBN_f, front_RBs)

        #### reconstruction
        self.recon_trunk = mutil.make_layer(ResidualBlock_noBN_f, back_RBs)

        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)
        # self.conv_last2 = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        #### Mamba
        self.fusion3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)  # blending
        self.fusion4 = nn.Conv2d(nf * 3, nf, 1, 1, bias=True)  # blending      gai_kernel
        self.fusion5 = nn.Conv2d(nf * 2, nf, 1, 1, bias=True)  # blending
        self.patch_norm = True
        self.pos_drop = nn.Dropout(p=0.)
        self.norm = nn.LayerNorm(nf)  ## embed_dim=64
        self.patch_embed = PatchEmbed(
            img_size=64,
            patch_size=1,
            in_chans=64,
            embed_dim=64,
            norm_layer=nn.LayerNorm if self.patch_norm else None)
        self.patch_unembed = PatchUnEmbed(
            img_size=64,
            patch_size=1,
            in_chans=64,
            embed_dim=64,
            norm_layer=nn.LayerNorm if self.patch_norm else None)
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.layers = nn.ModuleList()
        for i_layer in range(1):
            layer = ResidualGroup(
                dim=64,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=6,
                norm_layer=True,
                downsample=None,
                use_checkpoint=False,
                img_size=64,
                patch_size=1,
                resi_connection='1conv')
            self.layers.append(layer)

        # LGTD
        self.tdm_l = TDM_L(nframes = 7)

    # def get_flow(self, x):
    #     b, n, c, h, w = x.size()
    #
    #     x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
    #     x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)
    #
    #     flows_backward = self.spynet(x_1, x_2).view(b, n - 1, 2, h, w)
    #     flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)
    #
    #     return flows_forward, flows_backward

    def forward(self, x, f_flow, b_flow):
        B, N, C, H, W = x.size()  # N input video frames

        #### extract LR features
        # L1
        L1_fea = self.lrelu(self.conv_first(x.view(-1, C, H, W)))
        L1_fea = self.feature_extraction(L1_fea)  # [4,64,h,w]
        L1_fea = L1_fea.view(B, N, -1, H, W)  # 1 3 5 7 frame
        flow_b_fea_0 = mutil.flow_warp(L1_fea[:, 1, :, :, :], f_flow[0])  # F2-，F4-，F6-
        flow_f_fea_0 = mutil.flow_warp(L1_fea[:, 0, :, :, :], b_flow[0])  # F2+，F4+，F6+
        fusion1 = self.fusion5(torch.cat([L1_fea[:,0,:,:,:], L1_fea[:,1,:,:,:]], dim=1))
        input_1 = self.pos_drop(self.patch_embed(fusion1))
        for layer1 in self.layers:
            x1 = layer1(input_1, (fusion1.shape[2], fusion1.shape[3]))
        x1_1 = self.norm(x1)
        x1_2 = self.fusion3(self.patch_unembed(x1_1, (fusion1.shape[2], fusion1.shape[3]))) + fusion1

        x1_3 = self.fusion4(torch.cat((flow_b_fea_0,x1_2,flow_f_fea_0), dim=1))

        ### frame 1..2
        flow_b_fea_1 = mutil.flow_warp(L1_fea[:, 2, :, :, :], f_flow[1])  # F2-，F4-，F6-
        flow_f_fea_1 = mutil.flow_warp(L1_fea[:, 1, :, :, :], b_flow[1])  # F2+，F4+，F6+

        fusion2 = self.fusion5(torch.cat([L1_fea[:,1,:,:,:], L1_fea[:,2,:,:,:]], dim=1))
        input_2 = self.pos_drop(self.patch_embed(fusion2))
        for layer2 in self.layers:
            x2 = layer2(input_2, (fusion2.shape[2], fusion2.shape[3]))
        x2_1 = self.norm(x2)
        x2_2 = self.fusion3(self.patch_unembed(x2_1, (fusion2.shape[2], fusion2.shape[3]))) + fusion2

        x2_3 = self.fusion4(torch.cat((flow_b_fea_1, x2_2, flow_f_fea_1), dim=1))

        ### frame 2..3
        flow_b_fea_2 = mutil.flow_warp(L1_fea[:, 3, :, :, :], f_flow[2])  # F2-，F4-，F6-
        flow_f_fea_2 = mutil.flow_warp(L1_fea[:, 2, :, :, :], b_flow[2])  # F2+，F4+，F6+

        fusion3 = self.fusion5(torch.cat([L1_fea[:,2,:,:,:], L1_fea[:,3,:,:,:]], dim=1))
        input_3 = self.pos_drop(self.patch_embed(fusion3))
        for layer3 in self.layers:
            x3 = layer3(input_3, (fusion3.shape[2], fusion3.shape[3]))
        x3_1 = self.norm(x3)
        x3_2 = self.fusion3(self.patch_unembed(x3_1, (fusion3.shape[2], fusion3.shape[3]))) + fusion3

        x3_3 = self.fusion4(torch.cat((flow_b_fea_2, x3_2, flow_f_fea_2), dim=1))

        to_mstt_fea = torch.stack([L1_fea[:,0,:,:,:], x1_3, L1_fea[:,1,:,:,:], x2_3, L1_fea[:,2,:,:,:], x3_3, L1_fea[:,3,:,:,:]], dim=1)
        to_mstt_fea_list = [to_mstt_fea[:, i, :, :, :] for i in range(to_mstt_fea.shape[1])]
        feats = self.tdm_l(to_mstt_fea_list).view(B, N+3, 64, H, W)
        _, T, C, _, _ = feats.size()
        feats = feats.view(B * T, C, H, W)
        # feats = x1_2.view(B * T, C, H, W)
        # feats = x_out.view(B * T, C, H, W)
        out = self.recon_trunk(feats)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.lrelu(self.HRconv(out))
        out = self.conv_last(out)
        _, _, K, G = out.size()
        outs = out.view(B, T, -1, K, G)
        return outs
