import sys
from tkinter.tix import Tree
sys.path.append('core')
sys.path.append('/home/lenovo/omnimotion/STIR-MFT/MFT-master/MFT_Endo_TTAP')

import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from thirdparty.RAFT.update import BasicUpdateBlock, OcclusionAndUncertaintyBlock
from thirdparty.RAFT.corr import CorrBlock
from utils.utils import coords_grid, InputPadder, upsample8
from thirdparty.RAFT.extractor import ResNetFPN, Feature_encoder
from thirdparty.RAFT.layer import conv1x1, conv3x3

from huggingface_hub import PyTorchModelHubMixin




class RAFT(
    nn.Module,
    PyTorchModelHubMixin, 
    # optionally, you can add metadata which gets pushed to the model card
    # repo_url="https://github.com/princeton-vl/SEA-RAFT",
    # pipeline_tag="optical-flow-estimation",
    # license="BSD-3-Clause",
):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.output_dim = args.dim * 2
        self.a = 0
        self.args.corr_levels = 4
        self.args.corr_radius = args.radius
        self.args.corr_channel = args.corr_levels * (args.radius * 2 + 1) ** 2
        self.cnet = ResNetFPN(args, input_dim=6, output_dim=2 * self.args.dim, norm_layer=nn.BatchNorm2d, init_weight=True)
        self.num = 0

        # conv for iter 0 results
        self.init_conv = conv3x3(2 * args.dim, 2 * args.dim)
        self.upsample_weight = nn.Sequential(
            # convex combination of 3x3 patches
            nn.Conv2d(args.dim, args.dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(args.dim * 2, 64 * 9, 1, padding=0)
        )
        self.flow_head = nn.Sequential(
            # flow(2) + weight(2) + log_b(2)
            nn.Conv2d(args.dim, 2 * args.dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * args.dim, 6, 3, padding=1)
        )
        
        self.occlusion_estimation = args.occlusion_module is not None
        self.uncertainty_estimation = self.occlusion_estimation and 'with_uncertainty' in args.occlusion_module
        self.OU_last_iter_only = getattr(args, 'OU_last_iter_only', False)
        self.relu_uncertainty = getattr(args, 'relu_uncertainty', False)


        if args.occlusion_module is not None: # run this
            self.size_occl_uncer_input_dims = 512 # 712 origin
            self.occlusion_block = OcclusionAndUncertaintyBlock(self.args, hidden_dim=self.size_occl_uncer_input_dims)

        if self.uncertainty_estimation:
            self.mult_uncetrainty_upsample = 8.0 if 'upsample8' in args.occlusion_module else 1.0
            self.eps_uncertainty = 10e-4

        if args.iters > 0:
            self.fnet = ResNetFPN(args, input_dim=3, output_dim=self.output_dim, norm_layer=nn.BatchNorm2d, init_weight=True)
            self.update_block = BasicUpdateBlock(args, hdim=args.dim, cdim=args.dim)

        self.residual_branch = nn.Sequential(
                nn.Conv2d(args.dim, 2 * args.dim, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(2 * args.dim, 2 * args.dim, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(2 * args.dim, 6, 3, padding=1)
                )

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def update_weight(self, current_iter, max_iter, a_max=1.0):

        if current_iter == 0:
            self.a = 0.3  # 初始值
        elif current_iter <= max_iter:
            self.a = (0.7 * (current_iter / max_iter) ** 2) + 0.3 # 非线性增长
        else:
            self.a = a_max
        print("a:", self.a, "current_iter:", self.num)

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords2 - coords1"""
        N, C, H, W = img.shape
        coords1 = coords_grid(N, H//8, W//8, device=img.device)
        coords2 = coords_grid(N, H//8, W//8, device=img.device)
        return coords1, coords2

    def upsample_data(self, flow, info, mask):
        """ Upsample [H/8, W/8, C] -> [H, W, C] using convex combination """
        N, C, H, W = info.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        up_info = F.unfold(info, [3, 3], padding=1)
        up_info = up_info.view(N, C, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        up_info = torch.sum(mask * up_info, dim=2)
        up_info = up_info.permute(0, 1, 4, 2, 5, 3)
        
        return up_flow.reshape(N, 2, 8*H, 8*W), up_info.reshape(N, C, 8*H, 8*W)

    def upsample_occ(self, flow, mask):
        """ Upsample [H/8, W/8, C] -> [H, W, C] using convex combination """
        N, C, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, C, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)

        return up_flow.reshape(N, 2, 8*H, 8*W)

    def upsample_unc(self, flow, mask):
        """ Upsample [H/8, W/8, C] -> [H, W, C] using convex combination """
        N, C, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, C, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)

        return up_flow.reshape(N, C, 8*H, 8*W)

    def upsample_flow(self, flow, mask, mult_coef=8.0, n_channels=2):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(mult_coef * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, n_channels, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, n_channels, 8*H, 8*W)

    def upsample8(maps_data, mode='bilinear'):
        new_size = (8 * maps_data.shape[2], 8 * maps_data.shape[3])
        return F.interpolate(maps_data, size=new_size, mode=mode, align_corners=True)


    def forward(self, image1, image2, iters=None, flow_gt=None, test_mode=False, dino1_feature=None, dino2_feature=None):
        """ Estimate optical flow between pair of frames """
        N, _, H, W = image1.shape
        if iters is None:
            iters = self.args.iters
        if flow_gt is None:
            flow_gt = torch.zeros(N, 2, H, W, device=image1.device)
        mft_searaft = True
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0
        image1 = image1.contiguous()
        image2 = image2.contiguous()
        flow_predictions = []
        info_predictions = []
        occl_predictions = []
        uncertainty_predictions = []

        # padding
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        N, _, H, W = image1.shape
        dilation = torch.ones(N, 1, H//8, W//8, device=image1.device)
        # run the context network
        cnet = self.cnet(torch.cat([image1, image2], dim=1))
        cnet = self.init_conv(cnet)
        net, context = torch.split(cnet, [self.args.dim, self.args.dim], dim=1)

        # init flow
        flow_update = self.flow_head(net)
        self.a = 0.00001
        res_flow_update = self.residual_branch(net) * self.a
        self.update_weight(self.num, 10000)
        self.num = self.num + 1
        flow_update=flow_update+res_flow_update

        weight_update = .25 * self.upsample_weight(net)
        flow_8x = flow_update[:, :2]
        info_8x = flow_update[:, 2:]

        flow_up, info_up = self.upsample_data(flow_8x, info_8x, weight_update)
        flow_predictions.append(flow_up)
        info_predictions.append(info_up)
            
        if self.args.iters > 0:
            # run the feature network
            fmap1_8x = self.fnet(image1)
            fmap2_8x = self.fnet(image2)
            corr_fn = CorrBlock(fmap1_8x, fmap2_8x, self.args)

        up_mask = None
        img_feature = torch.cat([image1, image2])

        for itr in range(iters):
            N, _, H, W = flow_8x.shape
            flow_8x = flow_8x.detach()
            coords2 = (coords_grid(N, H, W, device=image1.device) + flow_8x).detach()
            corr = corr_fn(coords2, dilation=dilation)

            net, motion_features = self.update_block(net, context, corr, flow_8x)
            flow_update = self.flow_head(net)
            self.a = 0.00001
            res_flow_update = self.residual_branch(net) * self.a
            # self.update_weight(20, 50000)
            flow_update=flow_update+res_flow_update

            weight_update = .25 * self.upsample_weight(net)
            flow_8x = flow_8x + flow_update[:, :2]
            info_8x = flow_update[:, 2:]
            # upsample predictions

            flow_up, info_up = self.upsample_data(flow_8x, info_8x, weight_update)
            flow_predictions.append(flow_up)
            info_predictions.append(info_up)

            if self.occlusion_estimation:
                occlusion, uncertainty = self.occlusion_block(net.detach(),  # hidden GRU state
                                                              context,  # context features
                                                              corr.detach(),  # correlation cost-volume + pyramid
                                                                              # ^ sampled at the previous flow position
                                                              flow_8x.detach(),  # flow
                                                              flow_update[:, :2].detach(),  # flow delta in last step # 比raft大两倍4-->2  
                                                              motion_features,
                                                              dino1_feature,
                                                              dino2_feature  # encoded cost-volume sample + flow
                                                              )


                # occl_up = upsample8(occlusion)  # upsample only
                occl_up = self.upsample_occ(occlusion, weight_update)  # upsample only 4,2,54,120
                # occl_up = self.upsample_flow(occlusion, up_mask, mult_coef=1.0)  # upsample only

                occl_predictions.append(occl_up)

                uncertainty_up = self.upsample_unc(uncertainty, weight_update) * self.mult_uncetrainty_upsample # 4,1,54,120
                # uncertainty_up = self.upsample_flow(uncertainty, up_mask, mult_coef=8.0, n_channels=1) # upsample and multiply by 8

                        # if self.relu_uncertainty: # not run
                        #     uncertainty_up = F.relu(uncertainty_up)

                        # if getattr(self.args, 'experimental_cleanup', False): # not run
                        #     uncertainty_up[~torch.isfinite(uncertainty_up)] = 35
                        #     uncertainty_up[uncertainty_up > 35] = 35
                        #     # print('cleaning up the mess')
                uncertainty_predictions.append(uncertainty_up)


        for i in range(len(info_predictions)):
            flow_predictions[i] = padder.unpad(flow_predictions[i])
            info_predictions[i] = padder.unpad(info_predictions[i])


        if test_mode:
            outputs = dict()
            outputs['flow'] = flow_up #除去init_flow
            if self.uncertainty_estimation:
                outputs['uncertainty'] = uncertainty_up
            if self.occlusion_estimation:
                outputs['occlusion'] = occl_up

        else:
            outputs = dict()
            outputs['flow'] = flow_predictions[1:] #除去init_flow
            if self.uncertainty_estimation:
                outputs['uncertainty'] = uncertainty_predictions
            if self.occlusion_estimation:
                outputs['occlusion'] = occl_predictions
                
        return outputs

