import torch
import torch.nn as nn
import torch.nn.functional as F
from thirdparty.RAFT.layer import ConvNextBlock

class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=2):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))



class ResidualBranch(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=6):
        super(ResidualBranch, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(hidden_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        return residual

class BasicMotionEncoder(nn.Module):
    def __init__(self, args, dim=128):
        super(BasicMotionEncoder, self).__init__()
        cor_planes = args.corr_channel
        self.convc1 = nn.Conv2d(cor_planes, dim*2, 1, padding=0)
        self.convc2 = nn.Conv2d(dim*2, dim+dim//2, 3, padding=1)
        self.convf1 = nn.Conv2d(2, dim, 7, padding=3)
        self.convf2 = nn.Conv2d(dim, dim//2, 3, padding=1)
        self.conv = nn.Conv2d(dim*2, dim-2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)
    

class OcclusionHead(nn.Module):
    # two output layers - according to contflow
    def __init__(self, input_dim=128, hidden_dim=256, architecture=None):
        super(OcclusionHead, self).__init__()
        self.architecture = architecture

        if architecture is None or architecture == 'simple':
            self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
            self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
            self.relu = nn.ReLU(inplace=True)
        elif architecture == 'morelayers':
            self.model = nn.Sequential(
                nn.Conv2d(input_dim, hidden_dim, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, 2, 3, padding=1),
            )
        else:
            raise NotImplementedError('This type of architecture is not implemented')

    def forward(self, x):
        if self.architecture is None or self.architecture == 'simple':
            x = self.conv2(self.relu(self.conv1(x)))
            return x
        else:
            x = self.model(x)
            return x



class UncertaintyHead(nn.Module):
    # single output layer
    def __init__(self, input_dim=128, hidden_dim=256, architecture=None):
        super(UncertaintyHead, self).__init__()
        self.architecture = architecture

        if architecture is None or architecture == 'simple':
            self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
            self.conv2 = nn.Conv2d(hidden_dim, 1, 3, padding=1)
            self.relu = nn.ReLU(inplace=True)
        elif architecture == 'morelayers':
            self.model = nn.Sequential(
                nn.Conv2d(input_dim, hidden_dim, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, 1, 3, padding=1),
            )
        else:
            raise NotImplementedError('This type of architecture is not implemented')

    def forward(self, x):
        if self.architecture is None or self.architecture == 'simple':
            return self.conv2(self.relu(self.conv1(x)))
        else:
            return self.model(x)



class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h

class SmallMotionEncoder(nn.Module):
    def __init__(self, args):
        super(SmallMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 96, 1, padding=0)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv = nn.Conv2d(128, 80, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

class ChannelAttention(nn.Module):
    """通道注意力模块"""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 处理极小通道数的情况
        reduced_dim = max(1, channel // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channel, reduced_dim),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_dim, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.avg_pool(x).view(x.size(0), -1)
        max_out = self.max_pool(x).view(x.size(0), -1)
        
        avg_weight = self.fc(avg_out).view(x.size(0), -1, 1, 1)
        max_weight = self.fc(max_out).view(x.size(0), -1, 1, 1)
        
        return x * (avg_weight + max_weight) / 2

class DinoGuidedAttention(nn.Module):
    """Dino特征指导的全局注意力"""
    def __init__(self, channel, dino_dim):
        super().__init__()
        self.dino_proj = nn.Conv2d(dino_dim, channel, 1)  # 特征维度对齐
        self.attention = ChannelAttention(channel)        # 单通道输入
        
    def forward(self, x, dino_feat):
        # 对齐Dino特征
        dino_feat = F.interpolate(dino_feat, size=x.shape[2:], mode='bilinear', align_corners=False)
        dino_feat = self.dino_proj(dino_feat)
        
        # 使用Dino特征指导注意力
        return self.attention(x * dino_feat)    
    



class FeatureInteractionModule(nn.Module):
    """特征交互模块"""
    def __init__(self, channels, common_dim=178):
        super().__init__()
        self.channels = channels
        self.num_features = len(channels)
        
        # 交叉注意力机制
        self.cross_attention = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=common_dim, num_heads=4) for i in range(self.num_features)
        ])
        
        # 特征分组
        self.group_norm = nn.ModuleList([
            nn.GroupNorm(num_groups=4, num_channels=channels[i]) for i in range(self.num_features)
        ])

        self.key_rojection = nn.ModuleList([
            nn.Conv2d(channels[i], common_dim, kernel_size=1) for i in range(self.num_features)
        ])
        self.value_projection = nn.ModuleList([
            nn.Conv2d(channels[i], common_dim, kernel_size=1) for i in range(self.num_features)
        ])
        self.query_projection = nn.ModuleList([
            nn.Conv2d(common_dim, common_dim, kernel_size=1) for i in range(self.num_features)
        ])

    def forward(self, features, flow_dino):
            # 对每个特征进行分组归一化
            features = [self.group_norm[i](features[i]) for i in range(self.num_features)] # [8, channel[i], 54, 120]
            # dino_flow torch.Size([8, 64, 54, 120])
            
            # 交叉注意力
            interacted_features = []
            for i in range(self.num_features):
                # 将当前特征映射到低维隐空间
                query = self.query_projection[i](flow_dino).flatten(start_dim=2).permute(0, 2, 1)  # 映射到低维空间

                # 将其他特征作为 key 和 value

                keys = self.key_rojection[i](features[i]).flatten(start_dim=2).permute(0, 2, 1)   # 映射到低维空间
                values = self.value_projection[i](features[i]).flatten(start_dim=2).permute(0, 2, 1)   # 映射到低维空间

                
                # 交叉注意力
                attn_output, _ = self.cross_attention[i](query, keys, values) # 8， 6480， 64
                attn_output = attn_output.permute(1, 2, 0).reshape(flow_dino.shape)  # [B, C, H, W]
                interacted_features.append(attn_output)
            
            return interacted_features

class FlowDinoFusion(nn.Module):
    def __init__(self, hidden_channel=64):
        super(FlowDinoFusion, self).__init__()
        self.flow_conv = nn.Conv2d(4, hidden_channel, kernel_size=1)
        self.dino_conv = nn.Conv2d(384, hidden_channel, kernel_size=1)
        self.fix_conv = nn.Conv2d(hidden_channel*2, hidden_channel, kernel_size=1)
        
    def forward(self, dino_feature, flow_feature):
        flow_feature = self.flow_conv(flow_feature)
        dino_feature = dino_feature.permute(0, 3, 2, 1)  # (8, 120, 54, 384) -> (8, 4, 54, 120)

        dino_feature = self.dino_conv(dino_feature)
        flow_combine = torch.cat([flow_feature, dino_feature], dim=1)
        flow_combine = self.fix_conv(flow_combine)
        return flow_combine


class OcclusionAndUncertaintyBlock(nn.Module):
    def __init__(self, args, hidden_dim=96, dino_dim=768):
        super(OcclusionAndUncertaintyBlock, self).__init__()
        self.args = args

        architecture = 'simple'
        if 'morelayers' in args.occlusion_module:
            architecture = 'morelayers'
        self.occlusion_detach = getattr(args, 'occlusion_input_detach', False)
        self.uncertainty_detach = getattr(args, 'uncertainty_input_detach', False)

        self.occl_head = OcclusionHead(hidden_dim, hidden_dim=128, architecture=architecture)

        if 'with_uncertainty' in args.occlusion_module:
            if 'separate' in args.occlusion_module:
                self.uncertainty_head = UncertaintyHead(hidden_dim, hidden_dim=128, architecture=architecture)
            else:
                raise NotImplementedError(f'Type {args.occlusion_module} of occlusion/uncertainty module is not implemented')


        self.ca_net = ChannelAttention(128)
        self.ca_inp = ChannelAttention(128)
        self.ca_corr = ChannelAttention(324)
        self.ca_flow = ChannelAttention(2)
        self.ca_delta_flow = ChannelAttention(2)
        self.ca_motion = ChannelAttention(128)

        self.dino_flow = FlowDinoFusion(hidden_channel=128) # c = 64
        
        # 第二步：特征交互模块
        self.feature_interaction = FeatureInteractionModule(channels=[128, 128, 324, 128], common_dim=128)
        


    def forward(self, net, inp, corr, flow, delta_flow, motion_features, dino1_feature, dino2_feature):
        # inp = torch.cat([net, inp, corr, flow, delta_flow, motion_features], dim=1)
        net = self.ca_net(net)
        inp = self.ca_inp(inp)
        corr = self.ca_corr(corr)
        flow = self.ca_flow(flow)
        delta_flow = self.ca_delta_flow(delta_flow)
        motion_features = self.ca_motion(motion_features)
        
        flow_combine = torch.cat([flow, delta_flow], dim=1)  # 在通道维度拼接

        dino_flow_feature = self.dino_flow(dino1_feature, flow_combine) # c=128
        # 第二步：特征交互
        features = [net, inp, corr, motion_features]
        res_features = self.feature_interaction(features, dino_flow_feature)
        
        # 特征拼接
        combined = torch.cat(res_features, dim=1)
        
        # 第三步：Dino特征指导的全局注意力
        # combined = self.dino_attention(combined, dino_feat)
        orig_inp = combined

        if ('with_uncertainty' not in self.args.occlusion_module) or ('separate' not in self.args.occlusion_module):
            raise NotImplementedError(f'Type {self.args.occlusion_module} of occlusion/uncertainty module is not implemented')

        if self.occlusion_detach:
            inp = orig_inp.detach()
        else:
            inp = orig_inp
        occl = self.occl_head(inp)

        if self.uncertainty_detach:
            inp = orig_inp.detach()
        else:
            inp = orig_inp
        uncertainty = self.uncertainty_head(inp)
        return occl, uncertainty

class BasicUpdateBlock_raft(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock_raft, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

    def forward(self, net, inp, corr, flow, upsample=True):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow, motion_features


class BasicUpdateBlock(nn.Module):
    def __init__(self, args, hdim=128, cdim=128):
        #net: hdim, inp: cdim
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args, dim=cdim)
        self.refine = []
        for i in range(args.num_blocks):
            self.refine.append(ConvNextBlock(2*cdim+hdim, hdim))
        self.refine = nn.ModuleList(self.refine)

        # self.mask = nn.Sequential(
        #     nn.Conv2d(128, 256, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 64*9, 1, padding=0))
        # self.flow_head = FlowHead(hdim, hidden_dim=256)

    def forward(self, net, inp, corr, flow, upsample=True):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        for blk in self.refine:
            net = blk(torch.cat([net, inp], dim=1))

        # delta_flow = self.flow_head(net)
        # mask = .25 * self.mask(net)
        return net, motion_features
