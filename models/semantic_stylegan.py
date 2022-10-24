# Copyright (C) 2022 ByteDance Inc.
# All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

# The software is made available under Creative Commons BY-NC-SA 4.0 license
# by ByteDance Inc. You can use, redistribute, and adapt it
# for non-commercial purposes, as long as you (a) give appropriate credit
# by citing our paper, (b) indicate any changes that you've made,
# and (c) distribute any derivative works under the same license.

# THE AUTHORS DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING ALL
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE.
# IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL
# DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING
# OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
from audioop import bias
from operator import truediv
from re import T
from einops import rearrange
import imp
import math
import random

import torch
from torch import nn
import torch.nn.functional as F
import math

import volumegan
from .utils import StyledConv, FixedStyledConv, ToRGB, PixelNorm, EqualLinear, ConvLayer, ResBlock, PositionEmbedding
from volumegan import PointsSampling, HierarchicalSampling, Renderer, NeRFSynthesisNetwork,  interpolate_feature
from volumegan import interpolate_feature_3d, interpolate_feature_triplane
PI = math.pi


class NerfSynthesis(nn.Module):
    def __init__(self,
                 use_depth,
                 depth_layers,
                 ps_cfg=dict(num_steps=12,
                             ray_start=0.88,
                             ray_end=1.12,
                             radius=1,
                             horizontal_mean=PI/2,
                             horizontal_stddev=0.3,
                             vertical_mean=PI/2,
                             vertical_stddev=0.15,
                             camera_dist='gaussian',
                             fov=12,  # TODO:需要根据数据集调整这一系列参数
                             perturb_mode=None),
                 hs_cfg=dict(clamp_mode='relu'),
                 vr_cfg=dict(clamp_mode='relu'),
                 fv_cfg=dict(feat_res=32,
                             init_res=4,
                             base_channels=128,
                             output_channels=16,
                             w_dim=512),
                 embed_cfg=dict(input_dim=3,
                                max_freq_log2=10-1,
                                N_freqs=10),
                 fg_cfg=dict(num_layers=4,
                             hidden_dim=128,
                             activation_type='lrelu'),
                 bg_cfg=None,
                 out_dim=128,):
        super().__init__()
        self.use_depth = use_depth
        self.pointsampler = PointsSampling(**ps_cfg)
        self.hierachicalsampler = HierarchicalSampling(**hs_cfg)
        self.volumerenderer = Renderer(**vr_cfg)
        self.nerfmlp = NeRFSynthesisNetwork(
            depth_layers,
            fv_cfg=fv_cfg,
            embed_cfg=embed_cfg,
            fg_cfg=fg_cfg,
            bg_cfg=bg_cfg,
            out_dim=out_dim,
        )
        

    def forward(self, w, ps_results, noise_std=0, ps_kwargs=dict(), nerf_res=32):
        ps_results = self.pointsampler(batch_size=w.shape[0],
                                resolution=nerf_res,
                                **ps_kwargs)  # TODO:test的时候可以更新
        nerf_synthesis_results = self.nerfmlp(wp=w,
                                              pts=ps_results['pts'],
                                              dirs=ps_results['ray_dirs'])
        hs_results = self.hierachicalsampler(coarse_rgbs=nerf_synthesis_results['rgb'],
                                             coarse_sigmas=nerf_synthesis_results['sigma'],
                                             pts_z=ps_results['pts_z'],
                                             ray_origins=ps_results['ray_origins'],
                                             ray_dirs=ps_results['ray_dirs'],
                                             noise_std=noise_std)
        fine_nerf_synthesis_results = self.nerfmlp(wp=w,
                                                   pts=hs_results['pts'],
                                                   dirs=hs_results['ray_dirs'])
        # Concat coarse and fine results
        rgbs = torch.cat(
            (fine_nerf_synthesis_results['rgb'], nerf_synthesis_results['rgb']), dim=-2)
        sigmas = torch.cat(
            (fine_nerf_synthesis_results['sigma'], nerf_synthesis_results['sigma']), dim=-2)
        pts_z_all = torch.cat(
            (hs_results['pts_z'], ps_results['pts_z']), dim=-2)
        _, indices = torch.sort(pts_z_all, dim=-2)
        rgbs = torch.gather(
            rgbs, -2, indices.expand(-1, -1, -1, -1, rgbs.shape[-1]))
        sigmas = torch.gather(sigmas, -2, indices)
        pts_z_all = torch.gather(pts_z_all, -2, indices)
        # Volume Rendering
        render_results = self.volumerenderer(rgbs=rgbs,
                                             sigmas=sigmas,
                                             pts_z=pts_z_all,
                                             noise_std=noise_std)
        # nerf output
        nerf_feat = render_results['rgb'].permute(0, 3, 1, 2)
        if self.use_depth:
            nerf_dep = render_results['depth'].permute(0, 3, 1, 2)
        else:
            nerf_dep = torch.zeros(nerf_feat.size(0), 1, nerf_feat.size(
                2), nerf_feat.size(3)).to(nerf_feat.device)
        return nerf_feat, nerf_dep

class TriplaneGeneraotr(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_channel, style_dim, n_layers=8, depth_layers=8, use_depth=False, detach_texture=False):
        super().__init__()
        self.local_nets = nn.ModuleList()
        for _ in range(3):
            self.local_nets.append(LocalGenerator(in_channel, out_channel, hidden_channel, style_dim,
                                        n_layers=n_layers, depth_layers=depth_layers, use_depth=use_depth, detach_texture=detach_texture))
    
    def forward(self, x, latent):
        feats, depths= [],[]
        for i in range(3):
            feat, depth = self.local_nets[i](x, latent)
            feats.append(feat)
            depths.append(depth)
        feats = torch.stack(feats, dim=1)
        depths = torch.stack(depths, dim=1)
        return feats, depths


class LocalGenerator(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_channel, style_dim, n_layers=8, depth_layers=8, use_depth=False, detach_texture=False, feat_size=64, depth=0):
        super().__init__()
        self.n_layers = n_layers
        self.depth_layers = depth_layers
        self.use_depth = use_depth
        self.detach_texture = detach_texture
        self.linears = nn.ModuleList()
        for _ in range(n_layers):
            self.linears.append(StyledConv(
                in_channel, hidden_channel, 1, style_dim, inject_noise=False))
            in_channel = hidden_channel
        self.to_feat = ToRGB(hidden_channel, out_channel, style_dim)
        self.to_sigma = ToRGB(hidden_channel, 1, style_dim, feat_size=feat_size)
        if self.use_depth:
            self.to_depth = ToRGB(hidden_channel, 1, style_dim, bias=depth, feat_size=feat_size)

    def forward(self, x, latent):
        # depth = torch.ones(x.size(0), 1, x.size(2), x.size(3)).to(x.device)*1e-8
        for i, linear in enumerate(self.linears):
            x = linear(x, latent[:, i])
            # if self.use_depth and i == self.depth_layers-1:
            if i == self.depth_layers-1:
                if self.use_depth:
                    # depth = torch.clamp(self.to_depth(x, None), 0, 1)
                    depth = self.to_depth(x, None)
                    if self.detach_texture and i < self.n_layers-1:
                        x = x.detach()
                else:
                    depth = torch.zeros(x.size(0), 1, x.size(2), x.size(3)).to(x.device)
                sigma = F.relu(self.to_sigma(x, None))
        feat = self.to_feat(x, None)
        return feat, depth, sigma


class RenderNet(nn.Module):
    def __init__(self, min_size, out_size, coarse_size, in_channel, img_dim, seg_dim, style_dim,
                 channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }
        self.out_size = out_size
        self.min_size = min_size
        self.log_out_size = int(math.log(out_size, 2))
        self.log_min_size = int(math.log(min_size, 2))
        self.coarse_size = coarse_size
        self.n_layers = (self.log_out_size-self.log_min_size) * 2

        feat_channel = in_channel
        self.convs = nn.ModuleList()
        self.noises = nn.Module()
        self.to_rgbs = nn.ModuleList()
        self.to_segs = nn.ModuleList()
        for i in range(self.log_out_size-self.log_min_size):
            cur_size = self.min_size*(2**(i+1))
            out_channel = self.channels[cur_size]
            if cur_size//2 == coarse_size:
                in_channel = in_channel + feat_channel
            self.convs.append(FixedStyledConv(
                in_channel, out_channel, 3, style_dim, upsample=True, blur_kernel=blur_kernel))
            self.convs.append(FixedStyledConv(
                out_channel, out_channel, 3, style_dim, upsample=False, blur_kernel=blur_kernel))
            self.noises.register_buffer(
                f'noise_{2*i}', torch.randn(1, 1, cur_size, cur_size))
            self.noises.register_buffer(
                f'noise_{2*i+1}', torch.randn(1, 1, cur_size, cur_size))
            self.to_rgbs.append(
                ToRGB(out_channel, img_dim, style_dim, upsample=True))
            self.to_segs.append(
                ToRGB(out_channel, seg_dim, style_dim, upsample=True))
            in_channel = out_channel

    def get_noise(self, noise, randomize_noise):
        if noise is None:
            if randomize_noise:
                noise = [None] * self.n_layers
            else:
                noise = [getattr(self.noises, f"noise_{i}")
                         for i in range(self.n_layers)]
        return noise

    def forward(self, x, noise=None, randomize_noise=False, skip_rgb=None, skip_seg=None):
        noise = self.get_noise(noise, randomize_noise)
        x_orig, x = x, F.adaptive_avg_pool2d(x, (self.min_size, self.min_size))
        rgb, seg = None, None
        for i in range(self.log_out_size-self.log_min_size):
            if x.size(2) == self.coarse_size:
                x = torch.cat((x, x_orig), 1)
            x = self.convs[2*i](x, None, noise=noise[2*i])
            x = self.convs[2*i+1](x, None, noise=noise[2*i+1])
            rgb = self.to_rgbs[i](x, None, rgb)
            if skip_rgb is not None and rgb.size(2) == skip_rgb.size(2):
                rgb += skip_rgb
            seg = self.to_segs[i](x, None, seg)
            if skip_seg is not None and seg.size(2) == skip_seg.size(2):
                seg += skip_seg
        return rgb, seg


class SemanticGenerator(nn.Module):
    def __init__(self, size=256, style_dim=512, n_mlp=8, channel_multiplier=2,
                 blur_kernel=[1, 3, 3, 1], lr_mlp=0.01, seg_dim=2,
                 coarse_size=64, min_feat_size=8, local_layers=10, local_channel=64,
                 coarse_channel=512, base_layers=2, depth_layers=6, residual_refine=True,
                 detach_texture=False, transparent_dims=(),
                 ps_cfg=dict(num_steps=12,
                             ray_start=0.8,
                             ray_end=1.2,
                             radius=1,
                             horizontal_mean=PI/2,
                             horizontal_stddev=0.4,
                             vertical_mean=PI/2,
                             vertical_stddev=0.2,
                             camera_dist='gaussian',
                             fov=12,  # TODO:需要根据数据集调整这一系列参数
                             perturb_mode=None),
                hs_cfg=dict(clamp_mode='relu'),
                vr_cfg=dict(clamp_mode='relu'),
                 **kwargs):
        super().__init__()

        assert depth_layers <= local_layers
        assert coarse_size < size
        assert min_feat_size < coarse_size and coarse_size % min_feat_size == 0
        self.size = size
        self.style_dim = style_dim
        self.log_size = int(math.log(size, 2))
        self.n_local = self.seg_dim = seg_dim
        self.base_layers = base_layers
        self.local_layers = local_layers
        self.local_channel = local_channel
        self.depth_layers = depth_layers
        self.coarse_channel = coarse_channel
        self.coarse_size = coarse_size
        self.min_feat_size = min_feat_size
        self.residual_refine = residual_refine
        self.detach_texture = detach_texture
        self.transparent_dims = list(transparent_dims)
        self.n_latent = self.base_layers + self.n_local * 2  # Default latent space
        self.n_latent_expand = self.n_local * self.local_layers  # Expanded latent space
        # self.feat_mlp = nn.Linear(coarse_channel*3,coarse_channel)
        print(
            f"n_latent: {self.n_latent}, n_latent_expand: {self.n_latent_expand}")

        self.pos_embed = PositionEmbedding(
            2, self.local_channel, N_freqs=self.log_size)
        self.local_nets = nn.ModuleList()
        depths = [5, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
        for i in range(0, self.n_local):
            use_depth = True  # disable pseudo-depth for background generator   #TODO:背景不需要深度
            self.local_nets.append(LocalGenerator(local_channel, coarse_channel, local_channel, style_dim,
                                                  n_layers=local_layers, depth_layers=depth_layers, use_depth=use_depth, detach_texture=detach_texture, feat_size=coarse_size, depth=depths[i]))
            # self.local_nets.append(NerfSynthesis(use_depth, self.depth_layers))

        self.render_net = RenderNet(min_feat_size, size, coarse_size, coarse_channel, 3, seg_dim, style_dim,
                                    channel_multiplier=channel_multiplier, blur_kernel=blur_kernel)

        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(
                EqualLinear(style_dim, style_dim,
                            lr_mul=lr_mlp, activation='fused_lrelu')
            )
        self.style = nn.Sequential(*layers)

        self.pointsampler = PointsSampling(**ps_cfg)
        self.hierachicalsampler = HierarchicalSampling(**hs_cfg)
        self.volumerenderer = Renderer(**vr_cfg)
        self.to_depths = nn.ModuleList()
        for i in range(self.n_local):
            self.to_depths.append(ToRGB(coarse_channel, 1, style_dim))
    def truncate_styles(self, styles, truncation, truncation_latent):
        if truncation < 1:
            style_t = []
            for style in styles:
                style_t.append(
                    truncation_latent + truncation *
                    (style - truncation_latent)
                )
            styles = style_t
        return styles

    def expand_latents(self, latent):
        ''' Expand the default latent codes.
        Input:
            latent: tensor of N x (n_base + n_local x 2) x style_dim
        Output:
            latent_expanded: tensor of N x (n_local x local_layers) x style_dim
        '''
        assert latent.ndim == 3
        if latent.size(1) == self.n_latent_expand:
            return latent

        assert latent.size(1) == self.n_latent
        latent_expanded = []
        for i in range(self.n_local):
            if i == 0:
                # Disable base code for background
                if self.depth_layers > 0:
                    latent_expanded.append(
                        latent[:, 2*i+self.base_layers].unsqueeze(1).repeat(1, self.depth_layers, 1))
                if self.local_layers - self.depth_layers > 0:
                    latent_expanded.append(latent[:, 2*i+self.base_layers+1].unsqueeze(
                        1).repeat(1, self.local_layers-self.depth_layers, 1))
            else:
                if self.base_layers > 0:
                    latent_expanded.append(latent[:, :self.base_layers])
                if self.depth_layers - self.base_layers > 0:
                    latent_expanded.append(latent[:, 2*i+self.base_layers].unsqueeze(
                        1).repeat(1, self.depth_layers-self.base_layers, 1))
                if self.local_layers - self.depth_layers > 0:
                    latent_expanded.append(latent[:, 2*i+self.base_layers+1].unsqueeze(
                        1).repeat(1, self.local_layers-self.depth_layers, 1))
        latent_expanded = torch.cat(latent_expanded, 1)
        return latent_expanded

    def mix_styles(self, styles):
        if len(styles) < 2:
            # Input is the latent code
            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, self.n_latent, 1)
            else:
                latent = styles[0]
        elif len(styles) > 2:
            # Input is the latent code (list)
            latent = torch.stack(styles, 1)
        else:
            # Input are two latent codes -> style mixing
            base_latent = styles[0].unsqueeze(1).repeat(1, self.base_layers, 1)
            latent = [base_latent]
            for i in range(self.n_local):
                N = styles[0].size(0)
                latent1 = []
                latent2 = []
                for j in range(N):
                    inject_index = random.randint(0, 2)
                    if inject_index == 0:
                        latent1_ = latent2_ = styles[0][j]
                    elif inject_index == 1:
                        latent1_, latent2_ = styles[0][j], styles[1][j]
                    else:
                        latent1_ = latent2_ = styles[1][j]
                    latent1.append(latent1_)
                    latent2.append(latent2_)
                latent1 = torch.stack(latent1)  # N x style_dim
                latent2 = torch.stack(latent2)  # N x style_dim
                latent.append(latent1.unsqueeze(1))
                latent.append(latent2.unsqueeze(1))
            latent = torch.cat(latent, 1)  # N x n_latent x style_dim
        # N  x (n_local x local_layers) x style_dim
        latent = self.expand_latents(latent)
        return latent

    def composite(self, feats, depths, mask=None):
        seg = F.softmax(torch.cat(depths, dim=1), dim=1)
        if mask is not None:
            # If mask is given, ignore specified classes
            assert mask.size(0) == seg.size(0)
            assert mask.size(1) == seg.size(1)
            mask = mask.reshape(seg.size(0), seg.size(1), 1, 1)
            seg = seg * mask
            seg = seg / (seg.sum(1, keepdim=True)+1e-8)
        if len(self.transparent_dims) > 0:
            coefs = torch.tensor([0. if i in self.transparent_dims else 1. for i in range(
                self.seg_dim)]).view(1, -1, 1, 1).to(seg.device)
            seg_normal = seg * coefs  # zero out transparent classes
            # re-normalize the feature map
            seg_normal = seg_normal / (seg_normal.sum(1, keepdim=True)+1e-8)

            coefs = torch.tensor([1. if i in self.transparent_dims else 0. for i in range(
                self.seg_dim)]).view(1, -1, 1, 1).to(seg.device)
            seg_trans = seg * coefs  # zero out non-transparent classes

            weights = seg_normal + seg_trans
        else:
            weights = seg
        feat = sum([feats[i]*weights[:, i:i+1] for i in range(self.seg_dim)])
        return feat, seg

    def composite_feat(self, feats, sigmas, mask=None):
        sigmas = torch.cat(sigmas, dim=2)
        seg = F.softmax(sigmas, dim=2)
        if mask is not None:    #TODO:for component
            # If mask is given, ignore specified classes
            assert mask.size(0) == seg.size(0)
            assert mask.size(1) == seg.size(1)
            mask = mask.reshape(seg.size(0), seg.size(1), 1, 1)
            seg = seg * mask
            seg = seg / (seg.sum(1, keepdim=True)+1e-8)
        if len(self.transparent_dims) > 0:
            coefs = torch.tensor([0. if i in self.transparent_dims else 1. for i in range(
                self.seg_dim)]).view(1, 1, -1, 1, 1).to(seg.device)
            seg_normal = seg * coefs  # zero out transparent classes
            # re-normalize the feature map
            seg_normal = seg_normal / (seg_normal.sum(2, keepdim=True)+1e-8)

            coefs = torch.tensor([1. if i in self.transparent_dims else 0. for i in range(
                self.seg_dim)]).view(1, 1, -1, 1, 1).to(seg.device)
            seg_trans = seg * coefs  # zero out non-transparent classes

            weights = seg_normal + seg_trans
        else:
            weights = seg
        feat = sum([feats[i]*weights[:,:,i:i+1] for i in range(self.seg_dim)])
        # feat = sum(feats[i][:,0,:,None]*weights[:,0,i:i+1,None]*feats[i][:,1,:,:,None]*weights[:,1,i:i+1,:,None]*feats[i][:,2,:,:,:,None]*weights[:,2,i:i+1,:,:,None] for i in range(13))
        # local_sigma = sigmas[:,0,:,None]*sigmas[:,1,:,:,None]*sigmas[:,2,:,:,:,None]
        # local_sigma = torch.zeros(sigmas[0].shape[0],13,sigmas[0].shape[3],sigmas[0].shape[3],sigmas[0].shape[3])
        # for x in range(feats[0].shape[-1]):
        #     for y in range(feats[0].shape[-1]):
        #         for z in range(feats[0].shape[-1]):
        #             local_sigma[...,x,y,z] = sigmas[:,0,:,y,z]*sigmas[:,1,:,x,z]*sigmas[:,2,:,x,y]
        local_sigma = sigmas
        sigma = torch.sum(local_sigma, dim=2, keepdim=True)
        return feat, local_sigma, sigma

    def make_coords(self, b, h, w, device):
        x_channel = torch.linspace(-1, 1, w, device=device).view(1,
                                                                 1, 1, -1).repeat(b, 1, w, 1)
        y_channel = torch.linspace(-1, 1, h, device=device).view(1,
                                                                 1, -1, 1).repeat(b, 1, 1, h)
        return torch.cat((x_channel, y_channel), dim=1)

    def volumegan_render(self, feat, ps_results, bound, sample_model=1):
        pts = ps_results['pts']
        h, w, d = pts.shape[1:4]
        pts = rearrange(pts, 'bs h w d c -> bs (h w d) c').contiguous()
        feat_volume = feat
        feat_pro = interpolate_feature_3d(pts, feat_volume, bound)
        feat_pro = rearrange(feat_pro, 'bs c (h w numd) -> bs h w numd c',h=h, w=w, numd=d)
        hs_results = self.hierachicalsampler(coarse_rgbs=feat_pro[...,1:],
                                            coarse_sigmas=feat_pro[...,0:1],
                                            pts_z=ps_results['pts_z'],
                                            ray_origins=ps_results['ray_origins'],
                                            ray_dirs=ps_results['ray_dirs'],
                                            noise_std=0,
                                            sample_model=sample_model)
        hs_pts = rearrange(hs_results['pts'], 'bs h w d c -> bs (h w d) c').contiguous()
        # ret_weight = rearrange(ret_weight, 'bs h w d c -> bs c h w d').contiguous()
        hs_feat = interpolate_feature_3d(hs_pts, feat_volume, bound)
        if sample_model == 1:
            hs_d = d
        elif sample_model == 0:
            hs_d = d*2
        else:
            hs_d = d//2
        hs_feat = rearrange(hs_feat, 'bs c (h w numd) -> bs h w numd c',h=h, w=w, numd=hs_d)
        feats_all = torch.cat([hs_feat, feat_pro], dim=-2)
        pts_z_all = torch.cat((hs_results['pts_z'], ps_results['pts_z']), dim=-2)
        # feats_all = feat_pro
        # pts_z_all = ps_results['pts_z']
        _, indices = torch.sort(pts_z_all, dim=-2)
        feats_all = torch.gather(feats_all, -2, indices.expand(-1,-1,-1,-1,feats_all.shape[-1]))
        hs_pts = rearrange(hs_pts, 'bs ( h w d ) c -> bs h w d c',h=h, w=w, d=hs_d)
        pts = rearrange(pts, 'bs ( h w d ) c -> bs h w d c',h=h, w=w, d=d)
        pts_all = torch.cat([hs_pts, pts], dim=-2)
        pts_all = torch.gather(pts_all, -2, indices.expand(-1,-1,-1,-1,pts_all.shape[-1]))
        pts_all = rearrange(pts_all, 'bs h w d c -> bs (h w d) c').contiguous()
        rgbs = feats_all[...,1:]
        sigmas = feats_all[...,0:1]
        pts_z_all = torch.gather(pts_z_all, -2, indices)
        render_results = self.volumerenderer(rgbs=rgbs,
                                            sigmas=sigmas,
                                            pts_z=pts_z_all,
                                            noise_std=0)
        nerf_feat = render_results['rgb'].permute(0, 3, 1, 2)
        ret_weight = render_results['weights']
        ret_weight = rearrange(ret_weight, 'bs h w d c -> bs c h w d').contiguous()
        return nerf_feat, ret_weight, pts_all

    def forward(self,
                latent,
                coords=None,
                truncation=1,
                truncation_latent=None,
                noise=None,
                randomize_noise=True,
                input_is_latent=False,
                composition_mask=None,
                return_latents=False,
                return_coarse=False,
                return_all=False,
                pre_train=False,
                sample_model=1,
                ps_kwargs = dict()
                ):
        if not input_is_latent:
            latent = [self.style(s) for s in latent]

        latent = self.truncate_styles(latent, truncation, truncation_latent)
        latent = self.mix_styles(latent)  # expanded latent code

        # Position Embedding
        if coords is None:
            coords = self.make_coords(
                latent.shape[0], self.coarse_size, self.coarse_size, latent.device)
            coords = [coords.clone() for _ in range(self.n_local)]

        ps_results = self.pointsampler(batch_size=latent.shape[0],
                                        resolution=self.coarse_size,
                                        **ps_kwargs)  # TODO:test的时候可以更新
        pts = ps_results['pts']
        h, w, d = pts.shape[1:4]
        pts = rearrange(pts, 'bs h w d c -> bs (h w d) c').contiguous()

        bound = [[-0.1886, -0.1671, -0.1956],[0.1887, 0.1692, 0.1872]]
        # bound = [[-0.1886, -0.1671, -0.1956*0.6],[0.1887, 0.1692, 0.1872*0.6]]
        bound = torch.Tensor(bound).to(pts)
        # Local Generators
        sigmas_3d = []
        feats_3d = []
        sample_sigmas = []
        for i in range(self.n_local):
            x = self.pos_embed(coords[i])
            local_latent = latent[:, i *
                                  self.local_layers:(i+1)*self.local_layers]
            feat, depth, sigma = self.local_nets[i](x, local_latent)
            if i == 1:
                face_depth = depth
            if i > 1:
                depth = face_depth + depth
            weight = torch.arange(self.coarse_size//2).reshape(1,1,self.coarse_size//2,1,1).repeat(depth.shape[0],depth.shape[1], 1, depth.shape[2],depth.shape[3]).to(depth.device)
            weight = torch.exp(-2*torch.square(weight - depth.unsqueeze(2)))
            feat_3d = weight*feat.unsqueeze(2)
            sigma_3d = weight*sigma.unsqueeze(2)
            # local_sigma = interpolate_feature_3d(pts, sigma_3d,bound)
            # sample_sigma = rearrange(local_sigma, 'bs c (h w numd) -> bs c h w numd',h=h, w=w, numd=d)
            # sample_sigma = torch.sum(sample_sigma,dim=-1,keepdim=False)
            feats_3d.append(feat_3d)
            sigmas_3d.append(sigma_3d)
            # sample_sigmas.append(sample_sigma)
        sigma = sum(sigmas_3d)
        feat = sum(feats_3d)
        # sample_sigmas = torch.cat(sample_sigmas, dim=1)
        feat_comp, ret_weight, pts_all = self.volumegan_render(torch.cat([sigma,feat],dim=1), ps_results, bound, sample_model=sample_model)
        if sample_model == 1:
            numd = d*2
        elif sample_model == 0:
            numd = d*3
        else:
            numd = d + d//2
        for i in range(self.n_local):
            local_sigma = interpolate_feature_3d(pts_all, sigmas_3d[i],bound)
            sample_sigma = rearrange(local_sigma, 'bs c (h w numd) -> bs c h w numd',h=h, w=w, numd=numd)
            sample_sigmas.append(sample_sigma)
        sample_sigmas = torch.cat(sample_sigmas, dim=1)
        # ret_weight = F.softmax(ret_weight, dim=-1)
        if pre_train:
            sample_sigmas = torch.sum(sample_sigmas, dim=-1)
        else:
            sample_sigmas = torch.sum(sample_sigmas*ret_weight, dim=-1)
        # sample_sigmas = torch.sum(sample_sigmas, dim=-1)
        seg_coarse = F.softmax(sample_sigmas, dim=1)
        # feats_2d = []
        # depths = []
        # for i in range(self.n_local):
        #     feat_2d = self.volumegan_render(torch.cat([sigmas_tri[i], feats_tri[i]], dim=2), ps_results)
        #     if i>0:
        #         depth = self.to_depths[i](feat_2d, None)
        #     else:
        #         depth = torch.zeros(feat_2d.size(0), 1, feat_2d.size(2), feat_2d.size(3)).to(feat_2d.device)
        #     feats_2d.append(feat_2d)
        #     depths.append(depth)
        # # Composition and render
        # feat_comp, seg_coarse = self.composite(feats_2d, depths, mask=composition_mask)

        # depths = []
        # for i in range(13):
        #     local_depth = -self.volumerenderer(rgbs=feats_all[...,i+1:i+2],
        #                                     sigmas=feats_all[...,i+1:i+2],
        #                                     pts_z=pts_z_all,
        #                                     noise_std=0)['depth'].permute(0,3,1,2)
        #     depths.append(local_depth)
        # seg_coarse = F.softmax(torch.cat(depths,dim=1),dim=1)
        seg_coarse = 2*seg_coarse-1  # normalize to [-1,1]

        skip_seg = seg_coarse if self.residual_refine else None
        rgb, seg = self.render_net(
            feat_comp.contiguous(), noise=noise, randomize_noise=randomize_noise, skip_rgb=None, skip_seg=skip_seg.contiguous())

        if return_latents:
            return rgb, latent
        elif return_coarse:
            return rgb, seg_coarse
        elif return_all:
            return rgb, seg, seg_coarse, [seg_coarse[:,i:i+1] for i in range(self.n_local)], latent
        else:
            return rgb, seg


class DualBranchDiscriminator(nn.Module):
    def __init__(self, img_size, seg_size, img_dim, seg_dim, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }
        log_size = int(math.log(img_size, 2))
        if seg_size is None:
            seg_size = img_size

        convs = [ConvLayer(img_dim, self.channels[img_size], 1)]
        in_channel = self.channels[img_size]
        for i in range(log_size, 2, -1):
            out_channel = self.channels[2 ** (i - 1)]
            convs.append(ResBlock(in_channel, out_channel, blur_kernel))
            in_channel = out_channel
        self.convs_img = nn.Sequential(*convs)

        log_size = int(math.log(seg_size, 2))
        convs = [ConvLayer(seg_dim, self.channels[seg_size], 1)]
        in_channel = self.channels[seg_size]
        for i in range(log_size, 2, -1):
            out_channel = self.channels[2 ** (i - 1)]
            convs.append(ResBlock(in_channel, out_channel, blur_kernel))
            in_channel = out_channel
        self.convs_seg = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, self.channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(self.channels[4] * 4 * 4,
                        self.channels[4], activation='fused_lrelu'),
            EqualLinear(self.channels[4], 1),
        )

    def _cal_stddev(self, x):
        batch, channel, height, width = x.shape
        group = min(batch, self.stddev_group)
        stddev = x.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        x = torch.cat([x, stddev], 1)

        return x

    def forward(self, img, seg=None):
        batch = img.shape[0]

        out = self.convs_img(img)
        if seg is not None:
            out = out + self.convs_seg(seg)

        out = self._cal_stddev(out)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out
