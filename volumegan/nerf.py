import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from volumegan import interpolate_feature
from third_party.stylegan2_official_ops import fma
from third_party.stylegan2_official_ops import bias_act
from third_party.stylegan2_official_ops import upfirdn2d
from third_party.stylegan2_official_ops import conv2d_gradfix


class NeRFSynthesisNetwork(nn.Module):
    def __init__(self,
                 fv_cfg=dict(feat_res=32,
                             init_res=4,
                             base_channels=512//2,
                             output_channels=32,
                             w_dim=512),
                 embed_cfg=dict(input_dim=3,
                                max_freq_log2=10-1,
                                N_freqs=10),
                 fg_cfg=dict(num_layers=4,
                             hidden_dim=256,
                             activation_type='lrelu',
                             ),
                 bg_cfg=None,
                 out_dim=512,
                 ):
        super().__init__()
        self.fg_cfg = fg_cfg
        self.bg_cfg = bg_cfg

        self.fv = FeatureVolume(**fv_cfg)
        self.fv_cfg = fv_cfg

        self.fg_embedder = Embedder(**embed_cfg)

        input_dim = self.fg_embedder.out_dim + self.fv_cfg['output_channels']

        self.fg_mlps = self.build_mlp(input_dim=input_dim, **fg_cfg)
        self.fg_density = DenseLayer(in_channels=fg_cfg['hidden_dim'],
                                           out_channels=1,
                                           add_bias=True,
                                           init_bias=0.0,
                                           use_wscale=True,
                                           wscale_gain=1,
                                           lr_mul=1,
                                           activation_type='linear')
        self.fg_color = DenseLayer(in_channels=fg_cfg['hidden_dim'],
                                             out_channels=out_dim,
                                             add_bias=True,
                                             init_bias=0.0,
                                             use_wscale=True,
                                             wscale_gain=1,
                                             lr_mul=1,
                                             activation_type='linear')
        if self.bg_cfg:
            self.bg_embedder = Embedder(**bg_cfg.embed_cfg)
            input_dim = self.bg_embedder.out_dim
            self.bg_mlps = self.build_mlp(input_dim, **bg_cfg)
            self.bg_density = DenseLayer(in_channels=bg_cfg['hidden_dim'],
                                                   out_channels=1,
                                                   add_bias=True,
                                                   init_bias=0.0,
                                                   use_wscale=True,
                                                   wscale_gain=1,
                                                   lr_mul=1,
                                                   activation_type='linear')
            self.bg_color = DenseLayer(in_channels=bg_cfg['hidden_dim'],
                                                 out_channels=out_dim,
                                                 add_bias=True,
                                                 init_bias=0.0,
                                                 use_wscale=True,
                                                 wscale_gain=1,
                                                 lr_mul=1,
                                                 activation_type='linear')

    def init_weights(self,):
        pass

    def build_mlp(self, input_dim, num_layers, hidden_dim, activation_type, **kwargs):
        default_conv_cfg = dict(resolution=32,
                                w_dim=512,
                                kernel_size=1,
                                add_bias=True,
                                scale_factor=1,
                                filter_kernel=None,
                                demodulate=True,
                                use_wscale=True,
                                wscale_gain=1,
                                lr_mul=1,
                                noise_type='none',
                                conv_clamp=None,
                                eps=1e-8)
        mlp_list = nn.ModuleList()
        in_ch = input_dim
        out_ch = hidden_dim
        for _ in range(num_layers):
            mlp = ModulateConvLayer(in_channels=in_ch,
                                    out_channels=out_ch,
                                    activation_type=activation_type,
                                    **default_conv_cfg)
            mlp_list.append(mlp)
            in_ch = out_ch
            out_ch = hidden_dim

        return mlp_list


    def forward(self, wp, pts, dirs, fused_modulate=False, impl='cuda'):

        hi, wi = pts.shape[1:3]
        fg_pts = rearrange(pts, 'bs h w d c -> bs (h w) d c').contiguous()
        w = wp              
        # pts: bs, h*h, d, 3
        fg_pts_embed = self.fg_embedder(fg_pts)
        bs, nump, numd, c = fg_pts_embed.shape
        fg_pts_embed = rearrange(fg_pts_embed, 'bs nump numd c -> bs c (nump numd) 1').contiguous()
        x = fg_pts_embed

        # feature volume
        if w.ndim == 3:
            fvw = w[:, 0]
        else:
            fvw = w
        volume = self.fv(fvw)
        # interpolate features from feature volume
        # point features: batch_size, num_channel, num_point
        bounds = self.fv_cfg.get('bounds', [[-0.1886, -0.1671, -0.1956],
                               [0.1887, 0.1692, 0.1872]])
        bounds = torch.Tensor(bounds).to(pts)

        fg_pts_sam = rearrange(fg_pts, 'bs nump numd c -> bs (nump numd) c')
        input_f = interpolate_feature(fg_pts_sam, volume, bounds)
        input_f = rearrange(input_f, 'bs c numd -> bs c numd 1')
        x = torch.cat([input_f, x], dim=1)

        for mlp_idx, fg_mlp in enumerate(self.fg_mlps):
            if wp.ndim == 3:
                lw = wp[:, mlp_idx]
            else:
                lw = wp
            x, style = fg_mlp(x, lw, fused_modulate=fused_modulate, impl=impl)
        fg_feat = rearrange(x, 'bs c (nump numd) 1 -> (bs nump numd) c', numd=numd).contiguous()
        fg_density = self.fg_density(fg_feat)
        fg_color = self.fg_color(fg_feat)

        fg_density = rearrange(fg_density, '(bs h w numd) c -> bs h w numd c', h=hi, w=wi, numd=numd).contiguous()
        fg_color = rearrange(fg_color, '(bs h w numd) c -> bs h w numd c', h=hi, w=wi, numd=numd).contiguous()

        if self.bg_cfg is not None and bg_pts is not None:
            # inverted sphere parameterization
            r = torch.norm(bg_pts, dim=-1)
            bg_pts = bg_pts / r[..., None]
            bg_pts = torch.cat([bg_pts, 1 / r[..., None]], dim=-1)

            bg_pts_embed = self.bg_embedder(bg_pts)
            bs, nump, numd, c = bg_pts_embed.shape
            bg_pts_embed = rearrange(bg_pts_embed, 'bs nump numd c -> bs c (nump numd) 1').contiguous()
            x = bg_pts_embed
            for bg_mlp in self.bg_mlps:
                x, style = bg_mlp(x, w, fused_modulate=fused_modulate, impl=impl)
            bg_feat = rearrange(x, 'bs c (nump numd) 1 -> (bs nump numd) c', numd=numd).contiguous()
            bg_density = self.bg_density(bg_feat)
            bg_color = self.bg_color(bg_feat)

            bg_density = rearrange(bg_density, '(bs  nump numd) c -> bs nump numd c', nump=nump, numd=numd).contiguous()
            bg_color = rearrange(bg_color, '(bs  nump numd) c -> bs nump numd c', nump=nump, numd=numd).contiguous()
        else:
            bg_color = None
            bg_density = None

        results = {
            'sigma': fg_density,
            'rgb': fg_color,
        }
        return results

def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight,
                                      a=0.2,
                                      mode='fan_in',
                                      nonlinearity='leaky_relu')

class FeatureVolume(nn.Module):
    def __init__(
        self,
        feat_res=32,
        init_res=4,
        base_channels=256,
        output_channels=32,
        w_dim=512,
        **kwargs
    ):
        super().__init__()
        self.num_stages = int(np.log2(feat_res // init_res)) + 1

        self.const = nn.Parameter(
            torch.ones(1, base_channels, init_res, init_res, init_res))
        inplanes = base_channels
        outplanes = base_channels

        self.stage_channels = []
        for i in range(self.num_stages):
            conv = nn.Conv3d(inplanes,
                             outplanes,
                             kernel_size=(3, 3, 3),
                             padding=(1, 1, 1))
            self.stage_channels.append(outplanes)
            self.add_module(f'layer{i}', conv)
            instance_norm = InstanceNormLayer(num_features=outplanes, affine=False)

            self.add_module(f'instance_norm{i}', instance_norm)
            inplanes = outplanes
            outplanes = max(outplanes // 2, output_channels)
            if i == self.num_stages - 1:
                outplanes = output_channels

        self.mapping_network = nn.Linear(w_dim, sum(self.stage_channels) * 2)
        self.mapping_network.apply(kaiming_leaky_init)
        with torch.no_grad(): self.mapping_network.weight *= 0.25
        self.upsample = UpsamplingLayer()
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, w, **kwargs):
        scale_shifts = self.mapping_network(w)
        scales = scale_shifts[..., :scale_shifts.shape[-1]//2]
        shifts = scale_shifts[..., scale_shifts.shape[-1]//2:]

        x = self.const.repeat(w.shape[0], 1, 1, 1, 1)
        for idx in range(self.num_stages):
            if idx != 0:
                x = self.upsample(x)
            conv_layer = self.__getattr__(f'layer{idx}')
            x = conv_layer(x)
            instance_norm = self.__getattr__(f'instance_norm{idx}')
            scale = scales[:, sum(self.stage_channels[:idx]):sum(self.stage_channels[:idx + 1])]
            shift = shifts[:, sum(self.stage_channels[:idx]):sum(self.stage_channels[:idx + 1])]
            scale = scale.view(scale.shape + (1, 1, 1))
            shift = shift.view(shift.shape + (1, 1, 1))
            x = instance_norm(x, weight=scale, bias=shift)
            x = self.lrelu(x)

        return x

class Embedder(nn.Module):
    def __init__(self, input_dim, max_freq_log2, N_freqs,
                 log_sampling=True, include_input=True,
                 periodic_fns=(torch.sin, torch.cos)):
        '''
        :param input_dim: dimension of input to be embedded
        :param max_freq_log2: log2 of max freq; min freq is 1 by default
        :param N_freqs: number of frequency bands
        :param log_sampling: if True, frequency bands are linerly sampled in log-space
        :param include_input: if True, raw input is included in the embedding
        :param periodic_fns: periodic functions used to embed input
        '''
        super().__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.out_dim = 0
        if self.include_input:
            self.out_dim += self.input_dim

        self.out_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(
                2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input):
        '''
        :param input: tensor of shape [..., self.input_dim]
        :return: tensor of shape [..., self.out_dim]
        '''
        assert (input.shape[-1] == self.input_dim)

        out = []
        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))
        out = torch.cat(out, dim=-1)

        assert (out.shape[-1] == self.out_dim)
        return out


class ModulateConvLayer(nn.Module):
    """Implements the convolutional layer with style modulation."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 resolution,
                 w_dim,
                 kernel_size,
                 add_bias,
                 scale_factor,
                 filter_kernel,
                 demodulate,
                 use_wscale,
                 wscale_gain,
                 lr_mul,
                 noise_type,
                 activation_type,
                 conv_clamp,
                 eps):
        """Initializes with layer settings.

        Args:
            in_channels: Number of channels of the input tensor.
            out_channels: Number of channels of the output tensor.
            resolution: Resolution of the output tensor.
            w_dim: Dimension of W space for style modulation.
            kernel_size: Size of the convolutional kernels.
            add_bias: Whether to add bias onto the convolutional result.
            scale_factor: Scale factor for upsampling.
            filter_kernel: Kernel used for filtering.
            demodulate: Whether to perform style demodulation.
            use_wscale: Whether to use weight scaling.
            wscale_gain: Gain factor for weight scaling.
            lr_mul: Learning multiplier for both weight and bias.
            noise_type: Type of noise added to the feature map after the
                convolution (if needed). Support `none`, `spatial` and
                `channel`.
            activation_type: Type of activation.
            conv_clamp: A threshold to clamp the output of convolution layers to
                avoid overflow under FP16 training.
            eps: A small value to avoid divide overflow.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resolution = resolution
        self.w_dim = w_dim
        self.kernel_size = kernel_size
        self.add_bias = add_bias
        self.scale_factor = scale_factor
        self.filter_kernel = filter_kernel
        self.demodulate = demodulate
        self.use_wscale = use_wscale
        self.wscale_gain = wscale_gain
        self.lr_mul = lr_mul
        self.noise_type = noise_type.lower()
        self.activation_type = activation_type
        self.conv_clamp = conv_clamp
        self.eps = eps

        self.space_of_latent = 'W'

        # Set up weight.
        weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
        fan_in = kernel_size * kernel_size * in_channels
        wscale = wscale_gain / np.sqrt(fan_in)
        if use_wscale:
            self.weight = nn.Parameter(torch.randn(*weight_shape) / lr_mul)
            self.wscale = wscale * lr_mul
        else:
            self.weight = nn.Parameter(
                torch.randn(*weight_shape) * wscale / lr_mul)
            self.wscale = lr_mul

        # Set up bias.
        if add_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
            self.bscale = lr_mul
        else:
            self.bias = None
        self.act_gain = bias_act.activation_funcs[activation_type].def_gain

        # Set up style.
        self.style = DenseLayer(in_channels=w_dim,
                                out_channels=in_channels,
                                add_bias=True,
                                init_bias=1.0,
                                use_wscale=use_wscale,
                                wscale_gain=wscale_gain,
                                lr_mul=lr_mul,
                                activation_type='linear')

        # Set up noise.
        if self.noise_type != 'none':
            self.noise_strength = nn.Parameter(torch.zeros(()))
            if self.noise_type == 'spatial':
                self.register_buffer(
                    'noise', torch.randn(1, 1, resolution, resolution))
            elif self.noise_type == 'channel':
                self.register_buffer(
                    'noise', torch.randn(1, out_channels, 1, 1))
            else:
                raise NotImplementedError(f'Not implemented noise type: '
                                          f'`{self.noise_type}`!')

        if scale_factor > 1:
            assert filter_kernel is not None
            self.register_buffer(
                'filter', upfirdn2d.setup_filter(filter_kernel))
            fh, fw = self.filter.shape
            self.filter_padding = (
                kernel_size // 2 + (fw + scale_factor - 1) // 2,
                kernel_size // 2 + (fw - scale_factor) // 2,
                kernel_size // 2 + (fh + scale_factor - 1) // 2,
                kernel_size // 2 + (fh - scale_factor) // 2)

    def extra_repr(self):
        return (f'in_ch={self.in_channels}, '
                f'out_ch={self.out_channels}, '
                f'ksize={self.kernel_size}, '
                f'wscale_gain={self.wscale_gain:.3f}, '
                f'bias={self.add_bias}, '
                f'lr_mul={self.lr_mul:.3f}, '
                f'upsample={self.scale_factor}, '
                f'upsample_filter={self.filter_kernel}, '
                f'demodulate={self.demodulate}, '
                f'noise_type={self.noise_type}, '
                f'act={self.activation_type}, '
                f'clamp={self.conv_clamp}')

    def forward_style(self, w, impl='cuda'):
        """Gets style code from the given input.

        More specifically, if the input is from W-Space, it will be projected by
        an affine transformation. If it is from the Style Space (Y-Space), no
        operation is required.

        NOTE: For codes from Y-Space, we use slicing to make sure the dimension
        is correct, in case that the code is padded before fed into this layer.
        """
        space_of_latent = self.space_of_latent.upper()
        if space_of_latent == 'W':
            if w.ndim != 2 or w.shape[1] != self.w_dim:
                raise ValueError(f'The input tensor should be with shape '
                                 f'[batch_size, w_dim], where '
                                 f'`w_dim` equals to {self.w_dim}!\n'
                                 f'But `{w.shape}` is received!')
            style = self.style(w, impl=impl)
        elif space_of_latent == 'Y':
            if w.ndim != 2 or w.shape[1] < self.in_channels:
                raise ValueError(f'The input tensor should be with shape '
                                 f'[batch_size, y_dim], where '
                                 f'`y_dim` equals to {self.in_channels}!\n'
                                 f'But `{w.shape}` is received!')
            style = w[:, :self.in_channels]
        else:
            raise NotImplementedError(f'Not implemented `space_of_latent`: '
                                      f'`{space_of_latent}`!')
        return style

    def forward(self,
                x,
                w,
                runtime_gain=1.0,
                noise_mode='const',
                fused_modulate=False,
                impl='cuda'):
        dtype = x.dtype
        N, C, H, W = x.shape

        fused_modulate = (fused_modulate and
                          not self.training and
                          (dtype == torch.float32 or N == 1))

        weight = self.weight
        out_ch, in_ch, kh, kw = weight.shape
        assert in_ch == C

        # Affine on `w`.
        style = self.forward_style(w, impl=impl)
        if not self.demodulate:
            _style = style * self.wscale  # Equivalent to scaling weight.
        else:
            _style = style

        # Prepare noise.
        noise = None
        noise_mode = noise_mode.lower()
        if self.noise_type != 'none' and noise_mode != 'none':
            if noise_mode == 'random':
                noise = torch.randn(
                    (N, *self.noise.shape[1:]), device=x.device)
            elif noise_mode == 'const':
                noise = self.noise
            else:
                raise ValueError(f'Unknown noise mode `{noise_mode}`!')
            noise = (noise * self.noise_strength).to(dtype)

        # Pre-normalize inputs to avoid FP16 overflow.
        if dtype == torch.float16 and self.demodulate:
            weight_max = weight.norm(float('inf'), dim=(1, 2, 3), keepdim=True)
            weight = weight * (self.wscale / weight_max)
            style_max = _style.norm(float('inf'), dim=1, keepdim=True)
            _style = _style / style_max

        if self.demodulate or fused_modulate:
            _weight = weight.unsqueeze(0)
            _weight = _weight * _style.reshape(N, 1, in_ch, 1, 1)
        if self.demodulate:
            decoef = (_weight.square().sum(dim=(2, 3, 4)) + self.eps).rsqrt()
        if self.demodulate and fused_modulate:
            _weight = _weight * decoef.reshape(N, out_ch, 1, 1, 1)

        if not fused_modulate:
            x = x * _style.to(dtype).reshape(N, in_ch, 1, 1)
            w = weight.to(dtype)
            groups = 1
        else:  # Use group convolution to fuse style modulation and convolution.
            x = x.reshape(1, N * in_ch, H, W)
            w = _weight.reshape(N * out_ch, in_ch, kh, kw).to(dtype)
            groups = N

        if self.scale_factor == 1:  # Native convolution without upsampling.
            up = 1
            padding = self.kernel_size // 2
            x = conv2d_gradfix.conv2d(
                x, w, stride=1, padding=padding, groups=groups, impl=impl)
        else:  # Convolution with upsampling.
            up = self.scale_factor
            f = self.filter
            # When kernel size = 1, use filtering function for upsampling.
            if self.kernel_size == 1:
                padding = self.filter_padding
                x = conv2d_gradfix.conv2d(
                    x, w, stride=1, padding=0, groups=groups, impl=impl)
                x = upfirdn2d.upfirdn2d(
                    x, f, up=up, padding=padding, gain=up ** 2, impl=impl)
            # When kernel size != 1, use stride convolution for upsampling.
            else:
                # Following codes are borrowed from
                # https://github.com/NVlabs/stylegan2-ada-pytorch
                px0, px1, py0, py1 = self.filter_padding
                px0 = px0 - (kw - 1)
                px1 = px1 - (kw - up)
                py0 = py0 - (kh - 1)
                py1 = py1 - (kh - up)
                pxt = max(min(-px0, -px1), 0)
                pyt = max(min(-py0, -py1), 0)
                if groups == 1:
                    w = w.transpose(0, 1)
                else:
                    w = w.reshape(N, out_ch, in_ch, kh, kw)
                    w = w.transpose(1, 2)
                    w = w.reshape(N * in_ch, out_ch, kh, kw)
                padding = (pyt, pxt)
                x = conv2d_gradfix.conv_transpose2d(
                    x, w, stride=up, padding=padding, groups=groups, impl=impl)
                padding = (px0 + pxt, px1 + pxt, py0 + pyt, py1 + pyt)
                x = upfirdn2d.upfirdn2d(
                    x, f, up=1, padding=padding, gain=up ** 2, impl=impl)

        if not fused_modulate:
            if self.demodulate:
                decoef = decoef.to(dtype).reshape(N, out_ch, 1, 1)
            if self.demodulate and noise is not None:
                x = fma.fma(x, decoef, noise, impl=impl)
            else:
                if self.demodulate:
                    x = x * decoef
                if noise is not None:
                    x = x + noise
        else:
            x = x.reshape(N, out_ch, H * up, W * up)
            if noise is not None:
                x = x + noise

        bias = None
        if self.bias is not None:
            bias = self.bias.to(dtype)
            if self.bscale != 1.0:
                bias = bias * self.bscale

        if self.activation_type == 'linear':  # Shortcut for output layer.
            x = bias_act.bias_act(
                x, bias, act='linear', clamp=self.conv_clamp, impl=impl)
        else:
            act_gain = self.act_gain * runtime_gain
            act_clamp = None
            if self.conv_clamp is not None:
                act_clamp = self.conv_clamp * runtime_gain
            x = bias_act.bias_act(x, bias,
                                  act=self.activation_type,
                                  gain=act_gain,
                                  clamp=act_clamp,
                                  impl=impl)

        assert x.dtype == dtype
        assert style.dtype == torch.float32
        return x, style


class DenseLayer(nn.Module):
    """Implements the dense layer."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 add_bias,
                 init_bias,
                 use_wscale,
                 wscale_gain,
                 lr_mul,
                 activation_type):
        """Initializes with layer settings.

        Args:
            in_channels: Number of channels of the input tensor.
            out_channels: Number of channels of the output tensor.
            add_bias: Whether to add bias onto the fully-connected result.
            init_bias: The initial bias value before training.
            use_wscale: Whether to use weight scaling.
            wscale_gain: Gain factor for weight scaling.
            lr_mul: Learning multiplier for both weight and bias.
            activation_type: Type of activation.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_bias = add_bias
        self.init_bias = init_bias
        self.use_wscale = use_wscale
        self.wscale_gain = wscale_gain
        self.lr_mul = lr_mul
        self.activation_type = activation_type

        weight_shape = (out_channels, in_channels)
        wscale = wscale_gain / np.sqrt(in_channels)
        if use_wscale:
            self.weight = nn.Parameter(torch.randn(*weight_shape) / lr_mul)
            self.wscale = wscale * lr_mul
        else:
            self.weight = nn.Parameter(
                torch.randn(*weight_shape) * wscale / lr_mul)
            self.wscale = lr_mul

        if add_bias:
            init_bias = np.float32(init_bias) / lr_mul
            self.bias = nn.Parameter(torch.full([out_channels], init_bias))
            self.bscale = lr_mul
        else:
            self.bias = None

    def extra_repr(self):
        return (f'in_ch={self.in_channels}, '
                f'out_ch={self.out_channels}, '
                f'wscale_gain={self.wscale_gain:.3f}, '
                f'bias={self.add_bias}, '
                f'init_bias={self.init_bias}, '
                f'lr_mul={self.lr_mul:.3f}, '
                f'act={self.activation_type}')

    def forward(self, x, impl='cuda'):
        dtype = x.dtype

        if x.ndim != 2:
            x = x.flatten(start_dim=1)

        weight = self.weight.to(dtype) * self.wscale
        bias = None
        if self.bias is not None:
            bias = self.bias.to(dtype)
            if self.bscale != 1.0:
                bias = bias * self.bscale

        # Fast pass for linear activation.
        if self.activation_type == 'linear' and bias is not None:
            x = torch.addmm(bias.unsqueeze(0), x, weight.t())
        else:
            x = x.matmul(weight.t())
            x = bias_act.bias_act(x, bias, act=self.activation_type, impl=impl)

        assert x.dtype == dtype
        return x

# pylint: enable=missing-function-docstring

class InstanceNormLayer(nn.Module):
    """Implements instance normalization layer."""
    def __init__(self, num_features, epsilon=1e-8, affine=False):
        super().__init__()
        self.eps = epsilon
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(1, num_features,1,1,1))
            self.bias = nn.Parameter(torch.Tensor(1, num_features,1,1,1))
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def forward(self, x, weight=None, bias=None):
        x = x - torch.mean(x, dim=[2, 3, 4], keepdim=True)
        norm = torch.sqrt(
            torch.mean(x**2, dim=[2, 3, 4], keepdim=True) + self.eps)
        x = x / norm
        isnot_input_none = weight is not None and bias is not None
        assert (isnot_input_none and not self.affine) or (not isnot_input_none and self.affine)
        if self.affine:
            x = x*self.weight + self.bias
        else:
            x = x*weight + bias
        return x


class UpsamplingLayer(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        if self.scale_factor <= 1:
            return x
        return F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
