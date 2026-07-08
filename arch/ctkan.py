import math

import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

from arch.kan import KANLinear
from arch.common import KANBlock, ConvLayer, D_ConvLayer, PatchEmbed, DW_bn_relu


def _build_sync_pairs(num_units, num_pairs, nself):
    """Sample activity pairs used by the synchronization feedback branch."""
    num_units = int(num_units)
    num_pairs = max(1, int(num_pairs))
    nself = max(0, int(nself))

    num_self = min(nself, num_pairs, num_units)
    pair_i = []
    pair_j = []
    if num_self > 0:
        self_idx = torch.randperm(num_units)[:num_self]
        pair_i.append(self_idx)
        pair_j.append(self_idx)

    remaining = num_pairs - num_self
    if remaining > 0:
        pair_i.append(torch.randint(0, num_units, (remaining,)))
        pair_j.append(torch.randint(0, num_units, (remaining,)))

    return torch.cat(pair_i).long(), torch.cat(pair_j).long()


class CTKANSplineKANLinear(nn.Module):
    """
    KAN linear layer with continuous-thought spline activity modulation.

    A standard KAN spline edge contributes:
        sum_k spline_weight[o, i, k] * B_k(x_i)

    CTKAN builds an internal activity state z^t[b, o, i, k] for each spline
    coefficient and generates a dynamic gate:
        spline_weight[o, i, k] * gate[b, o, i, k] * B_k(x_i)

    Thus the recurrent state lives inside the spline coefficient space rather
    than as an external token-level controller.
    """

    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        no_kan=False,
        ctm_ticks=20,
        ctm_dhidden=64,
        ctm_dropout=0.2,
        ctm_daction=1024,
        ctm_dout=1024,
        ctm_nself=32,
        ctm_memory=10,
        ctm_scale_init=1e-3,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.no_kan = bool(no_kan)
        self.ticks = int(ctm_ticks)
        self.memory_length = max(1, int(ctm_memory))

        if self.ticks <= 0:
            raise ValueError("ctm_ticks must be positive")

        if self.no_kan:
            self.linear = nn.Linear(self.in_features, self.out_features)
            self.num_coeff = 1
            self.dmodel = self.out_features * self.in_features
            return

        self.static_edge = KANLinear(
            self.in_features,
            self.out_features,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range,
        )

        self.num_coeff = int(grid_size + spline_order)
        self.dmodel = self.out_features * self.in_features * self.num_coeff

        action_pair_i, action_pair_j = _build_sync_pairs(
            self.dmodel,
            ctm_daction,
            ctm_nself,
        )
        output_pair_i, output_pair_j = _build_sync_pairs(
            self.dmodel,
            ctm_dout,
            ctm_nself,
        )
        self.register_buffer("action_pair_i", action_pair_i)
        self.register_buffer("action_pair_j", action_pair_j)
        self.register_buffer("output_pair_i", output_pair_i)
        self.register_buffer("output_pair_j", output_pair_j)

        hidden = max(1, int(ctm_dhidden))
        action_dim = int(action_pair_i.numel())
        output_dim = int(output_pair_i.numel())
        factor_dim = self.out_features + self.in_features + self.num_coeff

        self.action_to_factors = nn.Sequential(
            nn.Linear(action_dim, hidden),
            nn.GELU(),
            nn.Dropout(ctm_dropout),
            nn.Linear(hidden, factor_dim),
        )
        self.output_to_factors = nn.Sequential(
            nn.Linear(output_dim, hidden),
            nn.GELU(),
            nn.Dropout(ctm_dropout),
            nn.Linear(hidden, factor_dim),
        )

        self.z_init = nn.Parameter(
            torch.zeros(1, self.out_features, self.in_features, self.num_coeff)
        )
        self.input_gain = nn.Parameter(torch.tensor(1.0))
        self.state_gain = nn.Parameter(torch.tensor(0.5))
        self.output_scale = nn.Parameter(torch.tensor(float(ctm_scale_init)))

        memory_decay = math.exp(-1.0 / float(self.memory_length))
        self.register_buffer("memory_decay", torch.tensor(memory_decay, dtype=torch.float32))

        self._init_ctm_weights()

    def _init_ctm_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.zeros_(self.action_to_factors[-1].weight)
        nn.init.zeros_(self.action_to_factors[-1].bias)
        nn.init.zeros_(self.output_to_factors[-1].weight)
        nn.init.zeros_(self.output_to_factors[-1].bias)

    def _sample_sync(self, z, pair_i, pair_j):
        z_flat = z.flatten(1)
        return z_flat[:, pair_i] * z_flat[:, pair_j]

    def _factors_to_activity(self, factors):
        out_factor, in_factor, coeff_factor = torch.split(
            factors,
            [self.out_features, self.in_features, self.num_coeff],
            dim=-1,
        )
        return (
            out_factor[:, :, None, None]
            + in_factor[:, None, :, None]
            + coeff_factor[:, None, None, :]
        )

    def _run_ctm_ticks(self, spline_activity):
        batch = spline_activity.shape[0]
        z = self.z_init.to(dtype=spline_activity.dtype, device=spline_activity.device)
        z = z.expand(batch, -1, -1, -1)
        memory = z
        decay = self.memory_decay.to(dtype=spline_activity.dtype, device=spline_activity.device)

        for _ in range(self.ticks):
            action_sync = self._sample_sync(memory, self.action_pair_i, self.action_pair_j)
            action_feedback = self._factors_to_activity(self.action_to_factors(action_sync))
            pre_acts = self.input_gain * spline_activity + self.state_gain * memory + action_feedback
            z = torch.tanh(pre_acts)
            memory = decay * memory + (1.0 - decay) * z

        output_sync = self._sample_sync(memory, self.output_pair_i, self.output_pair_j)
        output_feedback = self._factors_to_activity(self.output_to_factors(output_sync))

        gate = 1.0 + self.output_scale * torch.tanh(memory + output_feedback)
        return gate

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError("CTKANSplineKANLinear expects [B, N, C] input")

        batch, num_tokens, channels = x.shape
        if channels != self.in_features:
            raise ValueError("input channel mismatch in CTKANSplineKANLinear")

        if self.no_kan:
            return self.linear(x)

        flat_x = x.reshape(batch * num_tokens, channels)
        basis = self.static_edge.b_splines(flat_x)
        basis = basis.view(batch, num_tokens, self.in_features, self.num_coeff)

        spline_weight = self.static_edge.scaled_spline_weight
        basis_mean = basis.mean(dim=1)
        spline_activity = basis_mean.unsqueeze(1) * spline_weight.unsqueeze(0)
        gate = self._run_ctm_ticks(spline_activity)

        base_output = F.linear(
            self.static_edge.base_activation(flat_x),
            self.static_edge.base_weight,
        )
        base_output = base_output.view(batch, num_tokens, self.out_features)

        gated_weight = spline_weight.unsqueeze(0) * gate
        spline_output = torch.einsum("bnik,boik->bno", basis, gated_weight)
        return base_output + spline_output


class CTKANSplineKANLayer(nn.Module):
    """KAN layer whose three spline linear layers use CTKAN modulation."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.,
        no_kan=False,
        **ctm_kwargs,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        kan_kwargs = dict(
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
            no_kan=no_kan,
            **ctm_kwargs,
        )

        self.fc1 = CTKANSplineKANLinear(in_features, hidden_features, **kan_kwargs)
        self.fc2 = CTKANSplineKANLinear(hidden_features, out_features, **kan_kwargs)
        self.fc3 = CTKANSplineKANLinear(hidden_features, out_features, **kan_kwargs)

        self.dwconv_1 = DW_bn_relu(hidden_features)
        self.dwconv_2 = DW_bn_relu(hidden_features)
        self.dwconv_3 = DW_bn_relu(hidden_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv_1(x, H, W)
        x = self.fc2(x)
        x = self.dwconv_2(x, H, W)
        x = self.fc3(x)
        x = self.dwconv_3(x, H, W)
        return x


class CTKANBlock(nn.Module):
    """Residual token block using CTKAN spline modulation."""

    def __init__(
        self,
        dim,
        drop=0.,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        no_kan=False,
        **ctm_kwargs,
    ):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm = norm_layer(dim)
        self.layer = CTKANSplineKANLayer(
            in_features=dim,
            hidden_features=dim,
            out_features=dim,
            drop=drop,
            no_kan=no_kan,
            **ctm_kwargs,
        )

    def forward(self, x, H, W):
        return x + self.drop_path(self.layer(self.norm(x), H, W))


class CTKAN(nn.Module):
    """
    CTKAN segmentation network.

    The model keeps the UKAN encoder-decoder layout and replaces only the
    bottleneck encoder token block and the deepest decoder token block with
    CTKAN spline-modulated KAN blocks.
    """

    def __init__(
        self,
        num_classes,
        input_channels=3,
        deep_supervision=False,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dims=[256, 320, 512],
        no_kan=False,
        drop_rate=0.,
        drop_path_rate=0.,
        norm_layer=nn.LayerNorm,
        depths=[1, 1, 1],
        ctm_ticks=20,
        ctm_dmodel=0,
        ctm_dinput=0,
        ctm_dhidden=64,
        ctm_synapse_depth=8,
        ctm_dropout=0.2,
        ctm_heads=8,
        ctm_jaction=32,
        ctm_jout=32,
        ctm_daction=1024,
        ctm_dout=1024,
        ctm_nself=32,
        ctm_memory=10,
        ctm_pairing="random",
        ctm_scale_init=1e-3,
        **kwargs,
    ):
        super().__init__()
        del ctm_dmodel, ctm_dinput, ctm_synapse_depth, ctm_heads, ctm_jaction, ctm_jout, ctm_pairing

        kan_input_dim = embed_dims[0]

        self.encoder1 = ConvLayer(input_channels, kan_input_dim // 8)
        self.encoder2 = ConvLayer(kan_input_dim // 8, kan_input_dim // 4)
        self.encoder3 = ConvLayer(kan_input_dim // 4, kan_input_dim)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])
        self.dnorm3 = norm_layer(embed_dims[1])
        self.dnorm4 = norm_layer(embed_dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        ctm_kwargs = dict(
            ctm_ticks=ctm_ticks,
            ctm_dhidden=ctm_dhidden,
            ctm_dropout=ctm_dropout,
            ctm_daction=ctm_daction,
            ctm_dout=ctm_dout,
            ctm_nself=ctm_nself,
            ctm_memory=ctm_memory,
            ctm_scale_init=ctm_scale_init,
        )

        self.block1 = nn.ModuleList([KANBlock(
            dim=embed_dims[1],
            drop=drop_rate,
            drop_path=dpr[0],
            norm_layer=norm_layer,
            no_kan=no_kan,
        )])
        self.block2 = nn.ModuleList([CTKANBlock(
            dim=embed_dims[2],
            drop=drop_rate,
            drop_path=dpr[1],
            norm_layer=norm_layer,
            no_kan=no_kan,
            **ctm_kwargs,
        )])
        self.dblock1 = nn.ModuleList([CTKANBlock(
            dim=embed_dims[1],
            drop=drop_rate,
            drop_path=dpr[0],
            norm_layer=norm_layer,
            no_kan=no_kan,
            **ctm_kwargs,
        )])
        self.dblock2 = nn.ModuleList([KANBlock(
            dim=embed_dims[0],
            drop=drop_rate,
            drop_path=dpr[1],
            norm_layer=norm_layer,
            no_kan=no_kan,
        )])

        self.patch_embed3 = PatchEmbed(img_size=img_size // 4, patch_size=3, stride=2,
                                       in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed4 = PatchEmbed(img_size=img_size // 8, patch_size=3, stride=2,
                                       in_chans=embed_dims[1], embed_dim=embed_dims[2])

        self.decoder1 = D_ConvLayer(embed_dims[2], embed_dims[1])
        self.decoder2 = D_ConvLayer(embed_dims[1], embed_dims[0])
        self.decoder3 = D_ConvLayer(embed_dims[0], embed_dims[0] // 4)
        self.decoder4 = D_ConvLayer(embed_dims[0] // 4, embed_dims[0] // 8)
        self.decoder5 = D_ConvLayer(embed_dims[0] // 8, embed_dims[0] // 8)

        self.final = nn.Conv2d(embed_dims[0] // 8, num_classes, kernel_size=1)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        batch = x.shape[0]

        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2))
        t1 = out
        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2))
        t2 = out
        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2))
        t3 = out

        out, H, W = self.patch_embed3(out)
        for block in self.block1:
            out = block(out, H, W)
        out = self.norm3(out)
        out = out.reshape(batch, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        out, H, W = self.patch_embed4(out)
        for block in self.block2:
            out = block(out, H, W)
        out = self.norm4(out)
        out = out.reshape(batch, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t4)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)

        for block in self.dblock1:
            out = block(out, H, W)
        out = self.dnorm3(out)
        out = out.reshape(batch, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t3)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for block in self.dblock2:
            out = block(out, H, W)
        out = self.dnorm4(out)
        out = out.reshape(batch, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))

        return self.final(out)

