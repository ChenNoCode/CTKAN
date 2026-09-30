import math

import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

from kan import KANLinear
from arch.common import KANBlock, ConvLayer, D_ConvLayer, PatchEmbed, DW_bn_relu


def _build_sync_pairs(num_units, num_pairs, nself):
    """
    从 z^t 的 D 个样条活动里抽样同步对。

    num_units = D = O * I * K
    每个 pair 表示两个样条活动之间的同步关系。
    """
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


class CtmSplineKANLinear(nn.Module):
    """
    真正嵌入 KAN 样条内部的 CTKAN Linear。

    原始 KANLinear 的一条样条边是：
        spline[o, i] = sum_k spline_weight[o, i, k] * B_k(x_i)

    这里把 CTM 的神经活动 z^t 对齐到每一个样条系数：
        z^t[b, o, i, k]

    含义：
        第 b 张图像，
        第 o 个输出神经元，
        第 i 个输入神经元，
        第 k 个样条基函数 / 样条系数，
        在第 t 个 tick 的动态活动。

    最后不是修改静态参数 spline_weight，而是生成动态 gate：
        spline_weight[o, i, k] * gate^T[b, o, i, k] * B_k(x_i)

    这样 z^t 就不再是外置控制器状态，而是 KAN 样条函数内部的活动状态。
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
        CtmTicks=20,
        CtmDhidden=64,
        CtmDropout=0.2,
        CtmDaction=1024,
        CtmDout=1024,
        CtmNself=32,
        CtmMemory=10,
        CtmScaleInit=5e-2,
        CtmTerminalReadout=True,
        CtmPriorMode="full",
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.no_kan = bool(no_kan)
        self.ticks = int(CtmTicks)
        self.memory_length = max(1, int(CtmMemory))
        self.terminal_readout = bool(CtmTerminalReadout)
        self.prior_mode = str(CtmPriorMode).lower()
        self.trace_enabled = False
        self.last_trace = None

        if self.ticks <= 0:
            raise ValueError("CtmTicks must be positive")
        if self.prior_mode not in {"full", "factorized", "none"}:
            raise ValueError(
                "CtmPriorMode must be one of: full, factorized, none"
            )

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
            CtmDaction,
            CtmNself,
        )
        if self.terminal_readout:
            output_pair_i, output_pair_j = _build_sync_pairs(
                self.dmodel,
                CtmDout,
                CtmNself,
            )
        else:
            output_pair_i = torch.empty(0, dtype=torch.long)
            output_pair_j = torch.empty(0, dtype=torch.long)
        self.register_buffer("action_pair_i", action_pair_i)
        self.register_buffer("action_pair_j", action_pair_j)
        self.register_buffer("output_pair_i", output_pair_i)
        self.register_buffer("output_pair_j", output_pair_j)
        self.register_buffer(
            "action_pair_indices",
            torch.cat((action_pair_i, action_pair_j)),
            persistent=False,
        )
        self.register_buffer(
            "output_pair_indices",
            torch.cat((output_pair_i, output_pair_j)),
            persistent=False,
        )

        hidden = max(1, int(CtmDhidden))
        action_dim = int(action_pair_i.numel())
        output_dim = int(output_pair_i.numel())
        factor_dim = self.out_features + self.in_features + self.num_coeff

        self.action_to_factors = nn.Sequential(
            nn.Linear(action_dim, hidden),
            nn.GELU(),
            nn.Dropout(CtmDropout),
            nn.Linear(hidden, factor_dim),
        )
        if self.terminal_readout:
            self.output_to_factors = nn.Sequential(
                nn.Linear(output_dim, hidden),
                nn.GELU(),
                nn.Dropout(CtmDropout),
                nn.Linear(hidden, factor_dim),
            )
        else:
            self.output_to_factors = None

        # The full prior preserves old checkpoints. Efficient variants can use
        # an axis-factorized O+I+K prior or remove the weak prior entirely.
        if self.prior_mode == "full":
            self.z_init = nn.Parameter(
                torch.zeros(
                    1,
                    self.out_features,
                    self.in_features,
                    self.num_coeff,
                )
            )
        elif self.prior_mode == "factorized":
            self.z_init_out = nn.Parameter(
                torch.zeros(1, self.out_features, 1, 1)
            )
            self.z_init_in = nn.Parameter(
                torch.zeros(1, 1, self.in_features, 1)
            )
            self.z_init_basis = nn.Parameter(
                torch.zeros(1, 1, 1, self.num_coeff)
            )

        # 这几个标量让 CTM 更新一开始稳定，后续训练自己决定外部输入和历史状态的比例。
        self.input_gain = nn.Parameter(torch.tensor(1.0))
        self.state_gain = nn.Parameter(torch.tensor(0.5))
        self.output_scale = nn.Parameter(torch.tensor(float(CtmScaleInit)))

        # 原文 CTM 有 FIFO memory。对 D=O*I*K 的样条活动完整保存 M 槽会很占显存，
        # 这里用 EMA 近似这段历史：CtmMemory 越大，历史衰减越慢。
        memory_decay = math.exp(-1.0 / float(self.memory_length))
        self.register_buffer("memory_decay", torch.tensor(memory_decay, dtype=torch.float32))

        self._init_ctm_weights()

    def _init_ctm_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # 最后一层置零：初始时 CTM 只由样条活动本身驱动，避免随机同步反馈扰乱 KAN。
        nn.init.zeros_(self.action_to_factors[-1].weight)
        nn.init.zeros_(self.action_to_factors[-1].bias)
        if self.output_to_factors is not None:
            nn.init.zeros_(self.output_to_factors[-1].weight)
            nn.init.zeros_(self.output_to_factors[-1].bias)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if not self.no_kan:
            self.action_pair_indices = torch.cat(
                (self.action_pair_i, self.action_pair_j)
            )
            self.output_pair_indices = torch.cat(
                (self.output_pair_i, self.output_pair_j)
            )

    def _initial_prior(self, reference):
        if self.prior_mode == "none":
            return reference.new_zeros(
                1,
                self.out_features,
                self.in_features,
                self.num_coeff,
            )
        if self.prior_mode == "factorized":
            prior = self.z_init_out + self.z_init_in + self.z_init_basis
        else:
            prior = self.z_init
        return torch.tanh(
            prior.to(dtype=reference.dtype, device=reference.device)
        )

    def _sample_sync(self, z, pair_i, pair_j):
        """
        z: [B, O, I, K]
        return: [B, P]

        这里计算的是样条活动之间的同步：
            sync_p = z[pair_i] * z[pair_j]
        """
        z_flat = z.flatten(1)
        return z_flat[:, pair_i] * z_flat[:, pair_j]

    def enable_trace(self, enabled=True):
        """Collect compact inference statistics for mechanism visualization."""
        self.trace_enabled = bool(enabled)
        self.last_trace = None

    def _factors_to_activity(self, factors):
        """
        把同步表征低秩投影回 [B, O, I, K]。

        直接用 Linear(P, O*I*K) 参数量太大，所以拆成：
            output factor + input factor + coeff factor
        再广播相加，得到每个样条活动的反馈。
        """
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
        """
        spline_activity: [B, O, I, K]

        这是 CTM 的 T 步内部思考。
        z^t 的每个元素都对应一个样条活动，而不是外部 token gate。
        """
        batch = spline_activity.shape[0]
        z = self._initial_prior(spline_activity).expand(batch, -1, -1, -1)
        memory = z
        decay = self.memory_decay.to(dtype=spline_activity.dtype, device=spline_activity.device)

        if self.trace_enabled:
            update_ratios = []
            memory_rms = [memory.detach().float().square().mean(dim=(1, 2, 3)).sqrt()]

        for _ in range(self.ticks):
            action_sync = self._sample_sync(memory, self.action_pair_i, self.action_pair_j)
            action_feedback = self._factors_to_activity(self.action_to_factors(action_sync))

            # 输入项来自当前样本真实激活到的样条基函数，历史项来自上一轮样条活动。
            pre_acts = self.input_gain * spline_activity + self.state_gain * memory + action_feedback
            z = torch.tanh(pre_acts)
            next_memory = decay * memory + (1.0 - decay) * z

            if self.trace_enabled:
                delta_rms = (next_memory - memory).detach().float().square().mean(
                    dim=(1, 2, 3)
                ).sqrt()
                next_rms = next_memory.detach().float().square().mean(
                    dim=(1, 2, 3)
                ).sqrt()
                update_ratios.append(delta_rms / next_rms.clamp_min(1e-12))
                memory_rms.append(next_rms)

            memory = next_memory

        if self.output_to_factors is None:
            output_feedback = 0.0
        else:
            output_sync = self._sample_sync(
                memory,
                self.output_pair_i,
                self.output_pair_j,
            )
            output_feedback = self._factors_to_activity(
                self.output_to_factors(output_sync)
            )

        # gate 初始接近 1；训练后可以增强或减弱每个样条系数的贡献。
        gate = 1.0 + self.output_scale * torch.tanh(memory + output_feedback)

        if self.trace_enabled:
            with torch.no_grad():
                before = spline_activity.detach().float().abs().mean(dim=(1, 2))
                after = (spline_activity * gate).detach().float().abs().mean(dim=(1, 2))
                self.last_trace = {
                    "update_ratio": torch.stack(update_ratios, dim=1).cpu(),
                    "memory_rms": torch.stack(memory_rms, dim=1).cpu(),
                    "gate_by_basis": gate.detach().float().mean(dim=(1, 2)).cpu(),
                    "gate_strength_by_basis": (
                        gate.detach().float() - 1.0
                    ).abs().mean(dim=(1, 2)).cpu(),
                    "contribution_before": before.cpu(),
                    "contribution_after": after.cpu(),
                    "output_scale": self.output_scale.detach().float().cpu(),
                }

        return gate

    def forward(self, x):
        """
        x: [B, N, I]
        return: [B, N, O]
        """
        self.last_trace = None

        if x.dim() != 3:
            raise ValueError("CtmSplineKANLinear expects [B, N, C] input")

        batch, num_tokens, channels = x.shape
        if channels != self.in_features:
            raise ValueError("input channel mismatch in CtmSplineKANLinear")

        if self.no_kan:
            return self.linear(x)

        flat_x = x.reshape(batch * num_tokens, channels)
        basis = self.static_edge.b_splines(flat_x)
        basis = basis.view(batch, num_tokens, self.in_features, self.num_coeff)

        spline_weight = self.static_edge.scaled_spline_weight

        # 每个样本先统计“哪些样条基函数被当前 token 激活了”。
        # 得到 [B, O, I, K]，它就是 z^t 的外部输入活动。
        basis_mean = basis.mean(dim=1)
        spline_activity = basis_mean.unsqueeze(1) * spline_weight.unsqueeze(0)

        gate = self._run_ctm_ticks(spline_activity)

        base_output = F.linear(
            self.static_edge.base_activation(flat_x),
            self.static_edge.base_weight,
        )
        base_output = base_output.view(batch, num_tokens, self.out_features)

        # 动态样条输出：
        #   sum_{i,k} B_k(x_i) * spline_weight[o,i,k] * gate[b,o,i,k]
        gated_weight = spline_weight.unsqueeze(0) * gate
        spline_output = torch.einsum("bnik,boik->bno", basis, gated_weight)
        return base_output + spline_output


class CtmSplineKANLayer(nn.Module):
    """
    用 CtmSplineKANLinear 替换 KANLayer 里的三个 KANLinear。

    也就是说，CTM 被嵌入到 KAN 的样条边内部：
        fc1 / fc2 / fc3 的每个 spline_weight[o,i,k]
        都有对应的 z^t[o,i,k] 样条活动。
    """

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

        self.fc1 = CtmSplineKANLinear(in_features, hidden_features, **kan_kwargs)
        self.fc2 = CtmSplineKANLinear(hidden_features, out_features, **kan_kwargs)
        self.fc3 = CtmSplineKANLinear(hidden_features, out_features, **kan_kwargs)

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


class CtmKANBlock(nn.Module):
    """
    Residual token block with CTKAN dynamics inside each KAN linear layer.
    """

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
        self.layer = CtmSplineKANLayer(
            in_features=dim,
            hidden_features=dim,
            out_features=dim,
            drop=drop,
            no_kan=no_kan,
            **ctm_kwargs,
        )

    def forward(self, x, H, W):
        return x + self.drop_path(self.layer(self.norm(x), H, W))


class _CTKANBackbone(nn.Module):
    """
    Shared U-shaped backbone with two deep CTKAN token blocks.

    结构位置：
        encoder block1: 普通 KAN，保持浅层 token 稳定和省显存。
        encoder block2: CtmKANBlock，位于 bottleneck token 处。
        decoder dblock1: CtmKANBlock，位于 decoder 最深 token 处。
        decoder dblock2: 普通 KAN，负责回到较高分辨率细节。

    样条级 z^t：
        对每个 CtmSplineKANLinear，
        D = O * I * K，
        z^t 可以看成 [B, D]，
        也可以还原成 [B, O, I, K]。
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
        CtmTicks=20,
        CtmDhidden=64,
        CtmDropout=0.2,
        CtmDaction=1024,
        CtmDout=1024,
        CtmNself=32,
        CtmMemory=10,
        CtmScaleInit=5e-2,
        **kwargs,
    ):
        super().__init__()

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
            CtmTicks=CtmTicks,
            CtmDhidden=CtmDhidden,
            CtmDropout=CtmDropout,
            CtmDaction=CtmDaction,
            CtmDout=CtmDout,
            CtmNself=CtmNself,
            CtmMemory=CtmMemory,
            CtmScaleInit=CtmScaleInit,
        )

        self.block1 = nn.ModuleList([KANBlock(
            dim=embed_dims[1],
            drop=drop_rate,
            drop_path=dpr[0],
            norm_layer=norm_layer,
            no_kan=no_kan,
        )])
        self.block2 = nn.ModuleList([CtmKANBlock(
            dim=embed_dims[2],
            drop=drop_rate,
            drop_path=dpr[1],
            norm_layer=norm_layer,
            no_kan=no_kan,
            **ctm_kwargs,
        )])
        self.dblock1 = nn.ModuleList([CtmKANBlock(
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

        # 第一次 CTKAN：encoder bottleneck token，样条级 z^t 位于 block2 的 KANLinear 内部。
        out, H, W = self.patch_embed4(out)
        for block in self.block2:
            out = block(out, H, W)
        out = self.norm4(out)
        out = out.reshape(batch, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t4)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)

        # 第二次 CTKAN：decoder 最深 token，样条级 z^t 位于 dblock1 的 KANLinear 内部。
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


class AdaptiveCtmSplineKANLinear(CtmSplineKANLinear):
    """Spline-level CTKAN linear with input-dominant, scale-stable dynamics."""

    def __init__(
        self,
        *args,
        CtmActivityStd=0.5,
        CtmPriorScale=0.05,
        CtmActivityEps=1e-6,
        CtmSyncEps=1e-4,
        **kwargs,
    ):
        self.activity_std_weight = float(CtmActivityStd)
        self.prior_scale = float(CtmPriorScale)
        self.activity_eps = float(CtmActivityEps)
        self.sync_eps = float(CtmSyncEps)
        super().__init__(*args, **kwargs)

        if not self.no_kan:
            # Small nonzero readouts let synchronization branches receive a
            # gradient immediately without perturbing the near-identity gate.
            projections = [self.action_to_factors]
            if self.output_to_factors is not None:
                projections.append(self.output_to_factors)
            for projection in projections:
                trunc_normal_(projection[-1].weight, std=1e-3)
                nn.init.zeros_(projection[-1].bias)

    def _sample_sync(self, z, pair_indices):
        z_flat = z.flatten(1)
        pair_count = pair_indices.numel() // 2
        paired = torch.index_select(z_flat, 1, pair_indices).view(
            z_flat.shape[0], 2, pair_count
        )
        synchronization = paired[:, 0].mul(paired[:, 1])
        return F.layer_norm(
            synchronization,
            (pair_count,),
            eps=self.sync_eps ** 2,
        )

    def _run_ctm_ticks(self, spline_activity):
        batch = spline_activity.shape[0]
        prior = self._initial_prior(spline_activity).expand(
            batch, -1, -1, -1
        )
        input_drive = self.input_gain * spline_activity
        memory = torch.tanh(input_drive + self.prior_scale * prior)
        decay = self.memory_decay.to(
            dtype=spline_activity.dtype,
            device=spline_activity.device,
        )

        if self.trace_enabled:
            update_ratios = []
            memory_rms = [memory.detach().float().square().mean(dim=(1, 2, 3)).sqrt()]
            memory_rms_by_basis = [
                memory.detach().float().square().mean(dim=(1, 2)).sqrt()
            ]

        update_rate = 1.0 - decay

        for _ in range(self.ticks):
            action_sync = self._sample_sync(
                memory,
                self.action_pair_indices,
            )
            action_feedback = self._factors_to_activity(self.action_to_factors(action_sync))
            pre_acts = (
                input_drive
                + self.state_gain * memory
                + action_feedback
            )
            z = torch.tanh(pre_acts)
            next_memory = torch.lerp(memory, z, update_rate)

            if self.trace_enabled:
                delta_rms = (next_memory - memory).detach().float().square().mean(
                    dim=(1, 2, 3)
                ).sqrt()
                next_rms = next_memory.detach().float().square().mean(
                    dim=(1, 2, 3)
                ).sqrt()
                update_ratios.append(delta_rms / next_rms.clamp_min(1e-12))
                memory_rms.append(next_rms)
                memory_rms_by_basis.append(
                    next_memory.detach().float().square().mean(dim=(1, 2)).sqrt()
                )

            memory = next_memory

        if self.output_to_factors is None:
            output_feedback = 0.0
        else:
            output_sync = self._sample_sync(
                memory,
                self.output_pair_indices,
            )
            output_feedback = self._factors_to_activity(
                self.output_to_factors(output_sync)
            )
        gate = 1.0 + self.output_scale * torch.tanh(memory + output_feedback)

        if self.trace_enabled:
            self.last_trace = {
                "update_ratio": torch.stack(update_ratios, dim=1).cpu(),
                "memory_rms": torch.stack(memory_rms, dim=1).cpu(),
                "memory_rms_by_basis": torch.stack(memory_rms_by_basis, dim=1).cpu(),
                "gate_by_basis": gate.detach().float().mean(dim=(1, 2)).cpu(),
                "gate_strength_by_basis": (
                    gate.detach().float() - 1.0
                ).abs().mean(dim=(1, 2)).cpu(),
                "output_scale": self.output_scale.detach().float().cpu(),
            }

        return gate

    def forward(self, x):
        self.last_trace = None

        if x.dim() != 3:
            raise ValueError("AdaptiveCtmSplineKANLinear expects [B, N, C] input")

        batch, num_tokens, channels = x.shape
        if channels != self.in_features:
            raise ValueError("input channel mismatch in AdaptiveCtmSplineKANLinear")

        if self.no_kan:
            return self.linear(x)

        flat_x = x.reshape(batch * num_tokens, channels)
        basis = self.static_edge.b_splines(flat_x)
        basis = basis.view(batch, num_tokens, self.in_features, self.num_coeff)

        spline_weight = self.static_edge.scaled_spline_weight
        if self.activity_std_weight == 0.0:
            basis_mean = basis.mean(dim=1)
            basis_std = None
        else:
            basis_var, basis_mean = torch.var_mean(
                basis,
                dim=1,
                unbiased=False,
            )
            basis_std = (
                basis_var.clamp_min(0.0) + self.activity_eps ** 2
            ).sqrt()

        # The thought drive includes token dispersion and is normalized per
        # sample, preventing coefficient magnitude from silencing dynamics.
        activity_summary = basis_mean
        if basis_std is not None:
            activity_summary = (
                activity_summary
                + self.activity_std_weight * basis_std
            )
        thought_activity = activity_summary.unsqueeze(1) * spline_weight.unsqueeze(0)
        if self.trace_enabled:
            activity_rms = (
                thought_activity.square().mean(
                    dim=(1, 2, 3),
                    keepdim=True,
                )
                + self.activity_eps ** 2
            ).sqrt()
            thought_activity = thought_activity / activity_rms
        else:
            activity_rms = None
            thought_activity = F.rms_norm(
                thought_activity,
                thought_activity.shape[1:],
                eps=self.activity_eps ** 2,
            )
        gate = self._run_ctm_ticks(thought_activity)

        if self.trace_enabled and self.last_trace is not None:
            with torch.no_grad():
                spline_activity = (
                    basis_mean.unsqueeze(1)
                    * spline_weight.unsqueeze(0)
                )
                before = spline_activity.detach().float().abs().mean(dim=(1, 2))
                after = (spline_activity * gate).detach().float().abs().mean(dim=(1, 2))
                self.last_trace["contribution_before"] = before.cpu()
                self.last_trace["contribution_after"] = after.cpu()
                self.last_trace["thought_activity_rms"] = (
                    activity_rms.detach().float().cpu()
                )

        base_output = F.linear(
            self.static_edge.base_activation(flat_x),
            self.static_edge.base_weight,
        )
        base_output = base_output.view(batch, num_tokens, self.out_features)
        gated_weight = spline_weight.unsqueeze(0) * gate
        spline_output = torch.bmm(
            basis.reshape(batch, num_tokens, -1),
            gated_weight.reshape(batch, self.out_features, -1).transpose(1, 2),
        )
        return base_output + spline_output


class StaticSplineTokenLinear(nn.Module):
    """Token-shaped wrapper around a standard KANLinear."""

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
        grid_range=(-1, 1),
        no_kan=False,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.no_kan = bool(no_kan)
        if self.no_kan:
            self.linear = nn.Linear(self.in_features, self.out_features)
        else:
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
                grid_range=list(grid_range),
            )

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError("StaticSplineTokenLinear expects [B, N, C]")
        batch, tokens, channels = x.shape
        if channels != self.in_features:
            raise ValueError("input channel mismatch in StaticSplineTokenLinear")
        flat_x = x.reshape(batch * tokens, channels)
        if self.no_kan:
            output = self.linear(flat_x)
        else:
            output = self.static_edge(flat_x)
        return output.reshape(batch, tokens, self.out_features)


class AdaptiveCtmSplineKANLayer(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.0,
        no_kan=False,
        **ctm_kwargs,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        linear_layout = str(
            ctm_kwargs.pop("CtmLinearLayout", "all")
        ).lower()
        grid_size = int(ctm_kwargs.pop("CtmGridSize", 5))
        spline_order = int(ctm_kwargs.pop("CtmSplineOrder", 3))
        layout_to_indices = {
            "all": {1, 2, 3},
            "first": {1},
            "middle": {2},
            "last": {3},
            "first_last": {1, 3},
        }
        if linear_layout not in layout_to_indices:
            raise ValueError(
                "CtmLinearLayout must be one of: "
                "all, first, middle, last, first_last"
            )
        dynamic_indices = layout_to_indices[linear_layout]
        self.linear_layout = linear_layout
        linear_kwargs = dict(
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
            no_kan=no_kan,
            **ctm_kwargs,
        )
        static_kwargs = {
            key: linear_kwargs[key]
            for key in (
                "grid_size",
                "spline_order",
                "scale_noise",
                "scale_base",
                "scale_spline",
                "base_activation",
                "grid_eps",
                "grid_range",
                "no_kan",
            )
        }

        def make_linear(index, source_dim, target_dim):
            if index in dynamic_indices:
                return AdaptiveCtmSplineKANLinear(
                    source_dim,
                    target_dim,
                    **linear_kwargs,
                )
            return StaticSplineTokenLinear(
                source_dim,
                target_dim,
                **static_kwargs,
            )

        self.fc1 = make_linear(1, in_features, hidden_features)
        self.fc2 = make_linear(2, hidden_features, out_features)
        self.fc3 = make_linear(3, hidden_features, out_features)
        self.dwconv_1 = DW_bn_relu(hidden_features)
        self.dwconv_2 = DW_bn_relu(hidden_features)
        self.dwconv_3 = DW_bn_relu(hidden_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.dwconv_1(self.fc1(x), H, W)
        x = self.dwconv_2(self.fc2(x), H, W)
        x = self.dwconv_3(self.fc3(x), H, W)
        return x


class AdaptiveCtmKANBlock(nn.Module):
    def __init__(
        self,
        dim,
        drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        no_kan=False,
        **ctm_kwargs,
    ):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm = norm_layer(dim)
        self.layer = AdaptiveCtmSplineKANLayer(
            in_features=dim,
            hidden_features=dim,
            out_features=dim,
            drop=drop,
            no_kan=no_kan,
            **ctm_kwargs,
        )

    def forward(self, x, H, W):
        return x + self.drop_path(self.layer(self.norm(x), H, W))


def _network_settings(kwargs):
    embed_dims = kwargs.get("embed_dims", [256, 320, 512])
    depths = kwargs.get("depths", [1, 1, 1])
    settings = {
        "embed_dims": embed_dims,
        "no_kan": kwargs.get("no_kan", False),
        "drop_rate": kwargs.get("drop_rate", 0.0),
        "norm_layer": kwargs.get("norm_layer", nn.LayerNorm),
        "dpr": [
            value.item()
            for value in torch.linspace(
                0,
                kwargs.get("drop_path_rate", 0.0),
                sum(depths),
            )
        ],
    }
    settings["ctm_kwargs"] = {
        "CtmTicks": kwargs.get("CtmTicks", 20),
        "CtmDhidden": kwargs.get("CtmDhidden", 64),
        "CtmDropout": kwargs.get("CtmDropout", 0.2),
        "CtmDaction": kwargs.get("CtmDaction", 1024),
        "CtmDout": kwargs.get("CtmDout", 1024),
        "CtmNself": kwargs.get("CtmNself", 32),
        "CtmMemory": kwargs.get("CtmMemory", 10),
        "CtmScaleInit": kwargs.get("CtmScaleInit", 5e-2),
        "CtmActivityStd": kwargs.get("CtmActivityStd", 0.5),
        "CtmPriorScale": kwargs.get("CtmPriorScale", 0.05),
        "CtmActivityEps": kwargs.get("CtmActivityEps", 1e-6),
        "CtmSyncEps": kwargs.get("CtmSyncEps", 1e-4),
        "CtmTerminalReadout": kwargs.get("CtmTerminalReadout", True),
        "CtmPriorMode": kwargs.get("CtmPriorMode", "full"),
        "CtmLinearLayout": kwargs.get("CtmLinearLayout", "all"),
        "CtmGridSize": kwargs.get("CtmGridSize", 5),
        "CtmSplineOrder": kwargs.get("CtmSplineOrder", 3),
    }
    return settings


def _make_block(dim, settings, drop_path_index):
    return AdaptiveCtmKANBlock(
        dim=dim,
        drop=settings["drop_rate"],
        drop_path=settings["dpr"][drop_path_index],
        norm_layer=settings["norm_layer"],
        no_kan=settings["no_kan"],
        **settings["ctm_kwargs"],
    )


class CTKANLight(_CTKANBackbone):
    """Two-block CTKAN used at the deepest encoder and decoder stages."""

    def __init__(self, *args, **kwargs):
        settings = _network_settings(kwargs)
        super().__init__(*args, **kwargs)
        dims = settings["embed_dims"]
        self.block2 = nn.ModuleList([_make_block(dims[2], settings, 1)])
        self.dblock1 = nn.ModuleList([_make_block(dims[1], settings, 0)])


__all__ = [
    "CTKANLight",
    "AdaptiveCtmSplineKANLinear",
    "AdaptiveCtmSplineKANLayer",
    "AdaptiveCtmKANBlock",
]
