import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math

from kan import KANLinear

# -------------------- KANLayer --------------------
class KANLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., no_kan=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features

        grid_size = 5
        spline_order = 3
        scale_noise = 0.1
        scale_base = 1.0
        scale_spline = 1.0
        base_activation = torch.nn.SiLU
        grid_eps = 0.02
        grid_range = [-1, 1]

        if not no_kan:
            self.fc1 = KANLinear(in_features, hidden_features, grid_size=grid_size, spline_order=spline_order,
                                 scale_noise=scale_noise, scale_base=scale_base, scale_spline=scale_spline,
                                 base_activation=base_activation, grid_eps=grid_eps, grid_range=grid_range)
            self.fc2 = KANLinear(hidden_features, out_features, grid_size=grid_size, spline_order=spline_order,
                                 scale_noise=scale_noise, scale_base=scale_base, scale_spline=scale_spline,
                                 base_activation=base_activation, grid_eps=grid_eps, grid_range=grid_range)
            self.fc3 = KANLinear(hidden_features, out_features, grid_size=grid_size, spline_order=spline_order,
                                 scale_noise=scale_noise, scale_base=scale_base, scale_spline=scale_spline,
                                 base_activation=base_activation, grid_eps=grid_eps, grid_range=grid_range)
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.fc3 = nn.Linear(hidden_features, out_features)

        self.dwconv_1 = DW_bn_relu(hidden_features)
        self.dwconv_2 = DW_bn_relu(hidden_features)
        self.dwconv_3 = DW_bn_relu(hidden_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.fc1(x.reshape(B * N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_1(x, H, W)
        x = self.fc2(x.reshape(B * N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_2(x, H, W)
        x = self.fc3(x.reshape(B * N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_3(x, H, W)
        return x


# -------------------- KANBlock --------------------
class KANBlock(nn.Module):
    def __init__(self, dim, drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, no_kan=False):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim)
        self.layer = KANLayer(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, no_kan=no_kan)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.layer(self.norm2(x), H, W))
        return x


# -------------------- DWConv --------------------
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


# -------------------- DW_bn_relu --------------------
class DW_bn_relu(nn.Module):
    def __init__(self, dim=768):
        super(DW_bn_relu, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2)
        return x


# -------------------- PatchEmbed --------------------
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


# -------------------- ConvLayer --------------------
class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.conv(input)


# -------------------- D_ConvLayer --------------------
class D_ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(D_ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.conv(input)


# -------------------- TemporalEdgeKANLayer --------------------
class TemporalEdgeKANLayer(nn.Module):
    def __init__(self, in_features, out_features, history_len=4, sync_pairs=16,
                 edge_chunk_size=32, no_kan=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.history_len = history_len
        self.sync_pairs = max(1, sync_pairs)
        self.edge_chunk_size = max(1, edge_chunk_size)
        self.no_kan = no_kan

        if no_kan:
            self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
            nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        else:
            self.static_edge = KANLinear(
                in_features, out_features,
                grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0,
                scale_spline=1.0, base_activation=torch.nn.SiLU, grid_eps=0.02, grid_range=[-1, 1],
            )

        self.nlm_weight = nn.Parameter(torch.randn(out_features, in_features, history_len) * 0.02)
        self.nlm_bias = nn.Parameter(torch.zeros(out_features, in_features))

        pairs = torch.randint(0, out_features, (self.sync_pairs, 2))
        self.register_buffer('sync_pairs_idx', pairs)
        self.sync_decay = nn.Parameter(torch.ones(self.sync_pairs) * 0.1)
        self.sync_to_input = nn.Linear(self.sync_pairs, in_features)
        self.init_sync = nn.Parameter(torch.zeros(1, self.sync_pairs))

    def _append_history(self, history, values, width):
        if (history is None or history.shape[0] != values.shape[0] or
                history.shape[1] != width or history.device != values.device or
                history.dtype != values.dtype):
            history = values.new_zeros(values.shape[0], width, self.history_len)
        return torch.cat([history[:, :, 1:], values.unsqueeze(-1)], dim=-1)

    def _initial_sync(self, batch, device, dtype):
        return self.init_sync.to(device=device, dtype=dtype).expand(batch, -1)

    def _compute_sync(self, output_history):
        pair_i = self.sync_pairs_idx[:, 0]
        pair_j = self.sync_pairs_idx[:, 1]
        traj_i = output_history[:, pair_i, :]
        traj_j = output_history[:, pair_j, :]
        t = torch.arange(output_history.shape[-1], device=output_history.device, dtype=output_history.dtype)
        decay = F.softplus(self.sync_decay).to(output_history.dtype).unsqueeze(-1)
        weights = torch.exp(-decay * (output_history.shape[-1] - 1 - t))
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)
        return (traj_i * traj_j * weights.unsqueeze(0)).sum(dim=-1)

    def _static_edge_chunk(self, flat_x, basis, start, end):
        if self.no_kan:
            base = F.silu(flat_x).unsqueeze(1)
            return base * self.base_weight[start:end].unsqueeze(0)
        base = self.static_edge.base_activation(flat_x).unsqueeze(1)
        base = base * self.static_edge.base_weight[start:end].unsqueeze(0)
        spline_weight = self.static_edge.scaled_spline_weight[start:end]
        spline = torch.einsum('bik,oik->boi', basis, spline_weight)
        return base + spline

    def forward(self, flat_x, state=None):
        edge_history = None if state is None else state.get('edge_history')
        output_history = None if state is None else state.get('output_history')
        prev_sync = None if state is None else state.get('sync')

        edge_history = self._append_history(edge_history, flat_x, self.in_features)

        if prev_sync is None or prev_sync.shape[0] != flat_x.shape[0] or prev_sync.device != flat_x.device:
            prev_sync = self._initial_sync(flat_x.shape[0], flat_x.device, flat_x.dtype)

        sync_feedback = self.sync_to_input(prev_sync).unsqueeze(1)
        basis = None if self.no_kan else self.static_edge.b_splines(flat_x)

        chunks = []
        for start in range(0, self.out_features, self.edge_chunk_size):
            end = min(start + self.edge_chunk_size, self.out_features)
            gate = torch.einsum('bim,oim->boi', edge_history, self.nlm_weight[start:end])
            gate = gate + self.nlm_bias[start:end].unsqueeze(0) + sync_feedback
            gate = torch.sigmoid(gate)
            static_edge = self._static_edge_chunk(flat_x, basis, start, end)
            chunks.append((static_edge * gate).sum(dim=-1))

        y = torch.cat(chunks, dim=-1)
        output_history = self._append_history(output_history, y, self.out_features)
        sync = self._compute_sync(output_history)

        next_state = {'edge_history': edge_history, 'output_history': output_history, 'sync': sync}
        return y, next_state


# -------------------- TemporalEdgeKANBlock --------------------
class TemporalEdgeKANBlock(nn.Module):
    def __init__(self, dim, drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 history_len=4, sync_pairs=16, edge_chunk_size=32, no_kan=False):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm = norm_layer(dim)
        self.temporal_edge = TemporalEdgeKANLayer(
            dim, dim, history_len=history_len, sync_pairs=sync_pairs,
            edge_chunk_size=edge_chunk_size, no_kan=no_kan,
        )
        self.dwconv = DW_bn_relu(dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, state=None):
        B, N, C = x.shape
        norm_x = self.norm(x).reshape(B * N, C)
        temporal_out, next_state = self.temporal_edge(norm_x, state)
        temporal_out = temporal_out.reshape(B, N, C).contiguous()
        temporal_out = self.dwconv(temporal_out, H, W)
        x = x + self.drop_path(temporal_out)
        return x, next_state
