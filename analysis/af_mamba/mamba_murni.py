"""Mamba (v1, mamba_ssm 2.2.2 `Mamba` bawaan) dalam PyTorch murni, tanpa kernel CUDA.

Hanya untuk inferensi bobot AF-Mamba (Lee & Chon 2026, github.com/yongbin98/AF_Mamba) di CPU atau GPU
mana pun. Nama parameter sama dengan mamba_ssm supaya `load_state_dict` langsung cocok:
in_proj, conv1d, x_proj, dt_proj, A_log, D, out_proj. Pemindaian selektif mengikuti
`selective_scan_ref` di mamba_ssm/ops/selective_scan_interface.py (delta_softplus=True,
delta_bias = dt_proj.bias).

Arsitektur AF-Mamba ditulis ulang di sini dari Models/AF_mamba.py mereka (repo itu tanpa lisensi, jadi
kodenya tidak disalin ke repo ini; strukturnya diikuti supaya nama bobot cocok).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner, padding=d_conv - 1, bias=True)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.zeros(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, h):                                   # h [b, l, d_model]
        b, L, _ = h.shape
        xz = self.in_proj(h).transpose(1, 2)                # [b, 2di, l]
        x, z = xz.chunk(2, dim=1)
        x = F.silu(self.conv1d(x)[..., :L])
        x_dbl = self.x_proj(x.transpose(1, 2).reshape(b * L, -1))
        dt, Bm, Cm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = (self.dt_proj.weight @ dt.t()).reshape(self.d_inner, b, L).permute(1, 0, 2)   # [b, di, l]
        Bm = Bm.reshape(b, L, self.d_state).transpose(1, 2)  # [b, n, l]
        Cm = Cm.reshape(b, L, self.d_state).transpose(1, 2)
        A = -torch.exp(self.A_log.float())
        y = selective_scan(x, dt, A, Bm, Cm, self.D.float(), z, self.dt_proj.bias.float())
        return self.out_proj(y.transpose(1, 2))


def selective_scan(u, delta, A, B, C, D, z, delta_bias):
    delta = F.softplus(delta + delta_bias[None, :, None])
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
    s = u.new_zeros(u.shape[0], u.shape[1], A.shape[1])
    ys = []
    for i in range(u.shape[2]):
        s = deltaA[:, :, i] * s + deltaB_u[:, :, i]
        ys.append(torch.einsum('bdn,bn->bd', s, C[:, :, i]))
    y = torch.stack(ys, dim=2) + u * D[None, :, None]
    return y * F.silu(z)


class FFNBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(d, d, 3, padding=1), nn.ReLU(), nn.Conv1d(d, d, 3, padding=1))

    def forward(self, x):
        return self.net(x)


class MambaBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mamba = Mamba(d_model=d)
        self.norm1 = nn.LayerNorm(d)
        self.ffn = FFNBlock(d)
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.norm1(self.mamba(x) + x)
        x = self.norm2(self.ffn(x.permute(0, 2, 1)).permute(0, 2, 1) + x)
        return x.permute(0, 2, 1)


class Chomp1d(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        return x[:, :, :-self.n]


class ResidualTCNBlock(nn.Module):
    def __init__(self, cin, cout, k=3, dilation=1, dropout=0.2):
        super().__init__()
        p = (k - 1) * dilation
        self.tcnet = nn.Sequential(nn.Conv1d(cin, cout, k, padding=p, dilation=dilation), Chomp1d(p),
                                   nn.BatchNorm1d(cout), nn.ReLU(), nn.Dropout(dropout))
        self.tcnet2 = nn.Sequential(nn.Conv1d(cout, cout, k, padding=p, dilation=dilation), Chomp1d(p),
                                    nn.BatchNorm1d(cout))
        self.relu = nn.ReLU()
        self.shortcut = nn.Sequential() if cin == cout else nn.Sequential(nn.Conv1d(cin, cout, 1))

    def forward(self, x):
        return self.relu(self.tcnet2(self.tcnet(x)) + self.shortcut(x))


class AFMamba(nn.Module):
    def __init__(self, cin=1, c=32, n_kelas=2):
        super().__init__()
        self.tcn = nn.Sequential(ResidualTCNBlock(cin, c, dilation=1), ResidualTCNBlock(c, c, dilation=2),
                                 ResidualTCNBlock(c, c, dilation=4))
        self.mamba = nn.Sequential(MambaBlock(c), nn.Dropout(0.2))
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(2 * c, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Linear(32, n_kelas))

    def forward(self, x):                                   # x [b, 1, 3600]
        x = self.mamba(self.tcn(x))
        x = torch.cat([self.global_avg_pool(x), self.global_max_pool(x)], dim=1).squeeze(-1)
        return self.fc(x)
