"""VPoser 人体姿态先验（torch 2.x 版，不依赖 torchgeometry）。

VPoser 是 21 关节的 VAE（latent 32 维），在 AMASS 上训练，编码"自然人体姿态"的
流形。这里用最小 torch 2.x 实现——原 ``human_body_prior`` 依赖死库 torchgeometry
（2019 改名 kornia，torch 2.x 上必崩），此处用 Rodrigues 对数映射替换了
``matrot2aa``，其余是纯 Linear/BatchNorm，torch 2.x 直接可用。

姿态约定（与 EasyMocap SMPLlayer 对齐，实测自 frame_*.npz）：
  poses(72): [0:3]=根关节(恒 0), [3:66]=21 body 关节(=VPoser), [66:72]=手(恒 0)
  Rh(3) 全局朝向、Th(3) 全局平移、shapes(10) 体型，均独立于 poses。

核心 API：
  - ``load_vposer(ckpt_path)`` 加载 VPoser；
  - ``fit_frame(vposer, smpl, target_body25, ...)`` 在潜空间优化姿态拟合 body25。
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

__all__ = [
    "VPoser", "matrot2aa", "load_vposer", "latent_to_poses",
    "forward_body25", "fit_frame",
]


class ContinousRotReprDecoder(nn.Module):
    """6D 连续旋转 -> 3x3 旋转矩阵（VPoser 官方实现）。"""

    def forward(self, module_input):
        reshaped_input = module_input.view(-1, 3, 2)
        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)
        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)
        return torch.stack([b1, b2, b3], dim=-1)


class VPoser(nn.Module):
    """21 关节人体姿态 VAE（latent 32 维）。输入 axis-angle(63)，输出旋转矩阵。"""

    def __init__(self, num_neurons=512, latentD=32, num_joints=21, use_cont_repr=True):
        super().__init__()
        self.latentD = latentD
        self.num_joints = num_joints
        self.use_cont_repr = use_cont_repr
        n_features = num_joints * 3
        self.bodyprior_enc_bn1 = nn.BatchNorm1d(n_features)
        self.bodyprior_enc_fc1 = nn.Linear(n_features, num_neurons)
        self.bodyprior_enc_bn2 = nn.BatchNorm1d(num_neurons)
        self.bodyprior_enc_fc2 = nn.Linear(num_neurons, num_neurons)
        self.bodyprior_enc_mu = nn.Linear(num_neurons, latentD)
        self.bodyprior_enc_logvar = nn.Linear(num_neurons, latentD)
        self.dropout = nn.Dropout(p=.1, inplace=False)
        self.bodyprior_dec_fc1 = nn.Linear(latentD, num_neurons)
        self.bodyprior_dec_fc2 = nn.Linear(num_neurons, num_neurons)
        if use_cont_repr:
            self.rot_decoder = ContinousRotReprDecoder()
        self.bodyprior_dec_out = nn.Linear(num_neurons, num_joints * 6)

    def encode(self, Pin):
        Xout = Pin.view(Pin.size(0), -1)
        Xout = self.bodyprior_enc_bn1(Xout)
        Xout = F.leaky_relu(self.bodyprior_enc_fc1(Xout), negative_slope=.2)
        Xout = self.bodyprior_enc_bn2(Xout)
        Xout = self.dropout(Xout)
        Xout = F.leaky_relu(self.bodyprior_enc_fc2(Xout), negative_slope=.2)
        return torch.distributions.normal.Normal(
            self.bodyprior_enc_mu(Xout), F.softplus(self.bodyprior_enc_logvar(Xout)))

    def decode(self, Zin):
        Xout = F.leaky_relu(self.bodyprior_dec_fc1(Zin), negative_slope=.2)
        Xout = self.dropout(Xout)
        Xout = F.leaky_relu(self.bodyprior_dec_fc2(Xout), negative_slope=.2)
        Xout = self.bodyprior_dec_out(Xout)
        if self.use_cont_repr:
            Xout = self.rot_decoder(Xout)
        else:
            Xout = torch.tanh(Xout)
        return Xout.view([-1, self.num_joints, 3, 3])


def matrot2aa(R):
    """旋转矩阵 -> 轴角（torch，对数映射，替换 torchgeometry）。R: (..., 3, 3)。"""
    cos_theta = torch.clamp(
        (R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1.0) / 2.0, -1.0, 1.0)
    theta = torch.acos(cos_theta)
    r = torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1],
    ], dim=-1)
    sin_theta = torch.sin(theta).unsqueeze(-1)
    safe = torch.where(sin_theta.abs() < 1e-9, torch.ones_like(sin_theta), sin_theta)
    return theta.unsqueeze(-1) * (r / (2.0 * safe))


def load_vposer(ckpt_path: str, device: str = "cuda") -> VPoser:
    """加载 VPoser 预训练权重（~2.7MB）。"""
    model = VPoser(num_neurons=512, latentD=32, num_joints=21, use_cont_repr=True)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def latent_to_poses(vposer: VPoser, z: torch.Tensor) -> torch.Tensor:
    """z (n,32) -> full poses (n,72)：根/手恒 0，body 来自 VPoser decode。"""
    matrot = vposer.decode(z)                 # (n,21,3,3)
    body_aa = matrot2aa(matrot)              # (n,21,3)
    n = body_aa.shape[0]
    dev = body_aa.device
    return torch.cat([torch.zeros(n, 3, device=dev),
                      body_aa.reshape(n, 63),
                      torch.zeros(n, 6, device=dev)], dim=1)


def forward_body25(vposer: VPoser, smpl, z, shapes, Rh, Th) -> torch.Tensor:
    """z/shapes/Rh/Th -> body25 关节 (n,25,3) 世界坐标。"""
    poses = latent_to_poses(vposer, z)
    return smpl(poses=poses, shapes=shapes, Rh=Rh, Th=Th,
                return_verts=False, return_tensor=True)


def fit_frame(vposer: VPoser, smpl, target_body25,
              shapes=None, n_iter: int = 150, lr: float = 0.05,
              lambda_z: float = 1e-3) -> Optional[Dict]:
    """单帧：在 VPoser 潜空间优化 z + Rh + Th 拟合 body25 目标（世界坐标，NaN=缺失）。

    shapes 固定（默认均值 0），因为从稀疏关键点优化体型欠定且非本次要解决的病；
    要解决的是"姿态扭曲/翻转"，VPoser 潜空间天然约束姿态自然。

    Returns:
        dict(vertices/joints/joints_body25/params) 与 EasyMocap reconstruct_batch
        单帧结果同构；失败返回 None。
    """
    device = vposer.bodyprior_dec_fc1.weight.device
    tgt = torch.as_tensor(target_body25, dtype=torch.float32, device=device).reshape(-1, 3)
    valid = torch.isfinite(tgt).all(dim=-1)
    if valid.sum() < 5:
        return None

    # 初始化：Th = 骨盆(MidHip=8)世界坐标；Rh=0；z=0(均值姿态)
    th_init = tgt[8].clone() if valid[8] else tgt[valid].mean(0)
    z = torch.zeros(1, 32, device=device, requires_grad=True)
    Rh = torch.zeros(1, 3, device=device, requires_grad=True)
    Th = th_init.reshape(1, 3).requires_grad_(True)
    shapes_t = torch.as_tensor(
        np.zeros(10) if shapes is None else np.asarray(shapes),
        dtype=torch.float32, device=device).reshape(1, -1)

    opt = torch.optim.Adam([z, Rh, Th], lr=lr)
    for _ in range(n_iter):
        opt.zero_grad()
        j25 = forward_body25(vposer, smpl, z, shapes_t, Rh, Th)
        diff = j25[0][valid] - tgt[valid]
        loss = (diff * diff).mean() + lambda_z * (z * z).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        poses = latent_to_poses(vposer, z)
        verts = smpl(poses=poses, shapes=shapes_t, Rh=Rh, Th=Th,
                     return_verts=True, return_tensor=False)
        j24 = smpl(poses=poses, shapes=shapes_t, Rh=Rh, Th=Th,
                   return_verts=False, return_smpl_joints=True, return_tensor=False)
        j25 = forward_body25(vposer, smpl, z, shapes_t, Rh, Th)
    return {
        "vertices": np.asarray(verts[0], dtype=np.float64),
        "joints": np.asarray(j24[0], dtype=np.float64),
        "joints_body25": np.asarray(j25[0].detach().cpu().numpy(), dtype=np.float64),
        "params": {
            "poses": poses.detach().cpu().numpy().reshape(-1).astype(np.float64),
            "shapes": shapes_t.detach().cpu().numpy().reshape(-1).astype(np.float64),
            "Rh": Rh.detach().cpu().numpy().reshape(-1).astype(np.float64),
            "Th": Th.detach().cpu().numpy().reshape(-1).astype(np.float64),
        },
    }
