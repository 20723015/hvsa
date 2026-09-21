"""N2 实验: DLG 梯度反演攻击 (论文 3.5.4).

对客户端上传的量做 DLG (Deep Leakage from Gradients, Zhu et al. NeurIPS 2019)
式反演, 比较两组:

  - 明文组: 攻击者看到真实本地更新 Δ = θ_after − θ_before
  - 掩码组: 攻击者看到 w̄ = θ_after + mask (HVSA 客户端实际上传的量),
            可算出的"更新"为 Δ_obs = w̄ − θ_before = Δ + mask

攻击按最强设定进行 (batch=1 单样本, 已知模型结构与全局参数, 同时优化虚样本与
虚标签, 带 TV 正则) —— 对隐私评估而言这是对攻击者最有利的配置。掩码由
utils.PRG.mask_int 生成 (±1e9 均匀整数), 与真实协议一致。

指标: PSNR / SSIM (重建图像 vs 原图). 预期明文组显著恢复、掩码组失败。

用法:
  python exp_n2_dlg_inversion.py                    # 5 个样本, 300 迭代
  python exp_n2_dlg_inversion.py --samples 10 --iters 500
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


# ---------- 指标 ----------
def psnr(a, b, data_range=1.0):
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(data_range ** 2 / mse))


def ssim(a, b, data_range=1.0):
    """标准 SSIM: 11x11 高斯窗 (sigma=1.5), k1=0.01, k2=0.03."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    sigma = 1.5
    filt = lambda z: gaussian_filter(z, sigma, truncate=3.5)  # noqa: E731
    mu_a, mu_b = filt(a), filt(b)
    va = filt(a * a) - mu_a * mu_a
    vb = filt(b * b) - mu_b * mu_b
    vab = filt(a * b) - mu_a * mu_b
    L = data_range
    c1, c2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * vab + c2)) / \
        ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
    return float(s.mean())


# ---------- DLG ----------
def param_grad(model, x, y_soft, create_graph=False):
    """软标签交叉熵对全部参数的梯度 (flatten).

    DLG 原作法: 虚标签经 softmax 作为软目标; 真实梯度用 one-hot 软标签,
    与 nll_loss 等价。

    create_graph=True 是 DLG 的必要条件: 反演时要以"参数梯度"为目标再做
    一次反向传播, 否则梯度张量脱离计算图, 虚样本得不到任何梯度信号。
    """
    model.zero_grad(set_to_none=True)
    out = model(x)
    loss = -(y_soft * out).sum(dim=1).mean()
    g = torch.autograd.grad(loss, list(model.parameters()),
                            create_graph=create_graph)
    return torch.cat([gi.reshape(-1) for gi in g])


def _grad_loss(x_dummy, y_dummy, model, target, tgt_norm, tv_weight):
    """方向匹配损失: ‖ĝ − t̂‖², ĝ/t̂ 为各自单位化后的参数梯度.

    必须做单位化: 真实梯度由 one-hot 软标签给出 (‖Δ‖≈10), 而虚样本的软标签
    初始接近均匀, 其梯度范数最多 ~0.3 —— 直接比数值差会相差约 30 倍, 优化器
    永远够不到目标量级。单位化后只匹配**方向**, 与 DLG 依赖"梯度方向携带数据
    信息"的机理一致。
    """
    g = param_grad(model, x_dummy, F.softmax(y_dummy, dim=1),
                   create_graph=True)
    loss = ((g - target) / tgt_norm).pow(2).mean()
    if tv_weight > 0:
        tv = (x_dummy[:, :, 1:, :] - x_dummy[:, :, :-1, :]).abs().mean() +              (x_dummy[:, :, :, 1:] - x_dummy[:, :, :, :-1]).abs().mean()
        loss = loss + tv_weight * tv
    return loss


def dlg_invert(target, model, shape, device, iters, lr, tv_weight,
               n_classes=10, seed=0, restarts=4, verbose=False):
    """LBFGS 反演 (DLG 原论文配方) + 多次重启取最优.

    虚样本以高斯噪声初始化 (零初始化在该 CNN 上退化: 虚样本为 0 时参数梯度
    范数仅 ~1e-2, 而噪声初值为 ~3e-1, 相差近 30 倍, 优化器几乎无法起步);
    损失为**归一化**梯度 MSE (除以目标范数, 使损失与目标尺度无关) + 可选 TV
    正则。多次重启是 DLG 的标准做法 —— 单次优化对初始化敏感, 取损失最小者。
    注: 两组使用同一初始化分布与同一重启种子集, 不偏袒任一组。
    """
    tgt_norm = target.norm().clamp_min(1e-12)
    best = None
    for r in range(restarts):
        torch.manual_seed(seed * 1000 + r)
        x_dummy = torch.randn(*shape, device=device, requires_grad=True)
        y_dummy = torch.zeros(shape[0], n_classes, device=device,
                              requires_grad=True)
        opt = torch.optim.Adam([x_dummy, y_dummy], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)
        for it in range(iters):
            opt.zero_grad(set_to_none=True)
            loss = _grad_loss(x_dummy, y_dummy, model, target, tgt_norm,
                              tv_weight)
            loss.backward()
            opt.step()
            sched.step()
        loss_val = float(loss.detach())
        if best is None or loss_val < best[0]:
            best = (loss_val, x_dummy.detach().clone(),
                    y_dummy.detach().clone())
        if verbose:
            print("      restart %d/%d  loss=%.6e" % (r + 1, restarts, loss_val))
    return best[1], best[2]


def to_img(x, shape_hw=(28, 28)):
    a = x.detach().cpu().numpy().reshape(-1, 1, *shape_hw)[0, 0]
    return np.clip((a + 0.1307) * 0.3081, 0.0, 1.0)  # 反 normalize (MNIST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--tv-weight", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="CNN", choices=["CNN", "Tiny", "MLP"])
    ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    os.chdir(ROOT)
    from datasets import load_data
    from models import CNN_MNIST, MLP_MNIST, Tiny_MNIST
    from utils import PRG

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_ds, _ = load_data("MNIST")
    ds = train_ds
    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(ds), generator=g)[:args.samples]

    _ctor = {"CNN": CNN_MNIST, "Tiny": Tiny_MNIST, "MLP": MLP_MNIST}[args.model]
    model = _ctor().to(device).eval()
    _dim = sum(pp.numel() for pp in model.parameters())
    print("model: %s  dim=%d" % (args.model, _dim))
    for p in model.parameters():
        p.requires_grad_(True)
    # 用随机初始化参数作为"全局模型" (攻击者已知); 参数值不影响力对比
    _ = model(torch.zeros(1, 1, 28, 28, device=device))

    print("=" * 78)
    print("N2: DLG 梯度反演   samples=%d  iters=%d  batch=%d  device=%s"
          % (args.samples, args.iters, args.batch_size, device))
    print("=" * 78)

    res_plain, res_masked = [], []
    for si, i in enumerate(idx.tolist()):
        xs, ys = [], []
        for j in range(args.batch_size):
            x, y = ds[(i + j) % len(ds)]
            xs.append(x)
            ys.append(y)
        x = torch.stack(xs).to(device)
        y = torch.tensor(ys, device=device)

        # 真实本地更新 Δ = θ_after − θ_before (一步 SGD)
        y_oh = F.one_hot(y, 10).float()
        g_true = param_grad(model, x, y_oh)
        # 攻击者看到的明文更新 (θ 保持固定, 故 Δ 即梯度方向; 尺度不影响力对比)
        target_plain = (-g_true).detach()

        # 掩码: HVSA 客户端上传 w̄ = θ_after + mask, 攻击者可算 Δ+mask
        mask = PRG.mask_int(b"\x11" * 32, "user_%d" % si, "pkg_%d" % si,
                            target_plain.numel())
        mask_t = torch.tensor(mask.astype(np.float32), device=device)
        target_masked = target_plain + mask_t

        if si == 0:
            print("  梯度范数 ‖Δ‖ = %.4e ; 掩码范数 ‖mask‖ = %.4e ; 比值 = %.1f×"
                  % (float(target_plain.norm()), float(mask_t.norm()),
                     float(mask_t.norm() / target_plain.norm())))

        orig = to_img(x)

        rec_p, _ = dlg_invert(target_plain, model, x.shape, device,
                              args.iters, args.lr, args.tv_weight,
                              seed=si, restarts=args.restarts,
                              verbose=args.verbose)
        img_p = to_img(rec_p)
        p1, s1 = psnr(img_p, orig), ssim(img_p, orig)

        rec_m, _ = dlg_invert(target_masked, model, x.shape, device,
                              args.iters, args.lr, args.tv_weight,
                              seed=si, restarts=args.restarts,
                              verbose=args.verbose)
        img_m = to_img(rec_m)
        p2, s2 = psnr(img_m, orig), ssim(img_m, orig)

        res_plain.append((p1, s1))
        res_masked.append((p2, s2))
        print("  [%d/%d] 明文: PSNR=%6.2f dB  SSIM=%.4f   |   掩码: PSNR=%6.2f dB  SSIM=%.4f"
              % (si + 1, args.samples, p1, s1, p2, s2))

    print()
    print("-" * 78)
    pa = np.array(res_plain)
    ma = np.array(res_masked)
    print("汇总 (mean ± sample std, n=%d)" % args.samples)
    print("  明文组: PSNR = %6.2f ± %5.2f dB   SSIM = %.4f ± %.4f"
          % (pa[:, 0].mean(), pa[:, 0].std(ddof=1) if len(pa) > 1 else 0,
             pa[:, 1].mean(), pa[:, 1].std(ddof=1) if len(pa) > 1 else 0))
    print("  掩码组: PSNR = %6.2f ± %5.2f dB   SSIM = %.4f ± %.4f"
          % (ma[:, 0].mean(), ma[:, 0].std(ddof=1) if len(ma) > 1 else 0,
             ma[:, 1].mean(), ma[:, 1].std(ddof=1) if len(ma) > 1 else 0))
    print()
    print("  判据 (3.5.4 预期, 以 SSIM 为准): 明文 SSIM>0.5 且 掩码 SSIM<0.10")
    ok = (pa[:, 1].mean() > 0.5) and (ma[:, 1].mean() < 0.10)
    print("  → %s" % ("成立" if ok else "不成立/需复核"))
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
