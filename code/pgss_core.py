# -*- coding: utf-8 -*-
"""PGSS 核心数值模块 —— **无任何导入副作用，无路径依赖**（GPT 第 36 轮 §2.3）。

为什么要有这个文件
    原先引擎从 `pgss_validate.py` 取 `LR_KW` / `LAMBDA_GRID` / `fit_b4`，
    而那个模块在**导入时**会：
        os.environ.setdefault("JOBLIB_TEMP_FOLDER", r"C:\\pgss_tmp")
        pathlib.Path(r"C:\\pgss_tmp").mkdir(exist_ok=True)
        ROOT = pathlib.Path(r"C:\\Users\\<旧用户>\\Desktop\\PGSS-...")   # 笔记本时代路径
        RES  = A / "T1_RESULTS"; RES.mkdir(parents=True, exist_ok=True)
    ⇒ 导入核心建模代码就会在 C 盘建目录、并绑定一个已不存在的旧项目根。
    本模块把**数值参数、目标函数、拟合函数**原样搬过来，去掉全部副作用。

**纪律：本文件只允许 import 第三方库，不得读写文件、不得建目录、不得设环境变量。**
数值定义与 `pgss_validate.py` 逐行一致（搬运，不修改模型定义）；
一致性由 `pgss_core_tests.py` 的逐函数比对测试保证。
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

# ── 数值参数（与 pgss_validate.py 同值）──────────────────────────
SEED_FIT = 20260921
LR_TOL = 1e-8
LR_KW = dict(penalty="l2", C=1.0, solver="lbfgs", max_iter=3000, tol=LR_TOL)
LR_UNPEN = dict(penalty=None, solver="lbfgs", max_iter=3000, tol=LR_TOL)
# Phase 2 数值修正 N-1：显式收紧容差。冻结规格定义的是目标函数，tol 只决定解得多准；
# sklearn 默认 tol=1e-4 会提前停，对 G2 的 Brier 误差约为 G0 的 10 倍 —— 随臂而异的系统性偏差。
LAMBDA_GRID = [0.0] + [10.0 ** k for k in np.arange(-4, 2.01, 0.5)] + [np.inf]
MIN_FIT = 30


# ── 预处理（中位数插补 + 标准化）──────────────────────────────
def prep_fit(Xtr):
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    A = np.where(np.isnan(Xtr), med, Xtr)
    mu, sd = A.mean(axis=0), A.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return med, mu, sd


def prep_apply(X, p):
    med, mu, sd = p
    return (np.where(np.isnan(X), med, X) - mu) / sd


def d2(X, z):
    """G2 的设计矩阵：[X, z, X*z] —— 允许组别特异截距与斜率"""
    zz = np.asarray(z, float)[:, None]
    return np.hstack([X, zz, X * zz])


def brier(y, p):
    return (np.asarray(y, float) - np.asarray(p, float)) ** 2


def two_track(y, z, p):
    y = np.asarray(y, float); z = np.asarray(z, bool); p = np.asarray(p, float)
    return 0.5 * (brier(y[z], p[z]).mean() + brier(y[~z], p[~z]).mean())


# ── FR-PP（Fused-Ridge Partial Pooling）目标函数与拟合 ──────────
#   Σℓ/n + (1/(2Cn))(‖β_A‖² + ‖β_B‖²) + λ‖β_A − β_B‖²
#   λ=inf 时退化为共享斜率（tie 分支），此时 ridge 为 (1/(Cn))‖β‖²（两份相加）
def _softplus(z):
    """log(1+exp(z))，与 `np.logaddexp(0, z)` **逐位相同**但快约 1.9 倍。

    数学上 log(1+e^z) = max(z,0) + log(1+e^{-|z|})，两种写法都数值稳定。
    实测（35,000 元素，单线程）：logaddexp 2.513 ms vs 本式 1.309 ms。
    逐位相同性在 ±1e-8 / ±1 / ±50 / ±800 四个尺度上各 20,000 个点验证，
    最大差均为 0.000e+00。

    纪律：**只有逐位相同的替换才允许**。同时测过 expit 的两个替代写法
    （1/(1+exp(-z)) 与 0.5*(1+tanh(z/2))），二者与 scipy expit 差 2.22e-16，
    **不是逐位相同，一律不用** —— 哪怕更快。
    """
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))


def _b4_obj(w, Xa, ya, Xb, yb, n, C, lam, tie):
    p = Xa.shape[1]
    if tie:
        bs = w[:p]; ia, ib = w[p], w[p + 1]
        za = Xa @ bs + ia; zb = Xb @ bs + ib
        pa = expit(za); pb = expit(zb)
        la = _softplus(za) - ya * za
        lb = _softplus(zb) - yb * zb
        val = (la.sum() + lb.sum()) / n + (1.0 / (C * n)) * (bs @ bs)
        ra = pa - ya; rb = pb - yb
        g = np.empty_like(w)
        g[:p] = (Xa.T @ ra + Xb.T @ rb) / n + (2.0 / (C * n)) * bs
        g[p] = ra.sum() / n; g[p + 1] = rb.sum() / n
        return val, g
    ba, ia = w[:p], w[p]
    bb, ib = w[p + 1:2 * p + 1], w[2 * p + 1]
    za = Xa @ ba + ia; zb = Xb @ bb + ib
    pa = expit(za); pb = expit(zb)
    la = _softplus(za) - ya * za
    lb = _softplus(zb) - yb * zb
    d = ba - bb
    val = ((la.sum() + lb.sum()) / n
           + (1.0 / (2 * C * n)) * (ba @ ba + bb @ bb) + lam * (d @ d))
    ra = pa - ya; rb = pb - yb
    g = np.empty_like(w)
    g[:p] = Xa.T @ ra / n + (1.0 / (C * n)) * ba + 2 * lam * d
    g[p] = ra.sum() / n
    g[p + 1:2 * p + 1] = Xb.T @ rb / n + (1.0 / (C * n)) * bb - 2 * lam * d
    g[2 * p + 1] = rb.sum() / n
    return val, g


class OptimFailure(RuntimeError):
    """优化器未成功收敛。**必须抛出，不得返回参数。**

    GPT 第 39 轮实测注入 success=False / status=2 / 异常终止，
    原实现仍返回参数（并给出全 0.5 预测）⇒ 失败被当成成功。
    scipy 明确提供 success/status/message，返回了 x 不等于优化成功。
    """


# 迭代预算：**预算不是判据**。R38.1 实测 eICU/女性 在 λ=100 处 nit 触到 5000 上限、
# ‖g‖∞ 停在 1.532e-06（status=1），而同一问题给到 50000 后在 1357–数千步内 status=0 收敛。
# λ 大时融合罚项 λ‖β_A−β_B‖² 极硬 ⇒ 病态 ⇒ 需要更多步。
# 提高上限**不可能改变任何已收敛的拟合**（它们按自身判据先停），因此不是放宽检查。
# 同时实测把内部 gtol 从 1e-12 放到 1e-8 对速度几乎无帮助（优化器不在 gtol 上终止），
# 故 gtol/ftol 一律不动，只提预算。

# 接受阈值：两条路径各自预先固定，**不随运行结果调整**
GTOL_ACCEPT = 1e-6     # 路径一：退出码为成功时的梯度上限
GTOL_CERTIFY = 1e-7    # 路径二：退出码 status=2 时的梯度上限（**更严**）


def _check_optim(r, who, gtol_accept=GTOL_ACCEPT, gtol_certify=GTOL_CERTIFY):
    """检查 scipy 优化结果。返回真实诊断字典；不可接受即抛 OptimFailure。

    为什么不是「只看 r.success」
        本项目的 `ftol=1e-16` 低于双精度对该目标函数可分辨的尺度，
        因此迭代点抵达数值地板后，L-BFGS-B 必然报
        `ABNORMAL_TERMINATION_IN_LNSRCH`（status=2）—— 这是**预期终止态**，不是实质失败。
        实测案例：MIMIC/住院死亡/年龄≥65 内部折 3，λ=1.0，
        status=2 但 ‖g‖∞ = 2.716e-09，是收敛良好的稳定点。
        若只看退出码，会把它判成失败。

    **预先规定的两条接受路径**（不随运行结果调整）
        路径一  success=True            且 ‖g‖∞ ≤ GTOL_ACCEPT (1e-6)
        路径二  success=False, status=2 且 ‖g‖∞ ≤ GTOL_CERTIFY(1e-7，**更严**)
                ⇒ 记 `accepted_on_gradient=True`，**计数上报，不隐藏**

    **一律拒绝**
        · status=1（迭代/函数预算耗尽）—— 无论梯度多好；预算耗尽是真失败
        · status=2 但梯度超过 GTOL_CERTIFY
        · 梯度非有限、目标函数非有限
        · 其他任何 status
    """
    g = np.asarray(getattr(r, "jac", np.array([np.nan])), float)
    gn = float(np.abs(g).max()) if g.size and np.isfinite(g).all() else float("nan")
    nit = int(getattr(r, "nit", -1)); nfev = int(getattr(r, "nfev", -1))
    st = int(r.status)
    diag = dict(who=who, success=bool(r.success), status=st,
                message=str(r.message)[:200], nit=nit, nfev=nfev,
                fun=float(r.fun), grad_inf_norm=gn,
                hit_maxiter=bool(nit >= 50000), hit_maxfun=bool(nfev >= 200000),
                accepted_on_gradient=False)
    if not np.isfinite(diag["fun"]):
        raise OptimFailure(f"{who} 目标函数非有限：{diag['fun']}")
    if r.success:
        if not np.isfinite(gn) or gn > gtol_accept:
            raise OptimFailure(f"{who} 退出码为成功但梯度不可接受：‖g‖∞={gn:.3e} "
                               f"> {gtol_accept:.0e}（nit={nit}）")
        return diag
    if st == 2 and np.isfinite(gn) and gn <= gtol_certify:
        diag["accepted_on_gradient"] = True
        return diag
    raise OptimFailure(f"{who} 优化不可接受：status={st} nit={nit} nfev={nfev} "
                       f"‖g‖∞={gn:.3e}（status=1 一律拒绝；status=2 需 ‖g‖∞ ≤ "
                       f"{gtol_certify:.0e}）msg={str(r.message)[:90]}")


def fit_b4(Xa, ya, Xb, yb, lam, C=1.0, w0=None, ftol=1e-16, diag_out=None):
    n = len(ya) + len(yb); p = Xa.shape[1]
    tie = np.isinf(lam)
    if w0 is None or len(w0) != (p + 2 if tie else 2 * p + 2):
        w0 = np.zeros(p + 2 if tie else 2 * p + 2)
    r = minimize(_b4_obj, w0, args=(Xa, ya, Xb, yb, n, C, 0.0 if tie else lam, tie),
                 method="L-BFGS-B", jac=True,
                 options=dict(maxiter=50000, maxfun=200000, ftol=ftol, gtol=1e-12))
    d = _check_optim(r, f"FR-PP(lam={'inf' if tie else lam})")
    if diag_out is not None: diag_out.append(d)
    w = r.x
    if tie:
        bs = w[:p]
        return ((bs, w[p]), (bs, w[p + 1])), w
    return ((w[:p], w[p]), (w[p + 1:2 * p + 1], w[2 * p + 1])), w


def b4_pred(par, X):
    b, i = par
    return np.clip(expit(np.asarray(X, float) @ b + i), 1e-6, 1 - 1e-6)


# ── 运行指纹（GPT §2.3：测试与正式运行都要记录）────────────────
def run_fingerprint(modules=()):
    """返回 Python 可执行文件、各模块 __file__ 与代码 SHA256。**只读，不落盘。**"""
    import hashlib, pathlib, platform, sys
    out = dict(python=sys.executable, platform=platform.platform(),
               numpy=np.__version__)
    for m in modules:
        try:
            f = pathlib.Path(m.__file__).resolve()
            out[f"{m.__name__}__file__"] = str(f)
            out[f"{m.__name__}__sha256"] = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
        except Exception as e:
            out[f"{m.__name__}__error"] = str(e)[:80]
    return out
