# -*- coding: utf-8 -*-
"""Sharing continuum 的模型实现与端点校验（GPT 第 34 轮 §A–§D）。

更名（§A）
    `B4` → **FR-PP（Fused-Ridge Partial Pooling）**。
    目标函数是 **L2 fusion**：
        Σℓ/n + (1/(2Cn))(‖β_A‖² + ‖β_B‖²) + λ‖β_A − β_B‖²
    **不是 Data Shared Lasso**（后者是 L1 的 β^(k)=θ+Δ^(k) 共享＋偏离模型，
    Gross & Tibshirani 2016）。Data Shared Lasso 仅作外部文献定位，不冒充本项目已实现的模型。

新增（§C）
    **G-SI = shared slopes + phenotype-specific intercepts**，属于 *shared* 架构、不属于 partial pooling：
        logit p = α_g + xᵀβ，**β 的 L2 惩罚强度与 sklearn 的 G0 严格一致**（截距不惩罚）。

端点校验（§C/§D）—— 见 `endpoint_report()`，结论如下（数值实测，不是推断）：
    (1) **G-SI 在强制 α_A=α_B 时 ≡ sklearn G0**（设计如此，容差内成立）。
    (2) **FR-PP λ=0 ≡ 逐轨独立 sklearn LR（hard specialist）**（容差内成立）。
    (3) **FR-PP λ=∞ ≠ G-SI，而是 ridge 强度恰好 2 倍的共享模型。**
        这不是 bug，而是 fused-ridge 参数化的代数必然：两轨各自带 (1/(2Cn))‖β_g‖²，
        融合成同一个 β 时两份 ridge 相加 ⇒ ‖β‖²/(Cn)，正是 G0 的两倍。
        ⇒ **(1) 与 (3)=G-SI 不可能同时成立**：任何单参数族都无法在同一个 C 下
        同时精确嵌套「逐轨独立 ridge」与「池化 ridge」，因为两端自由系数块数不同、ridge 总量差 2 倍。
        **处理（已实施）**：保留 FR-PP 原定义（其 λ=0 端点精确等于 hard specialist，见 (2)），
        同时把 **G-SI 与 FR-PP(λ=∞) 都放进共享候选集**，让共享端由两者中更强的那个代表 ——
        这比只放一个更满足 §C 的公平性目标。`fit_gsi(C=C/2)` 可精确复现 FR-PP(λ=∞)，已作为测试项。
"""
import sys, pathlib, warnings
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
HERE = pathlib.Path(__file__).parent; sys.path.insert(0, str(HERE))
import pgss_core as V      # 原为 pgss_validate（带 C 盘导入副作用）；
                           # R37 归档纠错：生产链改用无副作用的 pgss_core。
                           # 二者数值逐位一致，已由 pgss_core_tests 证明（fit_b4 四档 λ 误差 0）

LAMBDA_GRID = list(V.LAMBDA_GRID)          # §B：恢复完整 15 档（含 0 与 inf），不缩回 6 档
NAME_MAP = {"B4": "FR-PP", "B4-CV80": "FR-PP-CV80", "B4-HO-old": "FR-PP-HO-old",
            "B4_partial_pooling_LR": "FR-PP_partial_pooling_LR"}


# ── G-SI：共享斜率 ＋ 表型特异截距（§C）────────────────────
def _gsi_obj(w, X, y, z, n, C):
    p = X.shape[1]
    b = w[:p]; aA, aB = w[p], w[p + 1]
    a = np.where(z, aA, aB)
    lin = X @ b + a
    val = (V._softplus(lin) - y * lin).sum() / n + (b @ b) / (2.0 * C * n)
    r = expit(lin) - y
    g = np.empty_like(w)
    g[:p] = X.T @ r / n + b / (C * n)
    g[p] = r[z].sum() / n; g[p + 1] = r[~z].sum() / n
    return val, g


def _log_diag(diags, stage, program, n):
    """把真实 scipy 诊断登记到引擎 DIAG。引擎未导入时静默跳过（单测直调时）。"""
    try:
        import pgss_r36_engine as _E
    except Exception:
        return
    for d in diags:
        _E.DIAG.append(dict(stage=stage, program=program, tag=d.get("who"),
                            solver="scipy-L-BFGS-B", n=int(n),
                            n_iter=d.get("nit"), max_iter=50000,
                            hit_cap=bool(d.get("hit_maxiter") or d.get("hit_maxfun")),
                            convergence_warning=None, success=bool(d.get("success")),
                            status=d.get("status"), grad_inf_norm=d.get("grad_inf_norm"),
                            accepted_on_gradient=bool(d.get("accepted_on_gradient"))))


def fit_gsi(X, y, z, C=1.0, ftol=1e-16):
    """logit p = α_g + xᵀβ；β 的 ridge = ‖β‖²/(2Cn)（与 sklearn G0 严格一致），截距不惩罚。"""
    X = np.asarray(X, float); y = np.asarray(y, float); z = np.asarray(z, bool)
    n = len(y); p = X.shape[1]
    r = minimize(_gsi_obj, np.zeros(p + 2), args=(X, y, z, n, C), method="L-BFGS-B", jac=True,
                 options=dict(maxiter=50000, maxfun=200000, ftol=ftol, gtol=1e-12))
    _d = V._check_optim(r, "G-SI")
    _log_diag([_d], "outer", "G-SI", n)
    b, aA, aB = r.x[:p], r.x[p], r.x[p + 1]

    def pred(Xn, zn):
        return np.clip(expit(np.asarray(Xn, float) @ b + np.where(np.asarray(zn, bool), aA, aB)),
                       1e-6, 1 - 1e-6)
    return pred, dict(beta=b, alpha_A=float(aA), alpha_B=float(aB), fun=float(r.fun),
                      nit=int(r.nit), optim=_d)


def _g0_tied(X, y, C=1.0, ftol=1e-16):
    """把 G-SI 的两个截距强制相等（即单截距共享模型），用于与 sklearn G0 对照"""
    X = np.asarray(X, float); y = np.asarray(y, float); n = len(y); p = X.shape[1]

    def obj(w):
        b, a = w[:p], w[p]
        lin = X @ b + a
        val = (V._softplus(lin) - y * lin).sum() / n + (b @ b) / (2.0 * C * n)
        r = expit(lin) - y
        g = np.empty_like(w); g[:p] = X.T @ r / n + b / (C * n); g[p] = r.sum() / n
        return val, g
    r = minimize(obj, np.zeros(p + 1), method="L-BFGS-B", jac=True,
                 options=dict(maxiter=50000, maxfun=200000, ftol=ftol, gtol=1e-12))
    V._check_optim(r, "G0-tied")
    b, a = r.x[:p], r.x[p]
    return (lambda Xn: np.clip(expit(np.asarray(Xn, float) @ b + a), 1e-6, 1 - 1e-6))


def frpp(X, y, z, lam, C=1.0, stage="outer"):
    """FR-PP：原 `fit_b4`，仅更名。λ 可为 0 与 inf。

    真实 scipy 退出状态经 `V._check_optim` 校验后登记到引擎 DIAG（全覆盖，非抽样）。
    """
    X = np.asarray(X, float); y = np.asarray(y, float); z = np.asarray(z, bool)
    _d = []
    par, w = V.fit_b4(X[z], y[z], X[~z], y[~z], lam, C=C, diag_out=_d)
    _log_diag(_d, stage, "FR-PP", len(y))
    (ba, ia), (bb, ib) = par

    def pred(Xn, zn):
        Xn = np.asarray(Xn, float); zn = np.asarray(zn, bool)
        return np.clip(np.where(zn, expit(Xn @ ba + ia), expit(Xn @ bb + ib)), 1e-6, 1 - 1e-6)
    return pred, dict(beta_A=ba, beta_B=bb, int_A=float(ia), int_B=float(ib), lam=lam)


# ── 端点校验报告 ───────────────────────────────────────────
def endpoint_report(seed=11, n=1200, p=10, C=1.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)); z = rng.random(n) < 0.4
    lin = 0.7 * X[:, 0] - 0.6 * X[:, 1] + 0.9 * z * X[:, 2] - 0.4 + 0.5 * z
    y = (rng.random(n) < expit(lin)).astype(float)
    out = []
    # (1) G-SI 强制同截距 ≡ sklearn G0
    sk = LogisticRegression(**V.LR_KW); sk.fit(X, y)
    p_sk = np.clip(sk.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
    p_tied = _g0_tied(X, y, C=C)(X)
    out.append(dict(检验="(1) G-SI 强制 α_A=α_B  vs  sklearn G0",
                    max_abs_diff=float(np.abs(p_tied - p_sk).max()), 容差="1e-6",
                    通过=bool(np.abs(p_tied - p_sk).max() < 1e-6)))
    # (2) FR-PP λ=0 ≡ 逐轨独立 sklearn LR
    f0, _ = frpp(X, y, z, 0.0, C=C)
    pa = np.full(n, np.nan)
    for g, m in ((True, z), (False, ~z)):
        s = LogisticRegression(**V.LR_KW); s.fit(X[m], y[m])
        pa[m] = np.clip(s.predict_proba(X[m])[:, 1], 1e-6, 1 - 1e-6)
    d2 = float(np.abs(f0(X, z) - pa).max())
    out.append(dict(检验="(2) FR-PP λ=0  vs  逐轨独立 sklearn LR（hard specialist）",
                    max_abs_diff=d2, 容差="1e-6", 通过=bool(d2 < 1e-6)))
    # (3) FR-PP λ=inf vs G-SI(C) 与 G-SI(C/2)
    finf, _ = frpp(X, y, z, np.inf, C=C)
    p_inf = finf(X, z)
    g1, _ = fit_gsi(X, y, z, C=C); g2, _ = fit_gsi(X, y, z, C=C / 2)
    d3a = float(np.abs(p_inf - g1(X, z)).max()); d3b = float(np.abs(p_inf - g2(X, z)).max())
    out.append(dict(检验="(3a) FR-PP λ=∞  vs  G-SI(C)（GPT 要求的等价）",
                    max_abs_diff=d3a, 容差="1e-6", 通过=bool(d3a < 1e-6)))
    out.append(dict(检验="(3b) FR-PP λ=∞  vs  G-SI(C/2)（ridge 加倍后的 G-SI）",
                    max_abs_diff=d3b, 容差="1e-6", 通过=bool(d3b < 1e-6)))
    return out


if __name__ == "__main__":
    print("LAMBDA_GRID（%d 档）：" % len(LAMBDA_GRID),
          ", ".join("inf" if np.isinf(x) else f"{x:g}" for x in LAMBDA_GRID))
    print("\n=== 端点校验（§C/§D）===")
    for r in endpoint_report():
        print(f"  {'通过' if r['通过'] else '不通过'}  {r['检验']}   max|Δp| = {r['max_abs_diff']:.3e}")
    print("\n解读：(1)(2)(3b) 应通过、(3a) 应不通过 —— 见模块 docstring 的代数说明。")
