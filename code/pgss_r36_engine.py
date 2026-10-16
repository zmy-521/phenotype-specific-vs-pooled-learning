# -*- coding: utf-8 -*-
r"""R36 引擎：修正 GPT 第 36 轮 §1 与 §2（旧引擎 pgss_r35_engine 保留不覆盖）。

§1 两处修正
  (a) **分组贯穿到校准层**：`fit_all` 现在接收 `groups`，内部 2 折交叉拟合按分组切分
      （原实现是 `rng.permutation(n) % 2`，同一患者可同时进校准的训练侧与验证侧）。
  (b) **校准基础程序与部署程序一致**：每个校准训练半折先用其**全部训练记录**拟合
      *共同* 预处理，再在共同坐标下训练 G0 / G2 / 各轨 specialist。
      原实现里 Scf 的内部基础模型用 `prep_fit(Xraw[组内])`，而部署的 specialist 用
      整个训练折的共同坐标 —— 带 L2 惩罚时这不是同一个学习程序。

§2 可审计性
  `fit_all` 与 `run_fold` 返回**实际参与各步的记录集合与分组集合**，交给测试核验：
      prep_fit 用到的行索引 / 每个基础模型训练的分组集合 /
      每个校准预测的分组集合 / 每条记录获得 OOF 概率的次数
  纪律：不允许只靠「函数名里有没有 Group」判定通过。

§2.3 无路径副作用：只从 `pgss_core` 取数值与拟合函数，不导入 `pgss_validate`。
"""
import hashlib, pathlib, sys, warnings
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold as _SGKF
except ImportError:
    _SGKF = None
HERE = pathlib.Path(__file__).parent; sys.path.insert(0, str(HERE))
import pgss_core as C
import pgss_sharing as SH

SHARED_CAND = ["G0", "G1cf", "G2", "G2cf", "G-SI", "FRPPinf"]
HARD = ["Hard-A", "Hard-B", "Hard-both"]
PROGS = SHARED_CAND + ["SHARED+", "FR-PP"] + HARD
ORACLE = "ORACLE-TEST-LOWER-BOUND"
K_INNER = 5
K_CAL = 2                      # 校准交叉拟合折数：保持 2 折，不新增折数搜索
MIN_FIT = C.MIN_FIT


class FoldFailure(RuntimeError):
    """显式失败：收敛失败 / 单类别 / 候选缺失一律抛出，不静默跳过"""


# -- 真实拟合诊断收集器（GPT 第 39 轮：不得用抽样或字面值代替全覆盖记录）--
# R38 的收敛诊断把 success=True / hit_cap=False 写成字面值，且只抽样 4 个单元。
# 这里在**正式 benchmark 的 worker 内**记录每一次拟合的真实退出状态。
DIAG = []


def diag_reset():
    DIAG.clear()


def diag_take():
    """取出并清空。返回 (记录列表, 汇总)"""
    rec = list(DIAG); DIAG.clear()
    if not rec:
        return rec, {}
    import pandas as _pd
    D = _pd.DataFrame(rec)
    g = _pd.to_numeric(D.get("grad_inf_norm"), errors="coerce")
    it = _pd.to_numeric(D.get("n_iter"), errors="coerce")
    sc = D.get("success")
    return rec, dict(
        n_fits=int(len(D)),
        n_hit_cap=int(D["hit_cap"].fillna(False).sum()) if "hit_cap" in D else 0,
        n_conv_warn=int(D["convergence_warning"].fillna(False).sum())
        if "convergence_warning" in D else 0,
        n_optim_not_success=int((~sc.fillna(True)).sum()) if sc is not None else 0,
        max_grad_inf_norm=(float(g.max()) if g.notna().any() else None),
        max_n_iter=(int(it.max()) if it.notna().any() else None),
        stages=sorted(set(D["stage"].dropna().tolist())) if "stage" in D else [],
        programs=sorted(set(D["program"].dropna().tolist())) if "program" in D else [])


def thread_state_now():
    """在**当前 worker 进程内**读取实际线程池状态（不是事后另起进程诊断）"""
    import os as _os
    keys = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    out = dict(env={k: _os.environ.get(k) for k in keys})
    try:
        import threadpoolctl
        info = threadpoolctl.threadpool_info()
        out["threadpool"] = [dict(api=d["internal_api"], threads=d["num_threads"]) for d in info]
        out["all_single_threaded"] = all(d["num_threads"] == 1 for d in info)
    except Exception as e:
        out["threadpool"] = "threadpoolctl unavailable: %s" % e
        out["all_single_threaded"] = None
    return out


def _lr(Z, y, tag="", stage="outer", program="LR"):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        raise FoldFailure(f"{tag} 单类别 y（n={len(y)}，唯一值={np.unique(y).tolist()}）")
    if len(y) < MIN_FIT:
        raise FoldFailure(f"{tag} 样本不足 n={len(y)} < {MIN_FIT}")
    m = LogisticRegression(**C.LR_KW)
    with warnings.catch_warnings(record=True) as _w:
        warnings.simplefilter("always")
        m.fit(Z, y)
        cw = any(issubclass(x.category, ConvergenceWarning) for x in _w)
    it = np.asarray(getattr(m, "n_iter_", [0]))
    cap = C.LR_KW.get("max_iter", 3000)
    DIAG.append(dict(stage=stage, program=program, tag=tag, solver="sklearn-lbfgs",
                     n=int(len(y)), n_iter=int(it.max()), max_iter=int(cap),
                     hit_cap=bool(np.any(it >= cap)), convergence_warning=bool(cw),
                     success=bool(not np.any(it >= cap) and not cw),
                     status=None, grad_inf_norm=None))
    if np.any(it >= cap):
        raise FoldFailure(f"{tag} LR 未收敛（n_iter={it.tolist()}）")
    if cw:
        raise FoldFailure(f"{tag} LR 触发 ConvergenceWarning（n_iter={it.tolist()}）")
    return m


def _p(m, Z):
    return np.clip(m.predict_proba(Z)[:, 1], 1e-6, 1 - 1e-6)


def _platt(pcf, y, z, tag=""):
    """逐轨 (a,b)；拟合失败显式抛出，不退回 (0,1)"""
    out = {}
    for g, msk in (("A", z), ("B", ~z)):
        ok = msk & np.isfinite(pcf)
        if ok.sum() < MIN_FIT or len(np.unique(y[ok])) < 2:
            raise FoldFailure(f"{tag} {g} 轨重校准样本不足/单类别（n={int(ok.sum())}）")
        lg = np.log(pcf[ok] / (1 - pcf[ok]))
        mm = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000, tol=1e-9)
        with warnings.catch_warnings(record=True) as _w:
            warnings.simplefilter("always")
            mm.fit(lg[:, None], y[ok])
            cw = any(issubclass(x.category, ConvergenceWarning) for x in _w)
        it = int(np.asarray(getattr(mm, "n_iter_", [0])).max())
        DIAG.append(dict(stage="calibration", program="platt", tag=f"{tag}/{g}",
                         solver="sklearn-lbfgs", n=int(ok.sum()), n_iter=it, max_iter=2000,
                         hit_cap=bool(it >= 2000), convergence_warning=bool(cw),
                         success=bool(it < 2000 and not cw), status=None, grad_inf_norm=None))
        if it >= 2000 or cw:
            raise FoldFailure(f"{tag} {g} 轨重校准 LR 未收敛（n_iter={it}, warn={cw}）")
        out[g] = (float(mm.intercept_[0]), float(mm.coef_[0, 0]))
    return out


def _apply_ab(p, z, ab):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lg = np.log(p / (1 - p))
    a = np.where(z, ab["A"][0], ab["B"][0]); b = np.where(z, ab["A"][1], ab["B"][1])
    return np.clip(1.0 / (1.0 + np.exp(-(a + b * lg))), 1e-6, 1 - 1e-6)


def group_split(X, y, z, groups, k, seed, balance_on_events=True):
    """分组切分：**随机顺序 + 最轻折优先**（GPT §3 要求的「明确的随机分组算法」）。

    为什么不用 `GroupKFold` 或 `StratifiedGroupKFold`
        两者都先按组大小（或标签分布）排序再贪心分配。当**组数少且尺寸悬殊**时
        （真实 eICU：64 个连通分量，最大的占 10.1%），排序主导了结果：
          · `GroupKFold` 无随机性，重命名组名只能改变平局顺序 ⇒ 真实数据上三套 seed 完全相同
          · `StratifiedGroupKFold(shuffle=True)` 的 shuffle 只打乱初始顺序，
            仍被尺寸排序抹掉 ⇒ 实测同一数据上三套 seed 的 ARI 仍为 1.000
        因此两者都不能作为「划分稳定性重复」的随机来源。

    本实现
        1. 对分组做**随机排列**（随机性作用在顺序上，不被尺寸排序抹掉）
        2. 按该随机顺序，把每个组投给当前「负载最轻」的折
        3. 负载同时计记录数与事件数（事件数优先），使各折规模与事件率都可比
    `balance_on_events=False` 时**只按记录数均衡、完全不看 y**。
        迁移的校准层必须用它：GPT 第 36 轮要求「改被评价簇的结局不得改变用于它的截距」，
        而按事件数均衡会让分配依赖 y ⇒ 该保证只能靠运气。关掉后保证是结构性的。
    `pgss_r36_tests.m4` 在真实 eICU 分量上核验：不同 seed 的同折关系 ARI 必须 < 0.5。
    """
    groups = np.asarray(groups)
    yv = np.asarray(y).astype(int)
    uq, inv = np.unique(groups, return_inverse=True)
    sizes = np.bincount(inv, minlength=len(uq)).astype(np.int64)
    events = (np.bincount(inv, weights=yv, minlength=len(uq)).astype(np.int64)
              if balance_on_events else np.zeros(len(uq), dtype=np.int64))
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(uq))                 # ← 唯一的随机来源，不再排序
    load_n = np.zeros(k, dtype=np.int64); load_e = np.zeros(k, dtype=np.int64)
    assign = np.empty(len(uq), dtype=int)
    for gi in order:
        b = int(np.lexsort((load_n, load_e))[0])     # 事件数优先、记录数次之，取最轻折
        assign[gi] = b; load_n[b] += sizes[gi]; load_e[b] += events[gi]
    lab = assign[inv]
    folds = [(np.flatnonzero(lab != i), np.flatnonzero(lab == i)) for i in range(k)]
    if any(len(te) == 0 for _, te in folds):
        raise FoldFailure(f"分组切分出现空折（组数 {len(uq)} < 折数 {k}？）")
    return folds


def fit_all(Xraw, y, zz, groups, seed, lam=None):
    """在一个训练集上拟合全部候选。

    - 同一训练集只拟合一次 prep，所有候选共用（FR-PP λ=0 与独立 specialist 严格可比）
    - 校准层按 `groups` 切分（§1a）
    - 每个校准训练半折用其**全部**记录拟合共同 prep，再训 G0/G2/specialist（§1b）
    - 返回 `audit`：各步实际用到的行索引与分组集合，供行为测试核验（§2）
    """
    Xraw = np.asarray(Xraw, float); y = np.asarray(y).astype(int)
    zz = np.asarray(zz, bool); groups = np.asarray(groups)
    n = len(y)
    audit = dict(prep_rows=np.arange(n), cal_folds=[], oof_count=np.zeros(n, dtype=int))

    prep = C.prep_fit(Xraw)
    Z = C.prep_apply(Xraw, prep)
    D2 = C.d2(Z, zz)
    m0 = _lr(Z, y, "G0"); m2 = _lr(D2, y, "G2")
    gsi, _ = SH.fit_gsi(Z, y, zz, C=1.0)
    finf, _ = SH.frpp(Z, y, zz, np.inf, C=1.0)
    ms = {g: _lr(Z[m], y[m], f"S_{g}") for g, m in (("A", zz), ("B", ~zz))}
    frp = None
    if lam is not None:
        frp, _ = SH.frpp(Z, y, zz, lam, C=1.0)

    # ── 校准层：按分组切 2 折；每折用该折**全部**训练记录拟合共同 prep ──
    cal = group_split(Xraw, y, zz, groups, K_CAL, seed + 77)
    pcf0 = np.full(n, np.nan); pcf2 = np.full(n, np.nan); pcfS = np.full(n, np.nan)
    for h, (tr, te) in enumerate(cal):
        gtr = set(np.unique(groups[tr]).tolist()); gte = set(np.unique(groups[te]).tolist())
        if gtr & gte:
            raise FoldFailure(f"校准折 {h} 分组泄漏（交集 {len(gtr & gte)} 个）")
        pr = C.prep_fit(Xraw[tr])                      # §1b：该折全部训练记录拟合共同 prep
        Ztr = C.prep_apply(Xraw[tr], pr); Zte = C.prep_apply(Xraw[te], pr)
        pcf0[te] = _p(_lr(Ztr, y[tr], f"cal{h} G0"), Zte)
        pcf2[te] = _p(_lr(C.d2(Ztr, zz[tr]), y[tr], f"cal{h} G2"), C.d2(Zte, zz[te]))
        track_prep = {}
        for g, mg in (("A", zz), ("B", ~zz)):          # specialist 用**同一共同坐标**
            a = np.intersect1d(tr, np.flatnonzero(mg)); b = np.intersect1d(te, np.flatnonzero(mg))
            if len(a) < MIN_FIT or len(b) == 0 or len(np.unique(y[a])) < 2:
                raise FoldFailure(f"cal{h} Scf {g} 轨样本不足（tr={len(a)}, te={len(b)}）")
            pr_g = pr                                   # ← 实际用于该轨的 prep
            Za = C.prep_apply(Xraw[a], pr_g); Zb = C.prep_apply(Xraw[b], pr_g)
            pcfS[b] = _p(_lr(Za, y[a], f"cal{h} S_{g}"), Zb)
            track_prep[g] = tuple(np.asarray(x).copy() for x in pr_g)
        audit["oof_count"][te] += 1
        # R37：记录**实际用到的 prep 参数值**，而不仅是声明的行号。
        # 原实现只记 prep_rows=tr，于是把 prep 改成组内拟合后 M2 仍会通过
        # （GPT 第 37 轮用真实变异证明了这一点）。现在测试可按声明行号重算并比对数值。
        audit["cal_folds"].append(dict(fold=h, n_fit=int(len(tr)), n_eval=int(len(te)),
                                       prep_rows=tr.copy(), groups_fit=sorted(gtr),
                                       groups_eval=sorted(gte),
                                       prep_values=tuple(np.asarray(x).copy() for x in pr),
                                       track_prep=track_prep))
    if np.isnan(pcf0).any() or np.isnan(pcf2).any() or np.isnan(pcfS).any():
        raise FoldFailure("有记录未获得校准 OOF 概率")
    if not (audit["oof_count"] == 1).all():
        raise FoldFailure(f"校准 OOF 次数异常：min={audit['oof_count'].min()}, "
                          f"max={audit['oof_count'].max()}（须恰好 1）")
    ab1 = _platt(pcf0, y, zz, "G1cf"); ab2 = _platt(pcf2, y, zz, "G2cf")
    abS = _platt(pcfS, y, zz, "Scf")

    def predict(Xn_raw, zn):
        Xn_raw = np.asarray(Xn_raw, float); zn = np.asarray(zn, bool)
        Zn = C.prep_apply(Xn_raw, prep)
        p0 = _p(m0, Zn); p2 = _p(m2, C.d2(Zn, zn))
        sA = _p(ms["A"], Zn); sB = _p(ms["B"], Zn)
        o = {"G0": p0, "G1cf": _apply_ab(p0, zn, ab1),
             "G2": p2, "G2cf": _apply_ab(p2, zn, ab2),
             "G-SI": gsi(Zn, zn), "FRPPinf": finf(Zn, zn),
             "S_A_raw": sA, "S_B_raw": sB,
             "S_A_cf": _apply_ab(sA, np.ones(len(zn), bool), abS),
             "S_B_cf": _apply_ab(sB, np.zeros(len(zn), bool), abS)}
        o["S_raw"] = np.where(zn, sA, sB)
        o["S_cf"] = np.where(zn, o["S_A_cf"], o["S_B_cf"])
        if frp is not None: o["FR-PP"] = frp(Zn, zn)
        return o
    return predict, audit


def split_fingerprint(groups, folds):
    """划分指纹（§3）：**不受折号重命名影响** —— 用「同折关系」的规范形。

    对每个分组计算其所在折，再把折按「最小分组 id」重排成规范标签，
    最后对规范标签哈希。折号整体轮换不改变该哈希。
    """
    groups = np.asarray(groups)
    lab = np.full(len(groups), -1)
    for k, (_, te) in enumerate(folds): lab[te] = k
    uq = np.unique(groups)
    gl = {g: int(lab[groups == g][0]) for g in uq}
    order = {}
    for g in sorted(gl, key=lambda x: str(x)):
        order.setdefault(gl[g], len(order))
    canon = np.array([order[gl[g]] for g in groups])
    return hashlib.sha256(canon.tobytes()).hexdigest()[:16], canon


def coassign_agreement(canon_a, canon_b):
    """两套划分的「同折关系」一致度（§3：不受折号重命名影响）。

    返回 (一致比例, adjusted Rand index)。随机水平下 ARI ≈ 0；完全相同 ARI = 1。
    """
    from itertools import combinations
    a = np.asarray(canon_a); b = np.asarray(canon_b)
    n = len(a)
    if n > 4000:                                   # 子抽样控制 O(n²)
        idx = np.random.default_rng(0).choice(n, 4000, replace=False)
        a, b = a[idx], b[idx]; n = 4000
    sa = (a[:, None] == a[None, :]); sb = (b[:, None] == b[None, :])
    iu = np.triu_indices(n, 1)
    agree = float((sa[iu] == sb[iu]).mean())
    try:
        from sklearn.metrics import adjusted_rand_score
        ari = float(adjusted_rand_score(a, b))
    except Exception:
        ari = float("nan")
    return agree, ari


def run_fold(Xd_raw, yd, zd, gd, Xt_raw, zt, seed):
    """一个 outer fold：内部 OOF 同时选 SHARED+ / λ* / 每轨 specialist 版本。

    外层 test 不参与任何选择。全部内部切分（模型选择与校准）都按 `gd` 分组。
    """
    Xd_raw = np.asarray(Xd_raw, float); Xt_raw = np.asarray(Xt_raw, float)
    yd = np.asarray(yd).astype(int); zd = np.asarray(zd, bool); zt = np.asarray(zt, bool)
    gd = np.asarray(gd)
    n = len(yd)
    folds = group_split(Xd_raw, yd, zd, gd, K_INNER, seed)
    for h, (tr, te) in enumerate(folds):
        if set(np.unique(gd[tr]).tolist()) & set(np.unique(gd[te]).tolist()):
            raise FoldFailure(f"内部 CV 折 {h} 分组泄漏")
    sh, canon = split_fingerprint(gd, folds)
    keys = SHARED_CAND + ["S_A_raw", "S_B_raw", "S_A_cf", "S_B_cf"]
    oof = {k: np.full(n, np.nan) for k in keys}
    oof_lam = {lam: np.full(n, np.nan) for lam in C.LAMBDA_GRID}
    cov = np.zeros(n, dtype=int)
    inner_audit = []
    for kf, (tr, va) in enumerate(folds):
        pred, aud = fit_all(Xd_raw[tr], yd[tr], zd[tr], gd[tr], seed * 10 + kf)
        P = pred(Xd_raw[va], zd[va])
        for k in keys: oof[k][va] = P[k]
        cov[va] += 1
        prep = C.prep_fit(Xd_raw[tr])
        Ztr = C.prep_apply(Xd_raw[tr], prep); Zva = C.prep_apply(Xd_raw[va], prep)
        for lam in C.LAMBDA_GRID:
            try:
                f, _ = SH.frpp(Ztr, yd[tr], zd[tr], lam, C=1.0, stage="inner")
                oof_lam[lam][va] = f(Zva, zd[va])
            except Exception as e:
                raise FoldFailure(f"内部折 {kf} λ={lam} 拟合失败：{str(e)[:100]}")
        inner_audit.append(dict(fold=kf, n_tr=int(len(tr)), n_va=int(len(va)),
                                groups_tr=int(len(np.unique(gd[tr]))),
                                groups_va=int(len(np.unique(gd[va]))),
                                cal=aud["cal_folds"]))
    if not (cov == 1).all():
        raise FoldFailure(f"内部 OOF 覆盖异常：min={cov.min()}, max={cov.max()}")
    rs = {a: C.two_track(yd, zd, oof[a]) for a in SHARED_CAND}
    shared_plus = min(rs, key=rs.get)
    spec_ver = {}
    for g, mg in (("A", zd), ("B", ~zd)):
        cand = {v: float(C.brier(yd[mg], oof[f"S_{g}_{v}"][mg]).mean()) for v in ("raw", "cf")}
        spec_ver[g] = min(cand, key=cand.get)
    lam_star = min(oof_lam, key=lambda L: C.two_track(yd, zd, oof_lam[L]))
    pred, aud = fit_all(Xd_raw, yd, zd, gd, seed + 1, lam=lam_star)
    PT = pred(Xt_raw, zt)
    pstar = PT[shared_plus]
    sA = PT[f"S_A_{spec_ver['A']}"]; sB = PT[f"S_B_{spec_ver['B']}"]
    out = {a: PT[a] for a in SHARED_CAND}
    out["SHARED+"] = pstar; out["FR-PP"] = PT["FR-PP"]
    out["Hard-A"] = np.where(zt, sA, pstar)
    out["Hard-B"] = np.where(zt, pstar, sB)
    out["Hard-both"] = np.where(zt, sA, sB)
    info = dict(shared_plus=shared_plus,
                lam_star=("inf" if not np.isfinite(lam_star) else f"{lam_star:g}"),
                spec_version=f"A={spec_ver['A']},B={spec_ver['B']}",
                inner_split_hash=sh, n_dev=int(n), n_inner_folds=len(folds),
                n_groups_dev=int(len(np.unique(gd))))
    return out, info, dict(inner=inner_audit, refit=aud, canon=canon)


def oracle_lower_bound(preds, y, z, avail):
    """测试集择优下界：min_j R_test(G_j)。**含择优乐观性，仅诊断，不可部署。**"""
    orc = np.full(len(y), np.nan)
    for gm0 in (z, ~z):
        gm = gm0 & avail
        if gm.sum() == 0: continue
        best, bv = None, np.inf
        for a in SHARED_CAND:
            if a not in preds or np.isnan(preds[a][gm]).any(): continue
            v = float(C.brier(y[gm], preds[a][gm]).mean())
            if v < bv: bv, best = v, a
        if best is not None: orc[gm] = preds[best][gm]
    return orc
