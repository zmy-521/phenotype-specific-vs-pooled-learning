# -*- coding: utf-8 -*-
r"""PGSS XGB 主实验引擎 v1.0（SAP v1.0；GPT 第 47 轮三项修正后）。

本模块只定义「学习程序」：给定训练部分 T 与评分部分 V，产生 V 上的预测。
**产生 V 预测所需的插补、排序、面板构建、轮数确定与校准，全部只用 T**。

结构性隔离
  1. 一切内部划分（早停留出、校准交叉拟合、内层 CV、认证划分）**只依赖分组与种子，不看 y**；
     早停与校准划分的种子由 T 的单位集合派生 ⇒ 改动 T 以外任何标签都不可能改变 T 上的拟合产物。
  2. 早停后**一律在 T 上按 best_iteration+1 轮重拟合**，预测只用重拟合模型。本程序用原生 Booster 接口，
     早停模型若不限定迭代范围，默认预测会使用完整模型（U1 核验）。
  3. 校准器只用 T 内三折交叉拟合概率训练，再作用于 V。

v1.0 修正（GPT 第 47 轮）
  A. **缓存行序安全**：程序内部把 T 与 V 统一按 unit_id 升序规范化；缓存保存规范顺序的预测，
     返回前恢复调用方行序。所有缓存键带**任务命名空间**（库/结局/表型/数据指纹/引擎版本），未设置即拒绝运行（U13）。
  B. **全局排序保留表型**：全局排序模型的输入 = 全部 23 个原始列（女性任务的性别列作为强制列在内）+ Z（需追加时）；
     只对可选择的原始候选列排名，Z 与强制列不参与 top-k。SZ-GFS 与 HGFS **共用同一个全局面板函数**，
     原始变量成员与顺序一致；Z 在轨内是常数，专属模型不追加 Z（女性任务的性别列作为面板成员保留在内）（U21、U22）。
  C. **干跑模式**（DRYRUN）：沿用全部真实代码路径，fit_booster 只做检查与划分并记录计划、不训练，
     供执行路径资格审计使用；干跑与真实运行的拟合计划集合一致性由 U23 核验。
"""
import hashlib, json, time
import numpy as np, xgboost as xgb
from sklearn.linear_model import LogisticRegression
import pgss_r36_engine as E

ENGINE_VERSION = "xgb-1.0"
# ── 固定配置（SAP v1.0）──────────────────────────────────────
PARAM_BASE = dict(objective="binary:logistic", tree_method="hist", eta=0.05, subsample=0.8,
                  colsample_bytree=0.8, reg_lambda=1.0, max_bin=256, nthread=1,
                  disable_default_eval_metric=1)
GRID_TRACK = [dict(max_depth=d, min_child_weight=c) for d in (2, 3, 4, 6) for c in (1, 10)]
GRID_POOLED = [dict(max_depth=d, min_child_weight=c) for d in (2, 3, 4, 6, 7) for c in (1, 10)]
RANK_THETA = dict(max_depth=3, min_child_weight=1)
K_GRID = [5, 8, 12, 16, 23]
MAX_ROUNDS, PATIENCE = 2000, 50
ES_FOLDS = 7            # 早停留出 = T 按组 7 折中的 1 折（≈14%）
CAL_FOLDS = 3           # 训练内三折交叉拟合校准
INNER_FOLDS = 5
MIN_FIT = 30
POOLED = ("S0", "SZ", "SZ-GFS", "SZ-union")
TRACKED = ("Hfull", "HTFS", "HGFS")
WEIGHTS = ("none", "bal")

DRYRUN = False          # 干跑：只做检查与划分、记录计划，不训练
RECORD_LEAVES = True
FITLOG = []             # 每次拟合（含干跑计划）一条
CALLOG = []             # 每次 Platt 校准一条
REQLOG = {"program_requests": 0, "program_cache_hits": 0, "rank_requests": 0, "rank_cache_hits": 0,
          "track_requests": 0, "track_cache_hits": 0}
_PCACHE, _RCACHE, _TCACHE = {}, {}, {}
NS = None               # 任务命名空间（set_namespace 设置）


class FoldFailure(RuntimeError):
    """样本不足、单类别、划分不可行等：**必须抛出**，不得静默降级。"""


def set_namespace(**kw):
    """设置任务命名空间（库/结局/表型/数据指纹等）；缓存键全部带它。切换任务必须重设并清缓存。"""
    global NS
    NS = h16("ns", ENGINE_VERSION, kw)
    reset_caches()
    return NS


def reset_caches():
    FITLOG.clear(); CALLOG.clear(); _PCACHE.clear(); _RCACHE.clear(); _TCACHE.clear()
    for k in REQLOG: REQLOG[k] = 0


def h16(*parts):
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, np.ndarray):
            h.update(p.tobytes())
        else:
            h.update(json.dumps(p, ensure_ascii=False, sort_keys=True, default=str).encode())
        h.update(b"|")
    return h.hexdigest()[:16]


def uid_hash(uid):
    """单位**集合**哈希（排序后）。只在程序内部行序已规范化（按 unit_id 升序）后用于缓存键。"""
    return h16(np.ascontiguousarray(np.sort(np.asarray(uid, dtype=np.int64))))


def seed_of(*parts):
    return int(h16(*parts)[:8], 16) % (2 ** 31 - 1)


def _need_ns():
    if NS is None:
        raise RuntimeError("未设置任务命名空间（set_namespace），拒绝使用缓存")


# ── 划分（全部 y 无关）──────────────────────────────────────
def group_split(groups, k, seed):
    """按组、按记录数均衡、**不看 y** 的 k 折；空折即抛错。

    与 `pgss_r36_engine.group_split(..., balance_on_events=False)` **逐位等价**（U20）：
    同样的随机排列；每组投给当前记录数最少的折，平局取编号最小者；用堆实现。
    折的成员只取决于分组标签集合与种子，与行序无关。
    """
    import heapq
    groups = np.asarray(groups)
    uq, inv = np.unique(groups, return_inverse=True)
    sizes = np.bincount(inv, minlength=len(uq))
    order = np.random.default_rng(int(seed)).permutation(len(uq))
    heap = [(0, b) for b in range(k)]
    assign = np.empty(len(uq), dtype=np.int64)
    for gi in order:
        ln, b = heapq.heappop(heap)
        assign[gi] = b
        heapq.heappush(heap, (ln + int(sizes[gi]), b))
    lab = assign[inv]
    folds = [(np.flatnonzero(lab != i), np.flatnonzero(lab == i)) for i in range(k)]
    if any(len(te) == 0 for _, te in folds):
        raise FoldFailure(f"分组切分出现空折（组数 {len(uq)} < 折数 {k}？）")
    for tr, te in folds:
        if np.intersect1d(inv[tr], inv[te]).size:
            raise FoldFailure("分组划分泄漏")
    return folds


def group_codes(groups):
    """字符串分组 → 整数编码（按字符串排序编码 ⇒ 任何子集上 np.unique 的顺序不变，划分逐位相同）。"""
    return np.unique(np.asarray(groups), return_inverse=True)[1].astype(np.int64)


def holdout(groups, k, seed):
    """k 折中的第 0 折作留出：返回 (保留, 留出)。"""
    return group_split(groups, k, seed)[0]


# ── 预处理与权重 ─────────────────────────────────────────────
def impute_fit(X):
    med = np.nanmedian(X, axis=0)
    return np.where(np.isnan(med), 0.0, med)


def impute_apply(X, med):
    return np.where(np.isnan(X), med, X)


def train_weights(z, scheme, W0=None):
    """none → 全 1；bal → w_i = W0/(2·N_g)，W0 缺省为 N（平均权重 1）。单轨不定义均衡权重。"""
    z = np.asarray(z, bool)
    if scheme == "none":
        return np.ones(len(z))
    if scheme != "bal":
        raise ValueError(scheme)
    nA, nB = int(z.sum()), int((~z).sum())
    if nA == 0 or nB == 0:
        raise FoldFailure("均衡权重要求两轨都有记录（0% 借用端点须单独处理）")
    W0 = float(len(z)) if W0 is None else float(W0)
    return np.where(z, W0 / (2 * nA), W0 / (2 * nB))


def es_metric_weights(z, pooled):
    """早停评分：合并臂 = 两轨等权 Brier；专属臂 = 本轨 Brier。"""
    z = np.asarray(z, bool)
    if not pooled:
        return np.ones(len(z))
    nA, nB = int(z.sum()), int((~z).sum())
    if nA == 0 or nB == 0:
        raise FoldFailure("早停留出缺少某一轨，无法计算两轨等权 Brier")
    return np.where(z, 1.0 / (2 * nA), 1.0 / (2 * nB))


def _wbrier(predt, dmat):
    yy = dmat.get_label(); w = dmat.get_weight()
    if w.size == 0:
        w = np.ones_like(yy)
    return "wbrier", float(np.sum(w * (yy - predt) ** 2) / np.sum(w))


# ── 单次拟合：早停（T 内留出）+ 在 T 上重拟合 ─────────────────
class _StubBooster:
    """干跑用：确定性、非常数的预测；SHAP 贡献全 0（排序退化为原始列序）。"""

    def __init__(self, prev):
        self.prev = float(prev)

    def predict(self, dm, pred_contribs=False, **kw):
        X = dm.get_data().toarray() if hasattr(dm.get_data(), "toarray") else np.asarray(dm.get_data())
        if pred_contribs:
            return np.zeros((X.shape[0], X.shape[1] + 1))
        s = X[:, 0] if X.shape[1] else np.zeros(X.shape[0])
        s = (s - np.nanmean(s)) / (np.nanstd(s) + 1e-9)
        return np.clip(self.prev + 0.1 * np.tanh(s), 0.01, 0.99)


def fit_booster(Xc, y, w, z, groups, uid, theta, pooled, role, cols=None):
    """在训练部分上拟合一个 booster，返回**重拟合**模型与记录。

    调用方保证行序已按 unit_id 规范化。早停留出从该训练部分内部按组划出（种子由单位集合派生，不看 y）。
    """
    y = np.asarray(y).astype(int); z = np.asarray(z, bool)
    uh = uid_hash(uid)
    plan = dict(role=role, plan_key=h16("plan", uh, bool(pooled)), pooled=bool(pooled), n=int(len(y)),
                events=int(y.sum()), nonevents=int(len(y) - y.sum()), groups=int(len(np.unique(groups))))
    if pooled:
        plan.update(nA=int(z.sum()), nB=int((~z).sum()), evA=int(y[z].sum()), evB=int(y[~z].sum()))
    if len(y) < MIN_FIT or len(np.unique(y)) < 2:
        raise FoldFailure(f"{role}: 训练样本不足或单类别（n={len(y)}）")
    keep, hold = holdout(groups, ES_FOLDS, seed_of("es", uh))
    yh, zh = y[hold], z[hold]
    plan.update(hold_n=int(len(hold)), hold_events=int(yh.sum()), hold_nonevents=int(len(hold) - yh.sum()))
    if pooled:
        plan.update(hold_evA=int(yh[zh].sum()), hold_evB=int(yh[~zh].sum()),
                    hold_neA=int((1 - yh[zh]).sum()), hold_neB=int((1 - yh[~zh]).sum()))
    if len(np.unique(yh)) < 2 or len(np.unique(y[keep])) < 2:
        raise FoldFailure(f"{role}: 早停留出或保留部分单类别")
    ew = es_metric_weights(zh, pooled)
    if DRYRUN:
        FITLOG.append(plan)
        return _StubBooster(y.mean()), plan
    params = dict(PARAM_BASE, **theta, seed=seed_of("xgb", uh, theta, role.split("|")[0]))
    t0 = time.perf_counter()
    dk = xgb.DMatrix(Xc[keep], label=y[keep], weight=w[keep])
    dh = xgb.DMatrix(Xc[hold], label=yh, weight=ew)
    b_es = xgb.train(params, dk, num_boost_round=MAX_ROUNDS, evals=[(dh, "es")],
                     custom_metric=_wbrier, early_stopping_rounds=PATIENCE, verbose_eval=False)
    best = int(b_es.best_iteration)
    t1 = time.perf_counter()
    b = xgb.train(params, xgb.DMatrix(Xc, label=y, weight=w), num_boost_round=best + 1)
    t2 = time.perf_counter()
    if b.num_boosted_rounds() != best + 1:
        raise FoldFailure("重拟合轮数与 best_iteration+1 不一致")
    leaves = int(sum(t.count("leaf=") for t in b.get_dump())) if RECORD_LEAVES else None
    rec = dict(plan, n_cols=int(Xc.shape[1]), best_iteration=best, rounds=int(b.num_boosted_rounds()),
               es_rounds_trained=int(b_es.num_boosted_rounds()), leaves=leaves,
               sec_es=round(t1 - t0, 4), sec_refit=round(t2 - t1, 4),
               cols=json.dumps(cols) if cols is not None else None, **theta)
    FITLOG.append(rec)
    return b, rec


def predict(b, Xc):
    return np.clip(b.predict(xgb.DMatrix(Xc)), 1e-6, 1 - 1e-6)


# ── 任务上下文：列布局 ────────────────────────────────────────
def task_layout(X, z, feat_names):
    """确定 Z 列的处理：若 Z 与某输入列逐例相同 ⇒ 该列作强制表型列、不复制。"""
    zf = np.asarray(z, float)
    same = [j for j in range(X.shape[1]) if np.array_equal(X[:, j], zf)]
    if same:
        return dict(z_is_col=same[0], z_name=feat_names[same[0]], n_base=X.shape[1])
    return dict(z_is_col=None, z_name="Z", n_base=X.shape[1])


def _design(Xi, z, cols, add_z):
    D = Xi[:, cols]
    return np.hstack([D, np.asarray(z, float)[:, None]]) if add_z else D


# ── 特征排序（平均 |TreeSHAP|）───────────────────────────────
def global_rank(T, wscheme, lay):
    """全局排序：模型输入 = 全部原始列（含强制表型列）+ Z（需追加时）；只对可选择的原始候选列排名。"""
    nb = lay["n_base"]; zc = lay["z_is_col"]
    allc = list(range(nb)); forced = [] if zc is None else [zc]
    pool = [c for c in allc if c not in forced]
    add_z = zc is None
    key = h16("grank", NS, uid_hash(T["uid"]), wscheme, add_z, forced)
    REQLOG["rank_requests"] += 1
    if key in _RCACHE:
        REQLOG["rank_cache_hits"] += 1
        return _RCACHE[key]
    Xd = _design(T["Xi"], T["z"], allc, add_z)
    b, _ = fit_booster(Xd, T["y"], train_weights(T["z"], wscheme), T["z"], T["g"], T["uid"], RANK_THETA,
                       pooled=True, role=f"rank|all|{wscheme}", cols=allc + (["Z"] if add_z else []))
    imp = np.abs(b.predict(xgb.DMatrix(Xd), pred_contribs=True)[:, :-1]).mean(axis=0)[:nb]
    order = [c for c in np.argsort(-imp, kind="stable").tolist() if c in pool]
    _RCACHE[key] = order
    return order


def track_rank(T, g, lay):
    """轨内排序：只用该轨样本、不加权；Z 在轨内为常数，不加入；对全部原始列排名。"""
    allc = list(range(lay["n_base"]))
    key = h16("trank", NS, uid_hash(T["uid"]), g)
    REQLOG["rank_requests"] += 1
    if key in _RCACHE:
        REQLOG["rank_cache_hits"] += 1
        return _RCACHE[key]
    m = T["z"] if g == "A" else ~T["z"]
    Xi = T["Xi"][m][:, allc]
    b, _ = fit_booster(Xi, T["y"][m], np.ones(int(m.sum())), T["z"][m], T["g"][m], T["uid"][m], RANK_THETA,
                       pooled=False, role=f"rank|{g}|none", cols=allc)
    imp = np.abs(b.predict(xgb.DMatrix(Xi), pred_contribs=True)[:, :-1]).mean(axis=0)
    order = [allc[i] for i in np.argsort(-imp, kind="stable")]
    _RCACHE[key] = order
    return order


def global_panel(T, wscheme, k, lay):
    """**唯一的全局面板构建函数**（SZ-GFS 与 HGFS 共用）：强制列 ∪ 全局 top-k，按原始列序。"""
    nb = lay["n_base"]; zc = lay["z_is_col"]
    allc = list(range(nb)); forced = [] if zc is None else [zc]
    pool = [c for c in allc if c not in forced]
    if k >= len(pool):
        return allc
    top = set(global_rank(T, wscheme, lay)[:k])
    return [c for c in allc if c in top or c in forced]


def _cols_for(spec, T, lay):
    """合并臂：返回 (原始列索引, 是否追加 Z 列) —— 只用 T 决定面板。"""
    arch = spec["arch"]; nb = lay["n_base"]; zc = lay["z_is_col"]
    allc = list(range(nb)); forced = [] if zc is None else [zc]
    add_z = zc is None
    if arch == "S0":
        return allc, False
    if arch == "SZ":
        return allc, add_z
    if arch == "SZ-GFS":
        return global_panel(T, spec["w"], spec["k"], lay), add_z
    if arch == "SZ-union":
        a = set(track_rank(T, "A", lay)[:min(spec["kA"], nb)])
        b = set(track_rank(T, "B", lay)[:min(spec["kB"], nb)])
        return [c for c in allc if c in a or c in b or c in forced], add_z
    raise ValueError(arch)


def _track_cols(spec, T, g, lay):
    """专属臂：轨 g 的原始列。HGFS 与 SZ-GFS（不加权）共用 global_panel ⇒ 原始变量成员与顺序一致。"""
    arch = spec["arch"]; allc = list(range(lay["n_base"]))
    if arch == "Hfull":
        return allc
    if arch == "HTFS":
        k = spec["k" + g]
        if k >= len(allc):
            return allc
        top = set(track_rank(T, g, lay)[:k])
        return [c for c in allc if c in top]
    if arch == "HGFS":
        return global_panel(T, "none", spec["kG"], lay)
    raise ValueError(arch)


# ── 学习程序：T → V 预测（T、V 已按 unit_id 规范化）──────────────
def _raw_program(spec, T, V, lay, tracks):
    out = np.full(len(V["z"]), np.nan)
    if spec["arch"] in POOLED:
        cols, add_z = _cols_for(spec, T, lay)
        w = train_weights(T["z"], spec["w"])
        b, _ = fit_booster(_design(T["Xi"], T["z"], cols, add_z), T["y"], w, T["z"], T["g"], T["uid"],
                           spec["theta"], pooled=True, role=f"fit|{spec['arch']}|{spec['w']}",
                           cols=cols + (["Z"] if add_z else []))
        out[:] = predict(b, _design(V["Xi"], V["z"], cols, add_z))
        return out
    for g in tracks:
        mt = T["z"] if g == "A" else ~T["z"]
        mv = V["z"] if g == "A" else ~V["z"]
        if mv.sum() == 0:
            continue
        cols = _track_cols(spec, T, g, lay)
        tkey = h16("trk", NS, g, spec["theta" + g], cols, uid_hash(T["uid"]), uid_hash(V["uid"][mv]))
        REQLOG["track_requests"] += 1
        if tkey in _TCACHE:
            REQLOG["track_cache_hits"] += 1
            out[mv] = _TCACHE[tkey]
            continue
        b, _ = fit_booster(T["Xi"][mt][:, cols], T["y"][mt], np.ones(int(mt.sum())), T["z"][mt], T["g"][mt],
                           T["uid"][mt], spec["theta" + g], pooled=False, role=f"fit|{spec['arch']}|{g}", cols=cols)
        out[mv] = predict(b, V["Xi"][mv][:, cols])
        _TCACHE[tkey] = out[mv].copy()
    return out


def _platt_fit(p, y, z, tracks):
    ab = {}
    for g in tracks:
        m = z if g == "A" else ~z
        if m.sum() < MIN_FIT or len(np.unique(y[m])) < 2:
            raise FoldFailure(f"校准 {g} 轨样本不足或单类别")
        if DRYRUN:
            ab[g] = (0.0, 1.0); continue
        lg = np.log(p[m] / (1 - p[m]))
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000, tol=1e-10).fit(lg[:, None], y[m])
        if int(np.max(lr.n_iter_)) >= 2000:
            raise FoldFailure("校准 LR 未收敛")
        ab[g] = (float(lr.intercept_[0]), float(lr.coef_[0, 0]))
    return ab


def _platt_apply(p, z, ab):
    lg = np.log(p / (1 - p)); out = p.copy()
    for g, (a, b) in ab.items():
        m = z if g == "A" else ~z
        out[m] = 1 / (1 + np.exp(-(a + b * lg[m])))
    return np.clip(out, 1e-6, 1 - 1e-6)


def sub(D, idx):
    return {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in D.items()}


def canon(D):
    """按 unit_id 升序的规范行序。返回 (规范化后的数据, 排序下标)。"""
    o = np.argsort(np.asarray(D["uid"], dtype=np.int64), kind="stable")
    return sub(D, o), o


def program_predict(spec, Traw, Vraw, lay, tracks=("A", "B")):
    """完整学习程序：插补（T 中位数）→ 面板（T）→ 拟合（T 内早停+重拟合）→ 可选校准（T 内三折）→ V 预测。

    行序安全：内部把 T、V 规范化为 unit_id 升序；缓存保存规范顺序的预测；返回值按**调用方的 V 行序**。
    """
    _need_ns()
    Tc, _ = canon(Traw); Vc, vo = canon(Vraw)
    key = h16("prog", NS, spec, uid_hash(Tc["uid"]), uid_hash(Vc["uid"]), list(tracks))
    REQLOG["program_requests"] += 1
    if key in _PCACHE:
        REQLOG["program_cache_hits"] += 1
        pc = _PCACHE[key]
    else:
        pc = _program_canon(spec, Tc, Vc, lay, tracks)
        _PCACHE[key] = pc.copy()
    out = np.empty(len(pc)); out[vo] = pc
    return out


def _program_canon(spec, Tc, Vc, lay, tracks):
    med = impute_fit(Tc["X"])
    T = dict(Tc, Xi=impute_apply(Tc["X"], med)); V = dict(Vc, Xi=impute_apply(Vc["X"], med))
    raw_spec = dict(spec, cal=False)
    p = _raw_program(raw_spec, T, V, lay, tracks)
    if spec.get("cal"):
        folds = group_split(Tc["g"], CAL_FOLDS, seed_of("cal", uid_hash(Tc["uid"])))
        oof = np.full(len(Tc["y"]), np.nan)
        for tr, te in folds:
            oof[te] = program_predict(raw_spec, sub(Tc, tr), sub(Tc, te), lay, tracks)
        need = np.zeros(len(oof), bool)
        for g in tracks:
            need |= (Tc["z"] if g == "A" else ~Tc["z"])
        if np.isnan(oof[need]).any():
            raise FoldFailure("校准 OOF 概率缺失")
        ab = _platt_fit(oof[need], Tc["y"][need], Tc["z"][need], tracks)
        CALLOG.append(dict(spec=json.dumps(raw_spec, sort_keys=True), T=uid_hash(Tc["uid"]),
                           **{f"a_{g}": v[0] for g, v in ab.items()}, **{f"b_{g}": v[1] for g, v in ab.items()}))
        ok = ~np.isnan(p)
        p[ok] = _platt_apply(p[ok], Vc["z"][ok], ab)
    return p


# ── 评分 ────────────────────────────────────────────────────
def brier_bal(y, p, z):
    z = np.asarray(z, bool)
    return 0.5 * (np.mean((y[z] - p[z]) ** 2) + np.mean((y[~z] - p[~z]) ** 2))


def brier_track(y, p, z, g):
    m = z if g == "A" else ~z
    return float(np.mean((y[m] - p[m]) ** 2))


# ── 开发集上的完整开发（内层 CV 调参 + 选择 P*）───────────────
def develop(D, lay, seed, arms_pooled=POOLED, arms_tracked=TRACKED):
    """在开发集 D 上完成两阶段调参、校准选择与 P* 选择；返回各臂冻结规格与内层分数表。

    超参数、k_A/k_B、校准开关是开发期的顺序搜索，复用同一内层 CV；内层分数含选择乐观，不作为独立性能。
    """
    folds = group_split(D["g"], INNER_FOLDS, seed)
    rows = []

    def oof(spec, tracks=("A", "B")):
        o = np.full(len(D["y"]), np.nan)
        for tr, te in folds:
            o[te] = program_predict(spec, sub(D, tr), sub(D, te), lay, tracks)
        return o

    def sc_pooled(spec):
        s = brier_bal(D["y"], oof(spec), D["z"])
        rows.append(dict(scope="pooled", spec=json.dumps(spec, ensure_ascii=False, sort_keys=True), score=s))
        return s

    def sc_track(spec, g):
        s = brier_track(D["y"], oof(spec, (g,)), D["z"], g)
        rows.append(dict(scope=f"track{g}", spec=json.dumps(spec, ensure_ascii=False, sort_keys=True), score=s))
        return s

    specs = {}
    k_gfs = {}
    kk = {"A": 23, "B": 23}
    th = {}
    if arms_tracked:
        for g in ("A", "B"):
            th[g] = min(GRID_TRACK, key=lambda t: sc_track(dict(arch="Hfull", **{"theta" + g: t}), g))
        kk = {}
        for g in ("A", "B"):
            cand = {}
            for k in K_GRID:
                s = dict(arch="Hfull", **{"theta" + g: th[g]}) if k == 23 else \
                    dict(arch="HTFS", **{"theta" + g: th[g], "k" + g: k})
                cand[k] = sc_track(s, g)
            kk[g] = min(K_GRID, key=lambda k: (cand[k], -k))   # 并列取较大 k（不压缩优先）
        specs["Hfull"] = dict(arch="Hfull", thetaA=th["A"], thetaB=th["B"])
        specs["HTFS"] = dict(arch="HTFS", thetaA=th["A"], thetaB=th["B"], kA=kk["A"], kB=kk["B"])
    for w in WEIGHTS:
        tS0 = min(GRID_POOLED, key=lambda t: sc_pooled(dict(arch="S0", w=w, theta=t)))
        specs[f"S0|{w}"] = dict(arch="S0", w=w, theta=tS0)
        if lay["z_is_col"] is not None:
            tSZ = tS0                       # 女性任务：SZ 的输入、配置、权重与 S0 完全相同 ⇒ 规格就是 S0
            specs[f"SZ|{w}"] = dict(arch="S0", w=w, theta=tS0)
        else:
            tSZ = min(GRID_POOLED, key=lambda t: sc_pooled(dict(arch="SZ", w=w, theta=t)))
            specs[f"SZ|{w}"] = dict(arch="SZ", w=w, theta=tSZ)
        sz_spec = specs[f"SZ|{w}"]
        if "SZ-GFS" in arms_pooled:
            cand = {}
            for k in K_GRID:
                s = sz_spec if k == 23 else dict(arch="SZ-GFS", w=w, theta=tSZ, k=k)
                cand[k] = sc_pooled(s)
            kg = min(K_GRID, key=lambda k: (cand[k], -k))
            k_gfs[w] = kg
            specs[f"SZ-GFS|{w}"] = dict(sz_spec) if kg == 23 else dict(arch="SZ-GFS", w=w, theta=tSZ, k=kg)
        if "SZ-union" in arms_pooled and arms_tracked:
            specs[f"SZ-union|{w}"] = dict(arch="SZ-union", w=w, theta=tSZ, kA=kk["A"], kB=kk["B"])
    if "HGFS" in arms_tracked:
        specs["HGFS"] = dict(arch="HGFS", thetaA=th["A"], thetaB=th["B"], kG=k_gfs.get("none", 23))
    final, inner = {}, {}
    for name, s in specs.items():
        if s["arch"] in POOLED:
            r = sc_pooled(dict(s, cal=False)); c = sc_pooled(dict(s, cal=True))
        else:
            r = brier_bal(D["y"], oof(dict(s, cal=False)), D["z"])
            c = brier_bal(D["y"], oof(dict(s, cal=True)), D["z"])
            rows.append(dict(scope="tracked-bal", spec=json.dumps(dict(s, cal=False), sort_keys=True), score=r))
            rows.append(dict(scope="tracked-bal", spec=json.dumps(dict(s, cal=True), sort_keys=True), score=c))
        final[name] = dict(s, cal=bool(c < r)); inner[name] = min(r, c)
    order = [f"{a}|{w}" for a in POOLED for w in WEIGHTS if f"{a}|{w}" in final]
    pstar = min(order, key=lambda n: (inner[n], order.index(n)))
    return dict(specs=final, inner=inner, pstar=pstar, rows=rows,
                inner_folds=[(len(tr), len(te)) for tr, te in folds])


def final_predict(specs, D, Test, lay, names=None):
    """对每个冻结臂同时给出 {raw, cal} 两种测试预测，并标注被选的版本。"""
    out = {}
    for name, s in specs.items():
        if names is not None and name not in names:
            continue
        out[name] = dict(raw=program_predict(dict(s, cal=False), D, Test, lay),
                         cal=program_predict(dict(s, cal=True), D, Test, lay),
                         chosen="cal" if s["cal"] else "raw")
    return out


# ── 认证（fit–certify–test 参考实现）───────────────────────────
def cert_split(g_dev, seed, f_inv=3):
    """开发集按组划出认证集 C（三折之一）与拟合集 F；只看分组与种子。"""
    keep, hold = holdout(g_dev, f_inv, seed)
    return keep, hold


def cert_test(y, p_ref, p_s, z, clusters, m, track, ref="normal"):
    """逐轨单侧检验 H0: Δ_g ≤ m·R_g(P*)；配对量 d=(1−m)ℓ_P* − ℓ_S，簇级比率估计。"""
    from scipy.stats import norm, t as tdist
    msk = z if track == "A" else ~z
    lp = (y[msk] - p_ref[msk]) ** 2; ls = (y[msk] - p_s[msk]) ** 2
    d = (1 - m) * lp - ls
    codes, inv = np.unique(np.asarray(clusters)[msk], return_inverse=True)
    Dc = np.bincount(inv, weights=d); nc = np.bincount(inv).astype(float)
    G = len(codes); tot = nc.sum(); dbar = Dc.sum() / tot
    se = float(np.sqrt(G / (G - 1) * np.sum((Dc - dbar * nc) ** 2)) / tot) if G > 1 else np.nan
    stat = dbar / se if se and se > 0 else np.nan
    pv = float(1 - (norm.cdf(stat) if ref == "normal" else tdist.cdf(stat, G - 1))) if np.isfinite(stat) else np.nan
    kish = float(nc.sum() ** 2 / np.sum(nc ** 2))
    return dict(track=track, n=int(msk.sum()), events=int(y[msk].sum()), clusters=G, kish=round(kish, 2),
                max_cluster_share=round(float(nc.max() / tot), 4), R_ref=float(lp.mean()), R_s=float(ls.mean()),
                dbar=float(dbar), se=se, stat=float(stat) if np.isfinite(stat) else np.nan, p=pv, ref=ref)


# ── 学习曲线：开发集的按组嵌套子集（审计与正式运行共用同一定义）──────
LC_FRACS = (0.25, 0.50, 0.75)


def nested_fraction(groups, frac, seed):
    """按组随机顺序累积到 frac 比例的记录（嵌套：小比例 ⊂ 大比例）；返回开发集内的位置下标。种子 = seed*10+外层折。"""
    uq, inv = np.unique(np.asarray(groups), return_inverse=True)
    sizes = np.bincount(inv)
    order = np.random.default_rng(int(seed)).permutation(len(uq))
    cum = np.cumsum(sizes[order]); target = frac * len(groups)
    take = order[: int(np.searchsorted(cum, target) + 1)]
    return np.flatnonzero(np.isin(inv, take))
