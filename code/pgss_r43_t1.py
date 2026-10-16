# -*- coding: utf-8 -*-
r"""R43 第二部分：有限规模 T1（GPT 第 48 轮 §4）——已知机制下冻结 XGB 程序与选择器的工作特性。

范围（本批次固定矩阵）
  机制 N1、N2、P1、P2（公式与参数**直接取自** `pgss_xgb_sim0.py` 的 gen_X / pstar，代码指纹记入配置）；
  开发集 n ∈ {6000, 12000}；A 轨比例 π_A ∈ {0.15, 0.30}；每条件 20 个独立训练重复 ⇒ 16 条件 × 20 = 320 个重复。
  可选（**须 GPT 批准**，默认不跑）：N1-T、P2-T —— Z = 1[X₈ > Φ⁻¹(1−π_A)]，其余同 N1、P2。
设计（本批次登记，区别于 R41 Stage 0B）
  训练与评价**均无人为缺失**；分组单位 = 独立个体；评价集 100,000 个独立样本（≥ GPT 要求的 40,000）。
  学习程序 = R42 冻结引擎 xgb-1.0（同一网格、同一候选与搜索规则）。
真值（只进评价器）
  条件 Brier：E[(Y−p̂)²|X,Z] = p*(1−p*) + (p*−p̂)²，在评价集上取平均（Monte Carlo 近似，报告积分标准误）。
  选择器只看到开发集的 X、Z、可观察 Y，**从不接触 p***（端到端测试 E1 用「打乱评价 p*」核验，并配作弊变异对照）。
  若某候选对的真值相对差距离边界 m 小于 2 个积分标准误，追加 400,000 个独立评价样本重算该候选对
  （引擎确定性 ⇒ 重算得到的是同一模型；只提高评价精度，不重新训练别的模型、不换种子）。
选择器（同一重复、同一候选模型对；逐轨决定）
  ALWAYS_POOLED（永远 P*_D）· ALWAYS_SPLIT（永远 HTFS_D）·
  参照 R ∈ {P*, S0*} × 规则 {CV_ANY 内层 CV 点估计择优, CV_POINT_M 内层相对收益 ≥ m, CV_LB_M 内层 OOF 上的
  近似单侧下界 > m（与认证同一统计量，无保证）, FCT fit–certify–test（f=1/3、每轨 α=0.025、认证后不重拟合）} ·
  MIX 预测混合 t_g ∈ {0,.25,.5,.75,1}（内层 OOF 本轨 Brier 最小，平局取较小 t）·
  USTUN_ADAPT（Ustun 2019 思路的适配，**不称复现**：只保留 rationality 一项——留出认证集上 m=0 的单侧检验，
  参照 P*_F，损失换成 Brier；省略 envy-freeness 与递归解耦；具体检验形式未与 Ustun 原文核对）。
评价（两类分开，不混成一个「错误率」）
  ① 认证/选择判断：FCT 与 USTUN 针对 (P*_F 或 S0*_F, HTFS_F) 同一候选对；CV 规则针对 (P*_D 或 S0*_D, HTFS_D) 候选对；
     错误启用 = 启用但该候选对真值相对收益 < m；漏启用 = 未启用但真值相对收益 ≥ m（MIX 无二元决定，另报 t）。
  ② 整套程序的实际性能：各选择器部署后的真值风险，相对 P*_D 的差与 regret（相对 min(P*_D, HTFS_D) 的逐轨真值风险）。

用法
  python pgss_r43_t1.py config                      写出带指纹的配置
  python pgss_r43_t1.py run <jobs> [--only cost|main|thr] 运行（断点续跑：已有 _DONE 的重复跳过）
  python pgss_r43_t1.py one <scen> <n> <pi> <rep>  单个重复（调试）
输出：24_R43_结果补齐与T1/t1/runs/<tag>/{rep.json, arms.parquet, sel.parquet, fitlog.parquet, _DONE}
"""
import sys, os, json, pathlib, time, hashlib, traceback
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pgss_threads  # noqa: F401
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from scipy.stats import norm

T1DIR = HERE.parent / "24_R43_结果补齐与T1" / "t1"
RUNS = T1DIR / "runs"
SCEN_MAIN = ["N1", "N2", "P1", "P2"]
SCEN_THR = ["N1-T", "P2-T"]                 # 须 GPT 批准
N_GRID = [6000, 12000]
PI_GRID = [0.15, 0.30]
REPS = 20
N_EVAL, N_EVAL_EXTRA = 100_000, 400_000
M, ALPHA = 0.01, 0.025
T_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
THR_COL = 8
ROOT = 20431
CODE_FILES = ["pgss_r43_t1.py", "pgss_xgb_sim0.py", "pgss_xgb_engine.py", "pgss_r36_engine.py"]


def code_sha():
    h = hashlib.sha256()
    for f in CODE_FILES:
        h.update((HERE / f).read_bytes())
    return h.hexdigest()[:16]


def tag_of(scen, n, pi, rep):
    return f"{scen}__n{n}__pi{int(round(pi * 100)):02d}__r{rep:02d}"


# ── 数据生成（公式取自 pgss_xgb_sim0）───────────────────────────
def gen(scen, n, pi, rng):
    import pgss_xgb_sim0 as S0
    base = scen.split("-")[0]
    X = S0.gen_X(n, rng)
    if scen.endswith("-T"):
        z = X[:, THR_COL] > norm.ppf(1 - pi)
    else:
        z = rng.random(n) < pi
    p = S0.pstar(X, z, base)
    return X, z, p


def make_data(scen, n, pi, rep):
    ss = np.random.SeedSequence([ROOT, (SCEN_MAIN + SCEN_THR).index(scen), n, int(round(pi * 100)), rep])
    r_dev, r_eval, r_extra = [np.random.default_rng(s) for s in ss.spawn(3)]
    Xd, zd, pd_ = gen(scen, n, pi, r_dev)
    yd = (r_dev.random(n) < pd_).astype(int)
    Xe, ze, pe = gen(scen, N_EVAL, pi, r_eval)
    D = dict(X=Xd, y=yd, z=zd, g=np.arange(n, dtype=np.int64), uid=np.arange(n, dtype=np.int64))
    Ev = dict(X=Xe, y=np.zeros(N_EVAL, int), z=ze, g=np.arange(N_EVAL, dtype=np.int64) + 10**7,
              uid=np.arange(N_EVAL, dtype=np.int64) + 10**7)
    return D, Ev, pe, r_extra, pd_


def extra_eval(scen, pi, r_extra):
    Xe, ze, pe = gen(scen, N_EVAL_EXTRA, pi, r_extra)
    Ev2 = dict(X=Xe, y=np.zeros(N_EVAL_EXTRA, int), z=ze, g=np.arange(N_EVAL_EXTRA, dtype=np.int64) + 2 * 10**7,
               uid=np.arange(N_EVAL_EXTRA, dtype=np.int64) + 2 * 10**7)
    return Ev2, pe


# ── 评价器（唯一接触 p* 的地方）──────────────────────────────────
def unit_loss(p_hat, pe):
    return pe * (1 - pe) + (pe - p_hat) ** 2


def risk(p_hat, pe, z, g):
    m = z if g == "A" else ~z
    return float(unit_loss(p_hat[m], pe[m]).mean())


def pair_truth(p_ref, p_s, pe, z, g):
    """候选对的真值：相对收益 G=(R_ref−R_s)/R_ref 及其积分标准误（配对）。"""
    m = z if g == "A" else ~z
    lr, ls = unit_loss(p_ref[m], pe[m]), unit_loss(p_s[m], pe[m])
    R = lr.mean(); d = lr - ls
    return float(d.mean() / R), float(d.std(ddof=1) / np.sqrt(m.sum()) / R), float(R)


# ── 一个重复 ─────────────────────────────────────────────────────
def oof_of(EN, spec, D, lay, seed):
    folds = EN.group_split(D["g"], EN.INNER_FOLDS, seed)
    o = np.full(len(D["y"]), np.nan)
    for tr, te in folds:
        o[te] = EN.program_predict(spec, EN.sub(D, tr), EN.sub(D, te), lay)
    return o


def star(specs, inner, arch):
    ws = [w for w in ("none", "bal") if f"{arch}|{w}" in inner]
    return f"{arch}|{min(ws, key=lambda w: (inner[f'{arch}|{w}'], ws.index(w)))}"


def selectors(D, EN, cert, oof, cheat=None):
    """返回逐轨决定 {名称: {g: (专业化?, 部署来源, t, 依据1, 依据2)}}。只用开发数据与认证集统计量。
    cheat=(评价 p*, 评价 z, 评价集预测) 仅供端到端测试的变异对照：故意偷看真值。"""
    out = {}
    z, y = D["z"], D["y"].astype(float)
    for g in ("A", "B"):
        m_ = z if g == "A" else ~z
        out.setdefault("ALWAYS_POOLED", {})[g] = (False, "P*_D", None, np.nan, np.nan)
        out.setdefault("ALWAYS_SPLIT", {})[g] = (True, "HTFS_D", None, np.nan, np.nan)
        for ref in ("P*", "S0*"):
            Rr = float(np.mean((y[m_] - oof[ref][m_]) ** 2)); Rs = float(np.mean((y[m_] - oof["HTFS"][m_]) ** 2))
            if cheat is not None and ref == "P*":          # 变异对照：用真值挑模型
                pe_, ze_, pr_ = cheat; me = ze_ if g == "A" else ~ze_
                Rr = float(np.mean((pe_[me] - pr_["P*_D"][me]) ** 2)); Rs = float(np.mean((pe_[me] - pr_["HTFS_D"][me]) ** 2))
            out.setdefault(f"CV_ANY|{ref}", {})[g] = (bool(Rs < Rr), "HTFS_D" if Rs < Rr else f"{ref}_D", None, Rr, Rs)
            ok = (Rr - Rs) / Rr >= M
            out.setdefault(f"CV_POINT_M|{ref}", {})[g] = (bool(ok), "HTFS_D" if ok else f"{ref}_D", None, Rr, Rs)
            ct = EN.cert_test(y, oof[ref], oof["HTFS"], z, D["g"], M, g, ref="normal")
            ok = bool(np.isfinite(ct["p"]) and ct["p"] < ALPHA)
            out.setdefault(f"CV_LB_M|{ref}", {})[g] = (ok, "HTFS_D" if ok else f"{ref}_D", None, ct["dbar"], ct["p"])
            c = cert[ref][g]
            out.setdefault(f"FCT|{ref}", {})[g] = (c["specialize"], "HTFS_F" if c["specialize"] else f"{ref}_F", None,
                                                   c["dbar"], c["p"])
        c = cert["USTUN"][g]
        out.setdefault("USTUN_ADAPT", {})[g] = (c["specialize"], "HTFS_F" if c["specialize"] else "P*_F", None, c["dbar"], c["p"])
        sc = {t: float(np.mean((y[m_] - ((1 - t) * oof["P*"][m_] + t * oof["HTFS"][m_])) ** 2)) for t in T_GRID}
        tb = min(T_GRID, key=lambda t: (sc[t], t))
        out.setdefault("MIX", {})[g] = (None, "MIX", tb, sc[tb], sc[0.0])
    return out


def run_rep(scen, n, pi, rep, write=True, cheat=False, shuffle_pe=False):
    import pgss_xgb_engine as EN
    if os.environ.get("PGSS_T1_SMOKE") == "1":      # 只用于端到端测试：缩小网格
        EN.GRID_TRACK = EN.GRID_TRACK[:1]; EN.GRID_POOLED = EN.GRID_POOLED[:1]; EN.K_GRID = [5, 23]; EN.MAX_ROUNDS = 60
    t0 = time.time()
    tag = tag_of(scen, n, pi, rep)
    D, Ev, pe, r_extra, _ = make_data(scen, n, pi, rep)
    if shuffle_pe:                            # 端到端测试 E1：评价器之外不得用到 p*
        pe = np.random.default_rng(0).permutation(pe)
    names = [f"x{i}" for i in range(D["X"].shape[1])]
    lay = EN.task_layout(D["X"], D["z"], names)
    EN.set_namespace(sim="R43-T1", scen=scen, n=n, pi=pi, rep=rep, code=code_sha())
    seed_dev = 7 + 1000 * rep
    # main：D 上完整开发
    r = EN.develop(D, lay, seed_dev)
    specs, inner = r["specs"], r["inner"]
    fp = EN.final_predict(specs, D, Ev, lay)
    ch = lambda nm: fp[nm][fp[nm]["chosen"]]
    s0 = star(specs, inner, "S0"); sz = star(specs, inner, "SZ")
    preds_D = {"P*_D": ch(r["pstar"]), "HTFS_D": ch("HTFS"), "Hfull_D": ch("Hfull"), "HGFS_D": ch("HGFS"),
               "S0*_D": ch(s0), "SZ*_D": ch(sz)}
    oof = {"P*": oof_of(EN, specs[r["pstar"]], D, lay, seed_dev), "HTFS": oof_of(EN, specs["HTFS"], D, lay, seed_dev),
           "S0*": oof_of(EN, specs[s0], D, lay, seed_dev)}
    n_fit_main = len(EN.FITLOG); t_main = time.time() - t0
    # cert：D 按组划 F/C（f=1/3）→ develop(F) → 认证 → 不重拟合
    keep, hold = EN.cert_split(D["g"], seed_dev + 50_000)
    F, C = EN.sub(D, keep), EN.sub(D, hold)
    rF = EN.develop(F, lay, seed_dev + 70_000, arms_tracked=("Hfull", "HTFS"))
    s0F = star(rF["specs"], rF["inner"], "S0")
    specF = {"P*": rF["specs"][rF["pstar"]], "HTFS": rF["specs"]["HTFS"], "S0*": rF["specs"][s0F]}
    pc = {k: EN.program_predict(v, F, C, lay) for k, v in specF.items()}
    preds_F = {f"{k}_F": EN.program_predict(v, F, Ev, lay) for k, v in specF.items()}
    cert = {}
    for ref in ("P*", "S0*"):
        cert[ref] = {}
        for g in ("A", "B"):
            ct = EN.cert_test(C["y"], pc[ref], pc["HTFS"], C["z"], C["g"], M, g, ref="normal")
            cert[ref][g] = dict(ct, specialize=bool(np.isfinite(ct["p"]) and ct["p"] < ALPHA))
    cert["USTUN"] = {}
    for g in ("A", "B"):
        ct = EN.cert_test(C["y"], pc["P*"], pc["HTFS"], C["z"], C["g"], 0.0, g, ref="normal")
        cert["USTUN"][g] = dict(ct, specialize=bool(np.isfinite(ct["p"]) and ct["p"] < ALPHA))
    t_cert = time.time() - t0 - t_main
    sel = selectors(D, EN, cert, oof, cheat=((pe, Ev["z"], preds_D) if cheat else None))
    # ── 评价器 ──
    allp = dict(preds_D, **preds_F, ORACLE=pe.copy())
    z = Ev["z"]
    arm_rows = []
    for a, p in allp.items():
        arm_rows.append(dict(臂=a, R_A=risk(p, pe, z, "A"), R_B=risk(p, pe, z, "B"),
                             R_pop=float(unit_loss(p, pe).mean())))
    pairs = {"D|P*": ("P*_D", "HTFS_D"), "D|S0*": ("S0*_D", "HTFS_D"),
             "F|P*": ("P*_F", "HTFS_F"), "F|S0*": ("S0*_F", "HTFS_F")}
    truth, extra_used = {}, []
    Ev2 = None
    for pk, (a_ref, a_s) in pairs.items():
        for g in ("A", "B"):
            G, se, R = pair_truth(allp[a_ref], allp[a_s], pe, z, g)
            near = abs(G - M) < 2 * se
            if near:                         # 只提高评价精度：同一（确定性）模型在追加样本上重算
                if Ev2 is None:
                    Ev2, pe2 = extra_eval(scen, pi, r_extra)
                    if shuffle_pe:
                        pe2 = np.random.default_rng(1).permutation(pe2)
                src = (D if a_ref.endswith("_D") else F)
                sp_ref = {"P*_D": specs[r["pstar"]], "S0*_D": specs[s0], "P*_F": specF["P*"], "S0*_F": specF["S0*"]}[a_ref]
                sp_s = specs["HTFS"] if a_s == "HTFS_D" else specF["HTFS"]
                q_ref = EN.program_predict(sp_ref, src, Ev2, lay); q_s = EN.program_predict(sp_s, src, Ev2, lay)
                G2, se2, R2 = pair_truth(np.concatenate([allp[a_ref], q_ref]), np.concatenate([allp[a_s], q_s]),
                                         np.concatenate([pe, pe2]), np.concatenate([z, Ev2["z"]]), g)
                extra_used.append(f"{pk}|{g}")
                G, se, R = G2, se2, R2
            truth[(pk, g)] = dict(G=G, se=se, R_ref=R, 追加评价=bool(near))
    sel_rows = []
    for name, dec in sel.items():
        for g in ("A", "B"):
            spec_, src, t_, b1, b2 = dec[g]
            if src == "MIX":
                p = (1 - t_) * allp["P*_D"] + t_ * allp["HTFS_D"]
            else:
                p = allp[src]
            Rg = risk(p, pe, z, g)
            Rp = risk(allp["P*_D"], pe, z, g); Rh = risk(allp["HTFS_D"], pe, z, g)
            rule_pair = ("F|" if name.startswith(("FCT", "USTUN")) else "D|") + ("S0*" if name.endswith("S0*") else "P*")
            tr = truth[(rule_pair, g)]
            sel_rows.append(dict(选择器=name, 轨=g, 专业化=spec_, 部署=src, t=t_, 依据1=b1, 依据2=b2, R=Rg, R_PstarD=Rp, R_HTFSD=Rh,
                                 相对PstarD改善=(Rp - Rg) / Rp, regret=Rg - min(Rp, Rh), regret_相对=(Rg - min(Rp, Rh)) / Rp,
                                 判断所依候选对=rule_pair, 候选对真值相对收益=tr["G"], 候选对积分SE=tr["se"],
                                 候选对达m=bool(tr["G"] >= M),
                                 错误启用=(None if spec_ is None else bool(spec_ and tr["G"] < M)),
                                 漏启用=(None if spec_ is None else bool((not spec_) and tr["G"] >= M))))
    meta = dict(tag=tag, scen=scen, n=n, pi=pi, rep=rep, code_sha=code_sha(), engine=EN.ENGINE_VERSION,
                pstar_D=r["pstar"], pstar_cal=bool(specs[r["pstar"]]["cal"]), S0star_D=s0,
                HTFS_kA=specs["HTFS"]["kA"], HTFS_kB=specs["HTFS"]["kB"],
                SZGFS_k_none=specs["SZ-GFS|none"].get("k", 23), SZGFS_k_bal=specs["SZ-GFS|bal"].get("k", 23),
                pstar_F=rF["pstar"], HTFS_F_kA=rF["specs"]["HTFS"]["kA"], HTFS_F_kB=rF["specs"]["HTFS"]["kB"],
                n_A_dev=int(D["z"].sum()), events_A_dev=int(D["y"][D["z"]].sum()), events_B_dev=int(D["y"][~D["z"]].sum()),
                n_F=len(keep), n_C=len(hold), n_eval=N_EVAL, n_eval_A=int(z.sum()), 追加评价=extra_used,
                cert={k: {g: {kk: vv for kk, vv in v[g].items()} for g in ("A", "B")} for k, v in cert.items()},
                truth={f"{k[0]}|{k[1]}": v for k, v in truth.items()},
                sec_total=round(time.time() - t0, 1), sec_main=round(t_main, 1), sec_cert=round(t_cert, 1),
                n_fits=len(EN.FITLOG), n_fits_main=n_fit_main, requests=dict(EN.REQLOG))
    if write:
        od = RUNS / tag; od.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(arm_rows).to_parquet(od / "arms.parquet", index=False)
        pd.DataFrame(sel_rows).to_parquet(od / "sel.parquet", index=False)
        pd.DataFrame(EN.FITLOG).to_parquet(od / "fitlog.parquet", index=False)
        (od / "rep.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        (od / "_DONE").write_text(str(meta["sec_total"]), encoding="utf-8")
    return meta, pd.DataFrame(arm_rows), pd.DataFrame(sel_rows)


def _entry(args):
    scen, n, pi, rep = args
    od = RUNS / tag_of(*args)
    try:
        run_rep(scen, n, pi, rep)
        return args, "OK"
    except Exception as e:
        od.mkdir(parents=True, exist_ok=True)
        (od / "_FAILED").write_text(f"{type(e).__name__}: {e}\n{traceback.format_exc()[-3000:]}", encoding="utf-8")
        return args, "FAILED"


def job_list(only):
    if only == "cost":       # 测成本：每个 (n, π) 的 P2 与 N1 各 1 个重复
        return [(s, n, pi, 0) for s in ("P2", "N1") for n in N_GRID for pi in PI_GRID]
    scen = SCEN_THR if only == "thr" else SCEN_MAIN
    jl = [(s, n, pi, rep) for rep in range(REPS) for n in sorted(N_GRID, reverse=True) for pi in PI_GRID for s in scen]
    return jl


def write_config():
    import pgss_xgb_sim0 as S0, pgss_xgb_engine as EN, inspect
    T1DIR.mkdir(parents=True, exist_ok=True)
    cfg = dict(版本="R43-T1 v1", 代码指纹=code_sha(),
               代码文件SHA={f: hashlib.sha256((HERE / f).read_bytes()).hexdigest()[:16] for f in CODE_FILES},
               引擎=EN.ENGINE_VERSION, 机制_本批=SCEN_MAIN, 机制_待批准=SCEN_THR, 阈值列=THR_COL,
               n=N_GRID, pi_A=PI_GRID, 重复=REPS, 评价样本=N_EVAL, 追加评价样本=N_EVAL_EXTRA, 边界m=M, 每轨alpha=ALPHA,
               混合t网格=T_GRID, 认证f="1/3（engine.cert_split）", 认证参照分布="normal（独立个体）", 缺失="无", 分组="独立个体",
               生成_gen_X=inspect.getsource(S0.gen_X), 生成_pstar=inspect.getsource(S0.pstar), 种子根=ROOT,
               种子规则="SeedSequence([20431, 机制序号, n, round(100π), rep]).spawn(3) → 开发/评价/追加评价",
               开发种子="7+1000·rep；认证划分 +50000；F 上开发 +70000（同 run_unit 的偏移）",
               引擎网格=dict(GRID_TRACK=EN.GRID_TRACK, GRID_POOLED=EN.GRID_POOLED, K_GRID=EN.K_GRID,
                          MAX_ROUNDS=EN.MAX_ROUNDS, PATIENCE=EN.PATIENCE))
    s = json.dumps(cfg, ensure_ascii=False, indent=1, default=str)
    cfg["配置指纹"] = hashlib.sha256(s.encode()).hexdigest()[:16]
    (T1DIR / "R43_T1_CONFIG.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(cfg["配置指纹"], cfg["代码指纹"])


def run(jobs, only):
    from concurrent.futures import ProcessPoolExecutor
    RUNS.mkdir(parents=True, exist_ok=True)
    st = T1DIR / f"_STATUS_{only}.json"
    jl = [a for a in job_list(only) if not (RUNS / tag_of(*a) / "_DONE").exists()]
    tot = len(job_list(only)); done0 = tot - len(jl); ok = fail = 0; t0 = time.time()
    with ProcessPoolExecutor(max_workers=jobs, max_tasks_per_child=1) as ex:
        for args, res in ex.map(_entry, jl):
            ok += res == "OK"; fail += res == "FAILED"
            st.write_text(json.dumps(dict(批次=only, 总数=tot, 先前已完成=done0, 本次完成=ok, 失败=fail,
                                          已用分钟=round((time.time() - t0) / 60, 1),
                                          更新=time.strftime("%m-%d %H:%M:%S")), ensure_ascii=False, indent=1), encoding="utf-8")
            print(args, res, flush=True)
    (T1DIR / (f"_DONE_{only}" if fail == 0 else f"_FAILED_{only}")).write_text(str(fail), encoding="utf-8")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "config":
        write_config()
    elif cmd == "run":
        only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else "main"
        run(int(sys.argv[2]), only)
    elif cmd == "one":
        m, A, S = run_rep(sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), write=False)
        print(json.dumps({k: v for k, v in m.items() if k not in ("cert",)}, ensure_ascii=False, default=float, indent=1))
        print(A.to_string()); print(S.to_string())
