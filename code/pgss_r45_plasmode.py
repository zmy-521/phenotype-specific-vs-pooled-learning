# -*- coding: utf-8 -*-
r"""R45 有限半合成（plasmode）试验（GPT 第 49 轮 §7）：固定经验协变量分布下的探索性压力测试，不是独立外部验证。

数据框架：MIMIC data v5，「eGFR 降低」任务的可判定人群（合格 & 表型可判定）；学习输入 = 原始 23 项池（原缺失保留，原训练内预处理）。
生成器（只供评价器与开发集标签使用；**不提供给学习器**）
  U0–U7 = 年龄、RDW、血红蛋白、平均动脉压、血肌酐、尿素氮、碳酸氢盐、血小板；
  稳健标准化：(U − 中位数)/(IQR/1.349)，中位数与 IQR 取自整个经验人群；缺失 → 0（= 经验中位数）；截断 [−3, 3]。
  logit p_δ = a_δ + (1−δ)·f0(U) + δ·{Z·gA(U) + (1−Z)·gB(U)}，Z = eGFR 降低；
  f0、gA、gB **直接由 pgss_xgb_sim0.pstar 求出**（f0 = logit p_N1 + 1.6；gA = logit p_P2(z=1) + 1.4；gB = logit p_P2(z=0) + 1.7），
  即 T1 代码中的同一函数形式；a_δ 为共同截距，使整个经验人群的平均 p_δ = 15%（brentq），不使用真实死亡结局；各轨事件率允许不同、完整报告。
设计：δ ∈ {0, 0.5, 1} × 20 个重复 = 60。每个重复：按完整患者随机累积到 ≥ 6000 条记录作开发集，其余患者的全部记录作评价集
  （患者完全隔离）；同一重复内三档强度复用同一划分与同一组均匀随机数 u（Y = 1[u < p_δ]，耦合）；不同重复独立。
  种子 SeedSequence([20452, rep]).spawn(2) → 划分 / 标签均匀数。
学习程序与选择器：与 R43 T1 完全相同（冻结引擎 develop + final_predict；cert f=1/3、每轨 α=0.025、认证后不重拟合；
  T1.selectors：ALWAYS_*、CV_ANY/CV_POINT_M/CV_LB_M × {P*, S0*}、FCT × {P*, S0*}、MIX、USTUN_ADAPT），开发种子同 T1（7+1000·rep；+50000；+70000）。
  **新增普通合并候选融合 POOLMIX**：P* 与另一合并族的最佳者 ALT（P* 属含 Z 族 {SZ, SZ-GFS, SZ-union} ⇒ ALT = S0*；P* 属 S0 族 ⇒ ALT = SZ*）
  按轨在 t ∈ {0,.25,.5,.75,1} 上以内层 OOF 本轨 Brier 选 t（平局取较小 t）——与 MIX 同一结构，只是第二个模型来自合并族，
  用于区分「两个不同模型的误差互补」与「表型专业化」。
评价器：评价集上 p_δ 已知 ⇒ 条件 Brier = p(1−p) + (p − p̂)² 在有限评价人群上**精确**计算（无 Monte Carlo 积分误差，无追加评价）。
  报告：各臂逐轨风险、候选对真值收益（D|P*、D|S0*、F|P*、F|S0*）、选择器的启用/错误启用/漏启用与部署风险、认证样本成本（P*_F vs P*_D）、
  MIX 与 POOLMIX 的区别。**δ > 0 只表示生成函数含轨道特异结构，不等于已注入某个幅度的专业化收益**——收益由冻结模型的风险评价决定。
范围声明：保留真实患者分组与协变量结构；给定 p_δ 后按 Bernoulli 生成结局；**不模拟医院层残差相关，不覆盖 eICU 少量不均衡簇下的认证性质**。
用法：python pgss_r45_plasmode.py run <jobs> | one <delta> <rep> | gen
输出：27_R45_外部复现与机制核验/plasmode/runs/<tag>/{rep.json, arms.parquet, sel.parquet, fitlog.parquet, _DONE}（只含汇总量与合成标签统计，无患者级记录）
"""
import sys, os, json, pathlib, time, hashlib, traceback
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pgss_threads  # noqa: F401
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd

PL = HERE.parent / "27_R45_外部复现与机制核验" / "plasmode"
RUNS = PL / "runs"
DELTAS = [0.0, 0.5, 1.0]
REPS = 20
N_TARGET = 6000
EVENT_RATE = 0.15
CLIP = 3.0
PHENO = "eGFR降低"
U_COLS = ["年龄", "RDW", "血红蛋白", "平均动脉压", "血肌酐", "尿素氮", "碳酸氢盐", "血小板"]
ROOT = 20452
T_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
CODE_FILES = ["pgss_r45_plasmode.py", "pgss_r43_t1.py", "pgss_xgb_sim0.py", "pgss_xgb_engine.py", "pgss_r36_engine.py",
              "pgss_r38_bench.py"]


def code_sha():
    h = hashlib.sha256()
    for f in CODE_FILES:
        h.update((HERE / f).read_bytes())
    return h.hexdigest()[:16]


def tag_of(delta, rep):
    return f"d{int(round(delta * 100)):03d}__r{rep:02d}"


# ── 经验人群与生成器 ────────────────────────────────────────────
def population():
    import pgss_r38_bench as BN
    X, P, OK, meta = BN.load("mimic")
    e = meta.eligible.fillna(False).to_numpy(bool) & OK[PHENO].to_numpy(bool)
    idx = np.flatnonzero(e)
    return dict(X=X.to_numpy(float)[idx], names=list(X.columns), z=P[PHENO].to_numpy(bool)[idx],
                gid=meta.group_id.astype(str).to_numpy()[idx], uid=X.index.to_numpy()[idx].astype(np.int64))


def gen_parts(pop):
    """返回 (f0, gA, gB, 标准化参数)。f0/gA/gB 由 pgss_xgb_sim0.pstar 反解，保证与 T1 同一函数形式。"""
    import pgss_xgb_sim0 as S0
    ci = [pop["names"].index(c) for c in U_COLS]
    U = pop["X"][:, ci]
    med = np.nanmedian(U, axis=0); q1, q3 = np.nanpercentile(U, [25, 75], axis=0); sc = (q3 - q1) / 1.349
    Us = np.clip(np.where(np.isnan(U), 0.0, (U - med) / sc), -CLIP, CLIP)
    n = len(Us)
    Xf = np.zeros((n, S0.P)); Xf[:, :8] = Us
    lg = lambda p: np.log(p) - np.log1p(-p)
    f0 = lg(S0.pstar(Xf, np.zeros(n, bool), "N1")) + 1.6
    gA = lg(S0.pstar(Xf, np.ones(n, bool), "P2")) + 1.4
    gB = lg(S0.pstar(Xf, np.zeros(n, bool), "P2")) + 1.7
    return f0, gA, gB, dict(U列=U_COLS, 中位数=med.tolist(), 尺度=sc.tolist(), 截断=CLIP,
                            标准化后缺失比例=np.isnan(U).mean(axis=0).tolist())


def p_delta(f0, gA, gB, z, delta):
    from scipy.optimize import brentq
    lin = (1 - delta) * f0 + delta * np.where(z, gA, gB)
    a = brentq(lambda a: np.mean(1 / (1 + np.exp(-(a + lin)))) - EVENT_RATE, -20, 20, xtol=1e-14)
    return 1 / (1 + np.exp(-(a + lin))), float(a)


def split(pop, rep):
    ss = np.random.SeedSequence([ROOT, rep]); r_split, r_y = [np.random.default_rng(s) for s in ss.spawn(2)]
    uq, inv = np.unique(pop["gid"], return_inverse=True)
    sizes = np.bincount(inv); order = r_split.permutation(len(uq))
    take = order[: int(np.searchsorted(np.cumsum(sizes[order]), N_TARGET) + 1)]
    dev = np.isin(inv, take)
    return np.flatnonzero(dev), np.flatnonzero(~dev), r_y


# ── 一个 (δ, rep) ───────────────────────────────────────────────
def run_one(delta, rep, write=True, shuffle_pe=False, pop=None, parts=None):
    import pgss_xgb_engine as EN, pgss_r43_t1 as T1
    if os.environ.get("PGSS_T1_SMOKE") == "1":
        EN.GRID_TRACK = EN.GRID_TRACK[:1]; EN.GRID_POOLED = EN.GRID_POOLED[:1]; EN.K_GRID = [5, 23]; EN.MAX_ROUNDS = 60
    t0 = time.time()
    pop = population() if pop is None else pop
    f0, gA, gB, gpar = gen_parts(pop) if parts is None else parts
    p, a = p_delta(f0, gA, gB, pop["z"], delta)
    di, ei, r_y = split(pop, rep)
    u = r_y.random(len(di))
    y_dev = (u < p[di]).astype(int)
    gs = pop["gid"][di]
    D = dict(X=pop["X"][di], y=y_dev, z=pop["z"][di], g=EN.group_codes(gs), uid=pop["uid"][di])
    Ev = dict(X=pop["X"][ei], y=np.zeros(len(ei), int), z=pop["z"][ei], g=np.arange(len(ei), dtype=np.int64) + 10**7,
              uid=pop["uid"][ei])
    pe = p[ei].copy()
    if shuffle_pe:
        pe = np.random.default_rng(0).permutation(pe)
    lay = EN.task_layout(D["X"], D["z"], pop["names"])
    EN.set_namespace(sim="R45-plasmode", delta=delta, rep=rep, code=code_sha())
    seed_dev = 7 + 1000 * rep
    r = EN.develop(D, lay, seed_dev)
    specs, inner = r["specs"], r["inner"]
    fp = EN.final_predict(specs, D, Ev, lay)
    ch = lambda nm: fp[nm][fp[nm]["chosen"]]
    s0 = T1.star(specs, inner, "S0"); sz = T1.star(specs, inner, "SZ")
    alt = sz if specs[r["pstar"]]["arch"] == "S0" else s0      # 另一合并族的最佳者（S0 族 ↔ 含 Z 族）
    preds_D = {"P*_D": ch(r["pstar"]), "HTFS_D": ch("HTFS"), "Hfull_D": ch("Hfull"), "HGFS_D": ch("HGFS"),
               "S0*_D": ch(s0), "SZ*_D": ch(sz), "ALT_D": ch(alt)}
    oof = {"P*": T1.oof_of(EN, specs[r["pstar"]], D, lay, seed_dev), "HTFS": T1.oof_of(EN, specs["HTFS"], D, lay, seed_dev),
           "S0*": T1.oof_of(EN, specs[s0], D, lay, seed_dev), "ALT": T1.oof_of(EN, specs[alt], D, lay, seed_dev)}
    t_main = time.time() - t0
    keep, hold = EN.cert_split(D["g"], seed_dev + 50_000)
    F, C = EN.sub(D, keep), EN.sub(D, hold)
    rF = EN.develop(F, lay, seed_dev + 70_000, arms_tracked=("Hfull", "HTFS"))
    s0F = T1.star(rF["specs"], rF["inner"], "S0")
    specF = {"P*": rF["specs"][rF["pstar"]], "HTFS": rF["specs"]["HTFS"], "S0*": rF["specs"][s0F]}
    pc = {k: EN.program_predict(v, F, C, lay) for k, v in specF.items()}
    preds_F = {f"{k}_F": EN.program_predict(v, F, Ev, lay) for k, v in specF.items()}
    cert = {}
    for ref in ("P*", "S0*"):
        cert[ref] = {}
        for g in ("A", "B"):
            ct = EN.cert_test(C["y"], pc[ref], pc["HTFS"], C["z"], C["g"], T1.M, g, ref="normal")
            cert[ref][g] = dict(ct, specialize=bool(np.isfinite(ct["p"]) and ct["p"] < T1.ALPHA))
    cert["USTUN"] = {}
    for g in ("A", "B"):
        ct = EN.cert_test(C["y"], pc["P*"], pc["HTFS"], C["z"], C["g"], 0.0, g, ref="normal")
        cert["USTUN"][g] = dict(ct, specialize=bool(np.isfinite(ct["p"]) and ct["p"] < T1.ALPHA))
    t_cert = time.time() - t0 - t_main
    sel = T1.selectors(D, EN, cert, oof)
    yf = D["y"].astype(float)
    sel["POOLMIX"] = {}
    for g in ("A", "B"):
        m_ = D["z"] if g == "A" else ~D["z"]
        sc = {t: float(np.mean((yf[m_] - ((1 - t) * oof["P*"][m_] + t * oof["ALT"][m_])) ** 2)) for t in T_GRID}
        tb = min(T_GRID, key=lambda t: (sc[t], t))
        sel["POOLMIX"][g] = (None, "POOLMIX", tb, sc[tb], sc[0.0])
    # ── 评价器（唯一接触 p_δ 的地方；有限评价人群上精确）──
    allp = dict(preds_D, **preds_F, ORACLE=pe.copy())
    z = Ev["z"]
    arm_rows = [dict(臂=k, R_A=T1.risk(v, pe, z, "A"), R_B=T1.risk(v, pe, z, "B"), R_pop=float(T1.unit_loss(v, pe).mean()))
                for k, v in allp.items()]
    pairs = {"D|P*": ("P*_D", "HTFS_D"), "D|S0*": ("S0*_D", "HTFS_D"), "F|P*": ("P*_F", "HTFS_F"), "F|S0*": ("S0*_F", "HTFS_F")}
    truth = {}
    for pk, (a_ref, a_s) in pairs.items():
        for g in ("A", "B"):
            G, se, R = T1.pair_truth(allp[a_ref], allp[a_s], pe, z, g)
            truth[(pk, g)] = dict(G=G, se_单位间=se, R_ref=R)
    sel_rows = []
    for name, dec in sel.items():
        for g in ("A", "B"):
            spec_, src, t_, b1, b2 = dec[g]
            if src == "MIX":
                pp = (1 - t_) * allp["P*_D"] + t_ * allp["HTFS_D"]
            elif src == "POOLMIX":
                pp = (1 - t_) * allp["P*_D"] + t_ * allp["ALT_D"]
            else:
                pp = allp[src]
            Rg = T1.risk(pp, pe, z, g); Rp = T1.risk(allp["P*_D"], pe, z, g); Rh = T1.risk(allp["HTFS_D"], pe, z, g)
            rule_pair = ("F|" if name.startswith(("FCT", "USTUN")) else "D|") + ("S0*" if name.endswith("S0*") else "P*")
            tr = truth[(rule_pair, g)]
            sel_rows.append(dict(选择器=name, 轨=g, 专业化=spec_, 部署=src, t=t_, 依据1=b1, 依据2=b2, R=Rg, R_PstarD=Rp, R_HTFSD=Rh,
                                 相对PstarD改善=(Rp - Rg) / Rp,
                                 带符号差_相对两纯策略较优者=Rg - min(Rp, Rh), 带符号差_相对两纯策略较优者_相对=(Rg - min(Rp, Rh)) / Rp,
                                 判断所依候选对=rule_pair, 候选对真值相对收益=tr["G"], 候选对达m=bool(tr["G"] >= T1.M),
                                 错误启用=(None if spec_ is None else bool(spec_ and tr["G"] < T1.M)),
                                 错误启用_实际变差=(None if spec_ is None else bool(spec_ and tr["G"] < 0)),
                                 漏启用=(None if spec_ is None else bool((not spec_) and tr["G"] >= T1.M))))
    ze_dev = D["z"]
    meta = dict(tag=tag_of(delta, rep), delta=delta, rep=rep, code_sha=code_sha(), engine=EN.ENGINE_VERSION,
                截距a=a, 人群平均p=float(p.mean()), 人群A轨平均p=float(p[pop["z"]].mean()), 人群B轨平均p=float(p[~pop["z"]].mean()),
                n_dev=len(di), n_dev_patients=int(len(np.unique(gs))), n_eval=len(ei),
                n_A_dev=int(ze_dev.sum()), events_A_dev=int(y_dev[ze_dev].sum()), events_B_dev=int(y_dev[~ze_dev].sum()),
                n_eval_A=int(z.sum()), eval_pA_mean=float(pe[z].mean()), eval_pB_mean=float(pe[~z].mean()),
                dev_split_sha=hashlib.sha256(np.sort(pop["uid"][di]).tobytes()).hexdigest()[:16],
                u_sha=hashlib.sha256(u.tobytes()).hexdigest()[:16],
                pstar_D=r["pstar"], pstar_cal=bool(specs[r["pstar"]]["cal"]), S0star_D=s0, ALT=alt,
                HTFS_kA=specs["HTFS"]["kA"], HTFS_kB=specs["HTFS"]["kB"], pstar_F=rF["pstar"],
                n_F=len(keep), n_C=len(hold),
                cert={k: {g: dict(v[g]) for g in ("A", "B")} for k, v in cert.items()},
                truth={f"{k[0]}|{k[1]}": v for k, v in truth.items()}, 生成器=gpar,
                sec_total=round(time.time() - t0, 1), sec_main=round(t_main, 1), sec_cert=round(t_cert, 1), n_fits=len(EN.FITLOG))
    if write:
        od = RUNS / tag_of(delta, rep); od.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(arm_rows).to_parquet(od / "arms.parquet", index=False)
        pd.DataFrame(sel_rows).to_parquet(od / "sel.parquet", index=False)
        pd.DataFrame(EN.FITLOG).to_parquet(od / "fitlog.parquet", index=False)
        (od / "rep.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        (od / "_DONE").write_text(str(meta["sec_total"]), encoding="utf-8")
    return meta, pd.DataFrame(arm_rows), pd.DataFrame(sel_rows)


def _entry(args):
    delta, rep = args
    try:
        run_one(delta, rep)
        return args, "OK"
    except Exception as e:
        od = RUNS / tag_of(*args); od.mkdir(parents=True, exist_ok=True)
        (od / "_FAILED").write_text(f"{type(e).__name__}: {e}\n{traceback.format_exc()[-3000:]}", encoding="utf-8")
        return args, "FAILED"


def job_list():
    return [(d, r) for r in range(REPS) for d in DELTAS]


def run(jobs):
    from concurrent.futures import ProcessPoolExecutor
    RUNS.mkdir(parents=True, exist_ok=True)
    jl = [a for a in job_list() if not (RUNS / tag_of(*a) / "_DONE").exists()]
    ok = fail = 0; t0 = time.time()
    with ProcessPoolExecutor(max_workers=jobs, max_tasks_per_child=1) as ex:
        for args, res in ex.map(_entry, jl):
            ok += res == "OK"; fail += res == "FAILED"
            (PL / "_STATUS.json").write_text(json.dumps(dict(总数=len(job_list()), 本次待跑=len(jl), 完成=ok, 失败=fail,
                                                             已用分钟=round((time.time() - t0) / 60, 1), 更新=time.strftime("%m-%d %H:%M:%S")),
                                                        ensure_ascii=False, indent=1), encoding="utf-8")
            print(args, res, flush=True)
    (PL / ("_RUNS_DONE" if fail == 0 else "_RUNS_FAILED")).write_text(f"ok={ok} fail={fail}", encoding="utf-8")


def gen_report():
    """生成器的描述（运行前即可算，不涉及任何拟合）：各 δ 的截距、各轨平均 p、Bayes 风险。"""
    pop = population(); parts = gen_parts(pop)
    rows = []
    for d in DELTAS:
        p, a = p_delta(*parts[:3], pop["z"], d)
        bay = p * (1 - p)
        rows.append(dict(δ=d, 截距a=a, 人群平均p=float(p.mean()), A轨平均p=float(p[pop["z"]].mean()), B轨平均p=float(p[~pop["z"]].mean()),
                         A轨Bayes风险=float(bay[pop["z"]].mean()), B轨Bayes风险=float(bay[~pop["z"]].mean()),
                         人群人次=len(p), A轨人次=int(pop["z"].sum()), 唯一患者=int(len(np.unique(pop["gid"])))))
    G = pd.DataFrame(rows)
    PL.mkdir(parents=True, exist_ok=True)
    G.to_csv(PL / "R45_PLASMODE_GENERATOR.csv", index=False, encoding="utf-8-sig")
    (PL / "R45_PLASMODE_GENERATOR.json").write_text(json.dumps(dict(各δ=rows, 标准化=parts[3]), ensure_ascii=False, indent=1,
                                                               default=float), encoding="utf-8")
    print(G.to_string())


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        run(int(sys.argv[2]))
    elif cmd == "one":
        m, A, S = run_one(float(sys.argv[2]), int(sys.argv[3]), write=False)
        print(json.dumps({k: v for k, v in m.items() if k not in ("cert", "生成器")}, ensure_ascii=False, default=float, indent=1))
        print(A.to_string()); print(S.to_string())
    elif cmd == "gen":
        gen_report()
