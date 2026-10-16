# -*- coding: utf-8 -*-
r"""R46 T2 独立确认性模拟（方案 28_R46_最终核验与T2/t2/R46_T2_PROTOCOL_v1.0.md；GPT 第 50 轮 §7）。

8 个条件 × 200 次：T2-A = N1、N2、P1、P2、N1-T、P2-T（高斯，同 T1）；T2-B = N1-H、P2-H（多元 t₅：X = √(3/5)·G/√(V/5)）。
n = 6000、π_A = 0.15、无缺失、独立个体。结局线性预测子与 pgss_xgb_sim0.pstar 逐项相同（lin_of，运行前测试核对），概率用 expit。
学习程序与选择器：冻结引擎 xgb-1.0；T1 的 12 个选择器（pgss_r43_t1.selectors）+ POOLMIX（同 plasmode 定义）。
评价器：逐重复自适应评价规模（累计 100k → 500k → 1M；每段按 10 万一块生成与预测）；候选对边界 {m, 0}、MIX/POOLMIX 相对 P*_D 边界 {0}；
  任一项距边界 < 2 个积分 SE ⇒ 追加下一段；到 1M 仍接近 ⇒「边界不确定」。**该重复内全部量在同一最终评价集上计算。**
随机流：SeedSequence([20461, 条件序号, rep]).spawn(5) → 开发 / 评价段 0 / 段 1 / 段 2 / 引擎种子。
用法：python pgss_r46_t2.py run <jobs> [--only cost|all] | probe | one <cond> <rep>
输出：28_R46_最终核验与T2/t2/runs/<cond>__r<rep>/{rep.json, arms.parquet, pairs.parquet, sel.parquet, fitlog.parquet, _DONE}
"""
import sys, os, json, pathlib, time, hashlib, traceback
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pgss_threads  # noqa: F401
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from scipy.stats import norm
from scipy.special import expit

T2DIR = HERE.parent / "28_R46_最终核验与T2" / "t2"
RUNS = T2DIR / "runs"
CONDS = ["N1", "N2", "P1", "P2", "N1-T", "P2-T", "N1-H", "P2-H"]
N_DEV, PI, REPS = 6000, 0.15, 200
STAGES = [100_000, 400_000, 500_000]            # 追加段大小；累计 100k / 500k / 1M
BLOCK = 100_000                                 # 评价集逐块生成与预测的块大小（限制内存；结果与整段相同）
M, ALPHA = 0.01, 0.025
T_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
THR_COL, DF_T = 8, 5
ROOT = 20461
PAIRS = {"D|P*": ("P*_D", "HTFS_D"), "D|S0*": ("S0*_D", "HTFS_D"), "F|P*": ("P*_F", "HTFS_F"), "F|S0*": ("S0*_F", "HTFS_F")}
CODE_FILES = ["pgss_r46_t2.py", "pgss_r43_t1.py", "pgss_xgb_sim0.py", "pgss_xgb_engine.py", "pgss_r36_engine.py"]


def code_sha():
    h = hashlib.sha256()
    for f in CODE_FILES:
        h.update((HERE / f).read_bytes())
    return h.hexdigest()[:16]


def tag_of(cond, rep):
    return f"{cond}__r{rep:03d}"


# ── 生成 ───────────────────────────────────────────────────────
def lin_of(X, z, base):
    """与 pgss_xgb_sim0.pstar 逐项相同的线性预测子（运行前测试断言 expit(lin_of) ≡ pstar）。"""
    common = 0.8 * X[:, 0] - 0.5 * X[:, 1] + 0.4 * np.tanh(X[:, 2] * X[:, 3]) + 0.3 * X[:, 5]
    if base == "N1":
        return -1.6 + common
    if base == "N2":
        return -1.9 + common + 0.8 * z
    if base == "P1":
        return -1.6 + common + 1.0 * z * X[:, 6]
    if base == "P2":
        gA = 1.2 * np.tanh(1.5 * X[:, 0]) + 0.9 * X[:, 1] * X[:, 2] - 0.6 * (X[:, 3] ** 2 - 1)
        gB = 1.0 * X[:, 4] - 0.8 * np.abs(X[:, 5]) + 0.7 * X[:, 6] * X[:, 7]
        return np.where(z, -1.4 + gA, -1.7 + gB)
    raise ValueError(base)


def gen_X_t(n, rng):
    import pgss_xgb_sim0 as S0
    C = 0.2 * np.ones((S0.P, S0.P)) + 0.8 * np.eye(S0.P)
    G = rng.multivariate_normal(np.zeros(S0.P), C, size=n)
    V = rng.chisquare(DF_T, size=n)
    return np.sqrt((DF_T - 2) / DF_T) * G / np.sqrt(V / DF_T)[:, None]


def gen(cond, n, rng):
    import pgss_xgb_sim0 as S0
    base = cond.split("-")[0]
    if cond.endswith("-H"):
        X = gen_X_t(n, rng); z = rng.random(n) < PI
    else:
        X = S0.gen_X(n, rng)
        z = X[:, THR_COL] > norm.ppf(1 - PI) if cond.endswith("-T") else rng.random(n) < PI
    lin = lin_of(X, z, base)
    return X, z, expit(lin), lin


def streams(cond, rep):
    ss = np.random.SeedSequence([ROOT, CONDS.index(cond), rep])
    r = [np.random.default_rng(s) for s in ss.spawn(5)]
    seed_dev = int(r[4].integers(1, 2 ** 30))
    return r[0], r[1:4], seed_dev


def eval_blocks(cond, s, rng):
    """第 s 段评价集按 BLOCK 行一块依次生成（每块先 X 后 Z，用该段的同一随机流）；逐块产出以限制内存。
    预测也逐块进行：同一开发数据上的同一确定性程序，逐块预测与整段预测逐位相同（端到端测试 G8 核验）。"""
    for b in range(STAGES[s] // BLOCK):
        Xe, ze, pe, lin = gen(cond, BLOCK, rng)
        off = (s + 1) * 10 ** 7 + b * BLOCK
        yield dict(X=Xe, y=np.zeros(len(ze), int), z=ze, g=np.arange(len(ze), dtype=np.int64) + off,
                   uid=np.arange(len(ze), dtype=np.int64) + off), pe, lin


# ── 评价器（唯一接触 p 的地方）────────────────────────────────────
def uloss(ph, pe):
    return pe * (1 - pe) + (pe - ph) ** 2


def rel(p_ref, p_s, pe, msk):
    """相对 p_ref 的改善 G = (R_ref − R_s)/R_ref 及积分 SE（配对）。"""
    lr, ls = uloss(p_ref[msk], pe[msk]), uloss(p_s[msk], pe[msk])
    R = lr.mean(); d = lr - ls
    return float(d.mean() / R), float(d.std(ddof=1) / np.sqrt(msk.sum()) / R), float(R)


def mixed(t, a, b):
    return (1 - t) * a + t * b


def evaluate(P, pe, z, sel):
    """在累计评价集上：候选对（边界 m、0）与 MIX/POOLMIX（边界 0）的真值与积分 SE；返回 (truth, strat, near 列表)。"""
    truth, strat, near = {}, {}, []
    for pk, (a, b) in PAIRS.items():
        for g in ("A", "B"):
            msk = z if g == "A" else ~z
            G, se, R = rel(P[a], P[b], pe, msk)
            truth[(pk, g)] = dict(G=G, se=se, R_ref=R, 近m=bool(abs(G - M) < 2 * se), 近0=bool(abs(G) < 2 * se))
            near += [f"{pk}|{g}|m"] * truth[(pk, g)]["近m"] + [f"{pk}|{g}|0"] * truth[(pk, g)]["近0"]
    for nm, other in (("MIX", "HTFS_D"), ("POOLMIX", "ALT_D")):
        for g in ("A", "B"):
            msk = z if g == "A" else ~z
            t = sel[nm][g][2]
            G, se, R = rel(P["P*_D"], mixed(t, P["P*_D"], P[other]), pe, msk)
            strat[(nm, g)] = dict(G=G, se=se, 近0=bool(se > 0 and abs(G) < 2 * se))
            near += [f"{nm}|{g}|0"] * strat[(nm, g)]["近0"]
    return truth, strat, near


# ── 一个重复 ─────────────────────────────────────────────────────
def run_rep(cond, rep, write=True, runs=None, shuffle_pe=False, cheat=False, force_stages=None, bug_mixed_eval=False):
    import psutil
    import pgss_xgb_engine as EN, pgss_r43_t1 as T1
    if os.environ.get("PGSS_T1_SMOKE") == "1":          # 只用于端到端测试：缩小网格
        EN.GRID_TRACK = EN.GRID_TRACK[:1]; EN.GRID_POOLED = EN.GRID_POOLED[:1]; EN.K_GRID = [5, 23]; EN.MAX_ROUNDS = 60
    runs = RUNS if runs is None else runs
    t0 = time.time(); proc = psutil.Process()
    r_dev, r_ev, seed_dev = streams(cond, rep)
    Xd, zd, pdv, lind = gen(cond, N_DEV, r_dev)
    yd = (r_dev.random(N_DEV) < pdv).astype(int)
    D = dict(X=Xd, y=yd, z=zd, g=np.arange(N_DEV, dtype=np.int64), uid=np.arange(N_DEV, dtype=np.int64))
    Ev0, pe0, lin0 = next(eval_blocks(cond, 0, r_ev[0]))          # 第 0 段 = 1 块（100,000）
    if shuffle_pe:                                   # 端到端测试：评价器之外不得用到 p
        pe0 = np.random.default_rng(0).permutation(pe0)
    names = [f"x{i}" for i in range(Xd.shape[1])]
    lay = EN.task_layout(D["X"], D["z"], names)
    EN.set_namespace(sim="R46-T2", cond=cond, rep=rep, code=code_sha())
    r = EN.develop(D, lay, seed_dev)
    specs, inner = r["specs"], r["inner"]
    fp = EN.final_predict(specs, D, Ev0, lay)
    ch = lambda nm: fp[nm][fp[nm]["chosen"]]
    s0 = T1.star(specs, inner, "S0"); sz = T1.star(specs, inner, "SZ")
    alt = sz if specs[r["pstar"]]["arch"] == "S0" else s0
    specD = {"P*_D": specs[r["pstar"]], "HTFS_D": specs["HTFS"], "Hfull_D": specs["Hfull"], "HGFS_D": specs["HGFS"],
             "S0*_D": specs[s0], "SZ*_D": specs[sz], "ALT_D": specs[alt]}
    P0 = {"P*_D": ch(r["pstar"]), "HTFS_D": ch("HTFS"), "Hfull_D": ch("Hfull"), "HGFS_D": ch("HGFS"),
          "S0*_D": ch(s0), "SZ*_D": ch(sz), "ALT_D": ch(alt)}
    oof = {"P*": T1.oof_of(EN, specs[r["pstar"]], D, lay, seed_dev), "HTFS": T1.oof_of(EN, specs["HTFS"], D, lay, seed_dev),
           "S0*": T1.oof_of(EN, specs[s0], D, lay, seed_dev), "ALT": T1.oof_of(EN, specs[alt], D, lay, seed_dev)}
    n_fit_main = len(EN.FITLOG); t_main = time.time() - t0
    keep, hold = EN.cert_split(D["g"], seed_dev + 50_000)
    F, C = EN.sub(D, keep), EN.sub(D, hold)
    rF = EN.develop(F, lay, seed_dev + 70_000, arms_tracked=("Hfull", "HTFS"))
    s0F = T1.star(rF["specs"], rF["inner"], "S0")
    specF = {"P*_F": rF["specs"][rF["pstar"]], "HTFS_F": rF["specs"]["HTFS"], "S0*_F": rF["specs"][s0F]}
    pc = {k[:-2]: EN.program_predict(v, F, C, lay) for k, v in specF.items()}
    for k, v in specF.items():
        P0[k] = EN.program_predict(v, F, Ev0, lay)
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
    sel = T1.selectors(D, EN, cert, oof, cheat=((pe0, Ev0["z"], P0) if cheat else None))
    yf = D["y"].astype(float)
    sel["POOLMIX"] = {}
    for g in ("A", "B"):
        m_ = D["z"] if g == "A" else ~D["z"]
        sc = {t: float(np.mean((yf[m_] - ((1 - t) * oof["P*"][m_] + t * oof["ALT"][m_])) ** 2)) for t in T_GRID}
        tb = min(T_GRID, key=lambda t: (sc[t], t))
        sel["POOLMIX"][g] = (None, "POOLMIX", tb, sc[tb], sc[0.0])
    # ── 评价器：自适应规模 ──
    zs, pes, Ps = [Ev0["z"]], [pe0], {k: [v] for k, v in P0.items()}
    ext = dict(X_absmax=float(np.abs(Ev0["X"]).max()), X_abs_q999=float(np.quantile(np.abs(Ev0["X"]), .999)),
               X_frac_abs_gt4=float((np.abs(Ev0["X"]) > 4).mean()), lin_q001=float(np.quantile(lin0, .001)),
               lin_q999=float(np.quantile(lin0, .999)), lin_min=float(lin0.min()), lin_max=float(lin0.max()),
               p_frac_lt1e6=float((pe0 < 1e-6).mean()), p_frac_gt1m1e6=float((pe0 > 1 - 1e-6).mean()),
               dev_X_absmax=float(np.abs(Xd).max()), dev_event_rate=float(yd.mean()), dev_mean_p=float(pdv.mean()))
    del Ev0
    stage, trig = 0, []
    while True:
        P = {k: np.concatenate(v) for k, v in Ps.items()}; pe = np.concatenate(pes); z = np.concatenate(zs)
        truth, strat, near = evaluate(P, pe, z, sel)
        want = (force_stages is not None and stage < force_stages) or (force_stages is None and near)
        if not want or stage == len(STAGES) - 1:
            break
        trig.append(dict(段=stage, 触发项=near))
        stage += 1
        for b, (Ev, pe_s, _) in enumerate(eval_blocks(cond, stage, r_ev[stage])):
            if shuffle_pe:
                pe_s = np.random.default_rng(100 * stage + b).permutation(pe_s)
            for k, sp in specD.items():
                Ps[k].append(EN.program_predict(sp, D, Ev, lay))
            for k, sp in specF.items():
                Ps[k].append(EN.program_predict(sp, F, Ev, lay))
            zs.append(Ev["z"]); pes.append(pe_s)
            del Ev
    n_eval = len(pe)
    Pa = {k: v[: STAGES[0]] for k, v in P.items()} if bug_mixed_eval else P      # 变异对照：臂与选择器只用第 0 段
    pea, za = (pe[: STAGES[0]], z[: STAGES[0]]) if bug_mixed_eval else (pe, z)
    arm_rows = [dict(臂=k, R_A=float(uloss(v[za], pea[za]).mean()), R_B=float(uloss(v[~za], pea[~za]).mean()),
                     R_pop=float(uloss(v, pea).mean())) for k, v in dict(Pa, ORACLE=pea).items()]
    pair_rows = [dict(候选对=pk, 轨=g, 参照=PAIRS[pk][0], 候选=PAIRS[pk][1], **v) for (pk, g), v in truth.items()]
    sel_rows = []
    for name, dec in sel.items():
        for g in ("A", "B"):
            spec_, src, t_, b1, b2 = dec[g]
            pp = mixed(t_, Pa["P*_D"], Pa["HTFS_D"]) if src == "MIX" else mixed(t_, Pa["P*_D"], Pa["ALT_D"]) if src == "POOLMIX" else Pa[src]
            msk = za if g == "A" else ~za
            Gs, ses, Rp = rel(Pa["P*_D"], pp, pea, msk)
            Rg = float(uloss(pp[msk], pea[msk]).mean()); Rh = float(uloss(Pa["HTFS_D"][msk], pea[msk]).mean())
            rule_pair = ("F|" if name.startswith(("FCT", "USTUN")) else "D|") + ("S0*" if name.endswith("S0*") else "P*")
            tr = truth[(rule_pair, g)]
            hard = spec_ is not None
            sel_rows.append(dict(选择器=name, 轨=g, 专业化=spec_, 部署=src, t=t_, 依据1=b1, 依据2=b2, R=Rg, R_PstarD=Rp, R_HTFSD=Rh,
                                 相对PstarD改善=Gs, 相对PstarD改善_积分SE=ses,
                                 部署实际变差=bool(Gs < 0 and abs(Gs) >= 2 * ses), 部署变差边界不确定=bool(ses > 0 and abs(Gs) < 2 * ses),
                                 判断所依候选对=rule_pair, 候选对真值相对收益=tr["G"], 候选对积分SE=tr["se"],
                                 候选对达m=bool(tr["G"] >= M), 候选对近m=tr["近m"], 候选对近0=tr["近0"],
                                 不达标启用=(bool(spec_ and tr["G"] < M and not tr["近m"]) if hard else None),
                                 启用_m边界不确定=(bool(spec_ and tr["近m"]) if hard else None),
                                 实际变差启用=(bool(spec_ and tr["G"] < 0 and not tr["近0"]) if hard else None),
                                 漏启用=(bool((not spec_) and tr["G"] >= M and not tr["近m"]) if hard else None),
                                 未启用_m边界不确定=(bool((not spec_) and tr["近m"]) if hard else None)))
    za_d = D["z"]
    meta = dict(tag=tag_of(cond, rep), cond=cond, rep=rep, code_sha=code_sha(), engine=EN.ENGINE_VERSION, seed_dev=seed_dev,
                n_dev=N_DEV, n_A_dev=int(za_d.sum()), events_A_dev=int(yd[za_d].sum()), events_B_dev=int(yd[~za_d].sum()),
                pstar_D=r["pstar"], pstar_cal=bool(specs[r["pstar"]]["cal"]), S0star_D=s0, ALT=alt,
                HTFS_kA=specs["HTFS"]["kA"], HTFS_kB=specs["HTFS"]["kB"], pstar_F=rF["pstar"],
                n_F=len(keep), n_C=len(hold), n_eval=int(n_eval), 评价段=stage, 追加记录=trig,
                最终仍近边界=near, n_eval_A=int(z.sum()),
                eval_event_rate=float(pe.mean()), eval_pA=float(pe[z].mean()), eval_pB=float(pe[~z].mean()),
                eval_bayes_A=float((pe[z] * (1 - pe[z])).mean()), eval_bayes_B=float((pe[~z] * (1 - pe[~z])).mean()),
                极端值=ext, cert={k: {g: dict(v[g]) for g in ("A", "B")} for k, v in cert.items()},
                sec_total=round(time.time() - t0, 1), sec_main=round(t_main, 1), sec_cert=round(t_cert, 1),
                n_fits=len(EN.FITLOG), n_fits_main=n_fit_main,
                peak_rss_mb=round(getattr(proc.memory_info(), "peak_wset", proc.memory_info().rss) / 2 ** 20, 1),
                bug_mixed_eval=bug_mixed_eval)
    A_, PR, S_ = pd.DataFrame(arm_rows), pd.DataFrame(pair_rows), pd.DataFrame(sel_rows)
    if write:
        od = runs / tag_of(cond, rep); od.mkdir(parents=True, exist_ok=True)
        A_.to_parquet(od / "arms.parquet", index=False); PR.to_parquet(od / "pairs.parquet", index=False)
        S_.to_parquet(od / "sel.parquet", index=False)
        pd.DataFrame(EN.FITLOG).to_parquet(od / "fitlog.parquet", index=False)
        (od / "rep.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        (od / "_DONE").write_text(str(meta["sec_total"]), encoding="utf-8")
    return meta, A_, PR, S_


def _entry(args):
    cond, rep = args
    try:
        run_rep(cond, rep)
        return args, "OK"
    except Exception as e:
        od = RUNS / tag_of(cond, rep); od.mkdir(parents=True, exist_ok=True)
        (od / "_FAILED").write_text(f"{type(e).__name__}: {e}\n{traceback.format_exc()[-3000:]}", encoding="utf-8")
        return args, "FAILED"


def probe():
    """内存与耗时探针（不计入 1,600 次）：P2-H rep 0 强制追加到 1M，写到 t2/probe/。用于按方案 §9.4 定并发。"""
    run_rep("P2-H", 0, write=True, runs=T2DIR / "probe", force_stages=len(STAGES) - 1)


def job_list(only):
    if only == "cost":
        return [(c, 0) for c in CONDS]
    return [(c, rep) for rep in range(REPS) for c in CONDS]


def run(jobs, only):
    from concurrent.futures import ProcessPoolExecutor
    RUNS.mkdir(parents=True, exist_ok=True)
    jl = [a for a in job_list(only) if not (RUNS / tag_of(*a) / "_DONE").exists()]
    tot = len(job_list(only)); ok = fail = 0; t0 = time.time()
    with ProcessPoolExecutor(max_workers=jobs, max_tasks_per_child=1) as ex:
        for args, res in ex.map(_entry, jl):
            ok += res == "OK"; fail += res == "FAILED"
            (T2DIR / f"_STATUS_{only}.json").write_text(json.dumps(dict(批次=only, 总数=tot, 本次待跑=len(jl), 完成=ok, 失败=fail,
                                                                       已用分钟=round((time.time() - t0) / 60, 1),
                                                                       更新=time.strftime("%m-%d %H:%M:%S")), ensure_ascii=False, indent=1),
                                                        encoding="utf-8")
            print(args, res, flush=True)
    (T2DIR / (f"_DONE_{only}" if fail == 0 else f"_FAILED_{only}")).write_text(f"ok={ok} fail={fail}", encoding="utf-8")
    return ok, fail


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else "all"
        run(int(sys.argv[2]), only)
    elif cmd == "probe":
        probe()
    elif cmd == "one":
        m, A, PR, S = run_rep(sys.argv[2], int(sys.argv[3]), write=False)
        print(json.dumps({k: v for k, v in m.items() if k != "cert"}, ensure_ascii=False, default=float, indent=1))
        print(A.to_string()); print(PR.to_string()); print(S.to_string())
