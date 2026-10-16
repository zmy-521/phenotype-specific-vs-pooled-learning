# -*- coding: utf-8 -*-
r"""R42 正式主实验的统计汇总（SAP v1.0 §S2/S4/S5/S6）。只读 runs/ 产物，不拟合任何模型。

推断范围：**fixed-OOF-prediction cluster-resampling intervals**（第 Ⅱ 层）。三个 seed 不是三批独立患者；
max-T 只处理多重比较，不补回未重新训练的不确定性。

1. 逐单位损失：每个 seed 内 5 个外层折拼成全队列预测，Brier 损失对 3 个 seed 取均值。
2. 臂（每个外层折内的开发选择，测试时冻结）：
   P* = 该折选出的合并候选（其选定的校准版本）；HTFS/Hfull/HGFS = 选定版本；
   S0*/SZ*/SZ-GFS*/SZ-union* = 该架构在两种权重中内层分数较小者的选定版本；
   E2′：Hfull 与 SZ* 的 raw-vs-raw、cal-vs-cal；认证：CERT（认证后程序）、P*_F、HTFS_F。
3. 对照：E1 HTFS vs P*；E2 Hfull vs P*；E2′；E3 SZ* vs S0*；E4 HTFS vs SZ-union*；E5 SZ-union* vs SZ*；
   E6 SZ-GFS* vs SZ*；E6′ SZ-union* vs SZ-GFS*；E7 HTFS vs HGFS；CERT vs P*_D；CERT vs P*_F。
   尺度：逐轨、两轨等权、按人口加权；绝对与相对 Brier 差；正值 = 候选更好。
4. 联合簇重抽：统一注册表（MIMIC=患者；eICU=全局患者—医院连通分量），每个 replicate 一套多项权重作用于该库全部任务，
   B=2000，SeedSequence(20421).spawn(2) 两库独立；百分位区间；三态判定（m ∈ {0.5%,1%,2%}）。
5. max-T：θ_tg = (1−m)R_tg(P*) − R_tg(HTFS)，48 项（24 任务 × 2 轨），按各项重抽标准差标准化；零方差项单列、
   空轨 replicate 记缺失并计数，不删除。
6. eICU 敏感性：E1 每轨的分量级 t 区间（自由度 G−1）。
7. 认证：逐单元决定、各状态比例；同一候选对下「认证决定 × 测试折上实现的相对收益是否 ≥ m」的描述性交叉表。
8. 学习曲线：25/50/75% 各点的 HTFS vs P*、Hfull vs P*（可评价点；完整度标注），100% 复用主实验。

用法：python pgss_r42_inference.py [--partial]   （--partial：只汇总已完成的任务，用于检查脚本）
输出：23_R42_正式主实验/inference/R42_INFERENCE.xlsx、R42_INFERENCE.json
"""
import sys, json, pathlib, time
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pgss_threads  # noqa: F401
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from scipy.stats import t as tdist

R42 = HERE.parent / "23_R42_正式主实验"
RUNS = R42 / "runs"; OUT = R42 / "inference"
SEEDS = [20261003, 20261004, 20261005]
OUTCOMES = ["OUT_HOSP_MORT", "OUT_LOS72"]
PHENOS = ["女性", "年龄≥65", "贫血", "eGFR降低", "低白蛋白", "肥胖"]
B_BOOT, CHUNK, ROOT_SEED = 2000, 100, 20421
MARGINS = (0.005, 0.01, 0.02)
M_MAIN = 0.01
POOLED = ("S0", "SZ", "SZ-GFS", "SZ-union")
CONTRASTS = [("E1", "P*", "HTFS"), ("E2", "P*", "Hfull"), ("E2′raw", "SZ*raw", "Hfull_raw"),
             ("E2′cal", "SZ*cal", "Hfull_cal"), ("E3", "S0*", "SZ*"), ("E4", "SZ-union*", "HTFS"),
             ("E5", "SZ*", "SZ-union*"), ("E6", "SZ*", "SZ-GFS*"), ("E6′", "SZ-GFS*", "SZ-union*"),
             ("E7", "HGFS", "HTFS"), ("CERT_vs_P*D", "P*@C", "CERT"), ("CERT_vs_P*F", "P*_F", "CERT")]


def tag(db, oc, ph, seed, fold, br, fr=None):
    return f"{db}__{oc}__{ph}__{seed}__{fold}__{br}" + (f"__{int(round(100 * fr))}" if fr else "")


def arm_columns(P, sp):
    """把一个 main 单元的测试预测映射为臂（按该折的开发选择）。"""
    specs, inner, pstar = sp["specs"], sp["inner"], sp["pstar"]
    ch = lambda name: "cal" if specs[name]["cal"] else "raw"
    out = {"P*": P[f"{pstar}|{ch(pstar)}"].to_numpy()}
    for nm in ("HTFS", "Hfull", "HGFS"):
        out[nm] = P[f"{nm}|{ch(nm)}"].to_numpy()
    for a in POOLED:
        ws = [w for w in ("none", "bal") if f"{a}|{w}" in inner]
        w = min(ws, key=lambda w: (inner[f"{a}|{w}"], ws.index(w)))
        out[f"{a}*"] = P[f"{a}|{w}|{ch(f'{a}|{w}')}"].to_numpy()
        if a == "SZ":
            out["SZ*raw"] = P[f"SZ|{w}|raw"].to_numpy(); out["SZ*cal"] = P[f"SZ|{w}|cal"].to_numpy()
    out["Hfull_raw"] = P["Hfull|raw"].to_numpy(); out["Hfull_cal"] = P["Hfull|cal"].to_numpy()
    return out


def collect_task(db, oc, ph, partial):
    """返回 (uid, y, z, {arm: 逐单位 seed 平均损失}, 元信息) 或 None（不完整）。"""
    per_seed = []
    meta = dict(main_units=0, cert_units=0, missing=[])
    for seed in SEEDS:
        parts = []
        for fold in range(5):
            dm = RUNS / tag(db, oc, ph, seed, fold, "main"); dc = RUNS / tag(db, oc, ph, seed, fold, "cert")
            if not (dm / "_DONE").exists():
                meta["missing"].append(dm.name); continue
            P = pd.read_parquet(dm / "preds_test.parquet")
            sp = json.loads((dm / "specs.json").read_text(encoding="utf-8"))
            cols = arm_columns(P, sp)
            if (dc / "_DONE").exists():
                Q = pd.read_parquet(dc / "preds_test.parquet").set_index("unit_id").loc[P.unit_id]
                cols["CERT"] = Q.certified_program.to_numpy(); cols["P*_F"] = Q.Pstar_F.to_numpy()
                cols["HTFS_F"] = Q.HTFS_F.to_numpy(); meta["cert_units"] += 1
            else:
                meta["missing"].append(dc.name)
            meta["main_units"] += 1
            parts.append(pd.DataFrame({"unit_id": P.unit_id, "y": P.y, "z": P.z, **cols}))
        if parts:
            per_seed.append(pd.concat(parts).set_index("unit_id").sort_index())
    if len(per_seed) < 3 or meta["main_units"] < 15:
        if not partial or not per_seed:
            return None, meta
    base = per_seed[0]
    arms = [c for c in base.columns if c not in ("y", "z")]
    common = base.index
    for d in per_seed[1:]:
        common = common.intersection(d.index)
    y = base.loc[common, "y"].to_numpy(float); z = base.loc[common, "z"].to_numpy(bool)
    L = {}
    for a in arms:
        if all(a in d.columns for d in per_seed):
            L[a] = np.mean([(d.loc[common, a].to_numpy(float) - y) ** 2 for d in per_seed], axis=0)
    if "CERT" in L:
        valid = np.isfinite(L["CERT"])
        L["P*@C"] = np.where(valid, L["P*"], np.nan)
        meta["认证比较纳入单位比例"] = round(float(valid.mean()), 4)
    meta["n_units"] = len(common); meta["seeds_used"] = len(per_seed)
    return (common.to_numpy(np.int64), y, z, L), meta


def cluster_ids(db, uid):
    import pgss_r38_bench as BN, pgss_r381_joint_inference as JI
    X, P, OK, meta = BN.load(db)
    gid = pd.Series(meta.group_id.astype(str).to_numpy(), index=X.index.to_numpy().astype(np.int64))
    gmap = JI.global_cluster_map(db)
    return np.array([gmap[g] for g in gid.loc[uid].to_numpy()], dtype=object)


def risks_obs(L, z):
    return {a: (float(np.nanmean(v[z])), float(np.nanmean(v[~z])), float(np.nanmean(v))) for a, v in L.items()}


def contrast_vals(RA, RB, RP, ref, cand):
    """返回 dict(scope -> (abs, rel))；RA/RB/RP 是 {arm: 值或向量}。"""
    out = {}
    for sc, Rr, Rc in (("A", RA[ref], RA[cand]), ("B", RB[ref], RB[cand]),
                       ("bal", .5 * (RA[ref] + RB[ref]), .5 * (RA[cand] + RB[cand])), ("pop", RP[ref], RP[cand])):
        d = Rr - Rc
        out[sc] = (d, np.where(Rr > 0, d / Rr, np.nan))
    return out


def main(partial=False):
    import pgss_r38_bench as BN
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    tasks, metas = {}, []
    for db in ("mimic", "eicu"):
        for oc in OUTCOMES:
            for ph in PHENOS:
                r, m = collect_task(db, oc, ph, partial)
                metas.append(dict(库=db, 结局=oc, 表型=ph, 纳入=r is not None, **{k: v for k, v in m.items() if k != "missing"},
                                  缺失单元数=len(m["missing"])))
                if r is not None:
                    tasks[(db, oc, ph)] = r
        print(f"  {db} 读入完成 {time.time()-t0:.0f}s", flush=True)
    rows, boots, abs_boots = [], {}, {}
    ss = dict(zip(("mimic", "eicu"), np.random.SeedSequence(ROOT_SEED).spawn(2)))
    for db in ("mimic", "eicu"):
        keys = [k for k in tasks if k[0] == db]
        if not keys:
            continue
        cl = {k: cluster_ids(db, tasks[k][0]) for k in keys}
        reg = np.array(sorted(set().union(*[set(v) for v in cl.values()])), dtype=object)
        slot = {c: i for i, c in enumerate(reg)}; M = len(reg)
        agg = {}
        for k in keys:
            uid, y, z, L = tasks[k]
            s = np.fromiter((slot[c] for c in cl[k]), np.int64, len(uid))
            a = {}
            for arm, v in L.items():
                ok = np.isfinite(v)
                a[f"{arm}|nA"] = np.bincount(s[z & ok], minlength=M).astype(float)
                a[f"{arm}|nB"] = np.bincount(s[~z & ok], minlength=M).astype(float)
                a[f"{arm}|A"] = np.bincount(s[z & ok], weights=v[z & ok], minlength=M)
                a[f"{arm}|B"] = np.bincount(s[~z & ok], weights=v[~z & ok], minlength=M)
            agg[k] = a
            # 观察值
            Ro = risks_obs(L, z)
            RA = {a_: Ro[a_][0] for a_ in Ro}; RB = {a_: Ro[a_][1] for a_ in Ro}; RP = {a_: Ro[a_][2] for a_ in Ro}
            for cn, ref, cand in CONTRASTS:
                if ref in L and cand in L:
                    for sc, (dd, rr) in contrast_vals(RA, RB, RP, ref, cand).items():
                        rows.append(dict(库=db, 结局=k[1], 表型=k[2], 对照=cn, 参照=ref, 候选=cand, 尺度=sc,
                                         观察绝对差=float(dd), 观察相对改善=float(rr), R参照=float(
                                             {"A": RA[ref], "B": RB[ref], "bal": .5 * (RA[ref] + RB[ref]), "pop": RP[ref]}[sc])))
        rng = np.random.default_rng(ss[db]); done = 0
        while done < B_BOOT:
            b = min(CHUNK, B_BOOT - done)
            draw = rng.integers(0, M, size=(M, b))
            Mt = np.zeros((M, b))
            for j in range(b):
                Mt[:, j] = np.bincount(draw[:, j], minlength=M)
            for k in keys:
                a = agg[k]
                arms = [x[:-2] for x in a if x.endswith("|A")]
                RA, RB, RP = {}, {}, {}
                for x in arms:
                    dA = a[f"{x}|nA"] @ Mt; dB = a[f"{x}|nB"] @ Mt
                    sA = a[f"{x}|A"] @ Mt; sB = a[f"{x}|B"] @ Mt
                    RA[x] = np.where(dA > 0, sA / np.where(dA > 0, dA, 1), np.nan)
                    RB[x] = np.where(dB > 0, sB / np.where(dB > 0, dB, 1), np.nan)
                    RP[x] = np.where(dA + dB > 0, (sA + sB) / np.where(dA + dB > 0, dA + dB, 1), np.nan)
                for cn, ref, cand in CONTRASTS:
                    if ref in RA and cand in RA:
                        for sc, (dd, rr) in contrast_vals(RA, RB, RP, ref, cand).items():
                            boots.setdefault((db, k[1], k[2], cn, sc), []).append(rr)
                            abs_boots.setdefault((db, k[1], k[2], cn, sc), []).append(dd)
                for g, Rg in (("A", RA), ("B", RB)):
                    th = (1 - M_MAIN) * Rg["P*"] - Rg["HTFS"]
                    boots.setdefault((db, k[1], k[2], "θ", g), []).append(th)
            done += b
            print(f"  {db} 重抽 {done}/{B_BOOT}  {time.time()-t0:.0f}s", flush=True)
    for key in list(boots):
        boots[key] = np.concatenate(boots[key])
    for key in list(abs_boots):
        abs_boots[key] = np.concatenate(abs_boots[key])
    T = pd.DataFrame(rows)
    lo, hi, alo, ahi, nmiss = [], [], [], [], []
    for _, r in T.iterrows():
        v = boots[(r.库, r.结局, r.表型, r.对照, r.尺度)]; va = abs_boots[(r.库, r.结局, r.表型, r.对照, r.尺度)]
        lo.append(np.nanquantile(v, .025)); hi.append(np.nanquantile(v, .975))
        alo.append(np.nanquantile(va, .025)); ahi.append(np.nanquantile(va, .975)); nmiss.append(int(np.isnan(v).sum()))
    T["相对_P2.5"] = lo; T["相对_P97.5"] = hi; T["绝对_P2.5"] = alo; T["绝对_P97.5"] = ahi; T["缺失replicate"] = nmiss
    for m in MARGINS:
        T[f"三态_m{m:.3f}"] = np.where(T["相对_P2.5"] > m, "支持实质收益", np.where(T["相对_P97.5"] < m, "排除实质收益", "证据不足"))
    T["相对0方向"] = np.where(T["相对_P2.5"] > 0, "全正", np.where(T["相对_P97.5"] < 0, "全负", "跨零"))
    # max-T（E1 θ，48 项）
    th_rows = []
    for (db, oc, ph, cn, g), v in boots.items():
        if cn != "θ":
            continue
        uid, y, z, L = tasks[(db, oc, ph)]
        m_ = z if g == "A" else ~z
        theta = (1 - M_MAIN) * L["P*"][m_].mean() - L["HTFS"][m_].mean()
        th_rows.append(dict(库=db, 结局=oc, 表型=ph, 轨=g, θ观察=float(theta), θ重抽SD=float(np.nanstd(v)),
                            缺失replicate=int(np.isnan(v).sum()), _v=v))
    TH = pd.DataFrame(th_rows)
    use = TH.θ重抽SD > 0
    if use.any():
        Z = np.vstack([(r._v - r.θ观察) / r.θ重抽SD for _, r in TH[use].iterrows()])
        c95 = float(np.nanquantile(np.nanmax(np.abs(Z), axis=0), .95))
    else:
        c95 = np.nan
    TH["max-T临界值"] = c95
    TH["同时下界"] = TH.θ观察 - c95 * TH.θ重抽SD; TH["同时上界"] = TH.θ观察 + c95 * TH.θ重抽SD
    TH["零方差单列"] = ~use
    TH["同时判定_相对m"] = np.where(TH.同时下界 > 0, "支持（收益>m）", np.where(TH.同时上界 < 0, "排除（收益<m）", "证据不足"))
    TH = TH.drop(columns="_v")
    # eICU 敏感性：E1 每轨分量级 t 区间
    tt = []
    for (db, oc, ph), (uid, y, z, L) in tasks.items():
        if db != "eicu":
            continue
        cl = cluster_ids(db, uid)
        for g in ("A", "B"):
            m_ = z if g == "A" else ~z
            d = L["P*"][m_] - L["HTFS"][m_]
            codes, inv = np.unique(cl[m_], return_inverse=True)
            Dc = np.bincount(inv, weights=d); nc = np.bincount(inv).astype(float)
            G = len(codes); dbar = Dc.sum() / nc.sum()
            se = float(np.sqrt(G / (G - 1) * np.sum((Dc - dbar * nc) ** 2)) / nc.sum())
            q = tdist.ppf(.975, G - 1); Rr = L["P*"][m_].mean()
            tt.append(dict(库=db, 结局=oc, 表型=ph, 轨=g, 分量数=G, 绝对差=dbar, t下界=dbar - q * se, t上界=dbar + q * se,
                           相对差=dbar / Rr, 相对t下界=(dbar - q * se) / Rr, 相对t上界=(dbar + q * se) / Rr))
    TT = pd.DataFrame(tt)
    # 认证：逐单元决定与描述性交叉表
    cr = []
    for (db, oc, ph) in tasks:
        for seed in SEEDS:
            for fold in range(5):
                dc = RUNS / tag(db, oc, ph, seed, fold, "cert")
                if not (dc / "_DONE").exists():
                    continue
                cj = json.loads((dc / "cert.json").read_text(encoding="utf-8"))
                Q = pd.read_parquet(dc / "preds_test.parquet")
                for g in ("A", "B"):
                    m_ = Q.z.to_numpy(bool) if g == "A" else ~Q.z.to_numpy(bool)
                    yy = Q.y.to_numpy(float)[m_]
                    rr = ((yy - Q.Pstar_F.to_numpy()[m_]) ** 2).mean(); rs = ((yy - Q.HTFS_F.to_numpy()[m_]) ** 2).mean()
                    pr = cj[g]["primary"]
                    cr.append(dict(库=db, 结局=oc, 表型=ph, seed=seed, 外层折=fold, 轨=g, 专业化=cj[g]["specialize"],
                                   认证参照=pr["ref"], 认证p=pr["p"], 认证SE=pr["se"], 认证簇数=pr["clusters"],
                                   测试折实现相对收益=(rr - rs) / rr, 测试折实现达m=bool((rr - rs) / rr >= M_MAIN)))
    CR = pd.DataFrame(cr)
    CX = (CR.groupby(["库", "专业化", "测试折实现达m"]).size().rename("单元×轨").reset_index() if len(CR) else pd.DataFrame())
    CS = (CR.groupby(["库", "轨"]).agg(单元=("专业化", "size"), 专业化=("专业化", "sum")).reset_index() if len(CR) else pd.DataFrame())
    # 学习曲线
    lc = []
    for (db, oc, ph), (uid, y, z, L) in tasks.items():
        for fr in (0.25, 0.5, 0.75):
            per_seed = []
            for seed in SEEDS:
                parts = []
                for fold in range(5):
                    dl = RUNS / tag(db, oc, ph, seed, fold, "lc", fr)
                    if (dl / "_DONE").exists():
                        P = pd.read_parquet(dl / "preds_test.parquet")
                        sp = json.loads((dl / "specs.json").read_text(encoding="utf-8"))
                        ch = lambda n: "cal" if sp["specs"][n]["cal"] else "raw"
                        parts.append(pd.DataFrame({"unit_id": P.unit_id, "y": P.y, "z": P.z,
                                                   "P*": P[f"{sp['pstar']}|{ch(sp['pstar'])}"],
                                                   "HTFS": P[f"HTFS|{ch('HTFS')}"], "Hfull": P[f"Hfull|{ch('Hfull')}"]}))
                if parts:
                    per_seed.append(pd.concat(parts).set_index("unit_id"))
            if not per_seed:
                lc.append(dict(库=db, 结局=oc, 表型=ph, 比例=fr, 完整度=0.0)); continue
            common = per_seed[0].index
            for d in per_seed[1:]:
                common = common.intersection(d.index)
            yy = per_seed[0].loc[common, "y"].to_numpy(float); zz = per_seed[0].loc[common, "z"].to_numpy(bool)
            LL = {a: np.mean([(d.loc[common, a].to_numpy(float) - yy) ** 2 for d in per_seed], axis=0)
                  for a in ("P*", "HTFS", "Hfull")}
            row = dict(库=db, 结局=oc, 表型=ph, 比例=fr, 完整度=round(len(common) / len(uid), 4), seed数=len(per_seed))
            for cn, cand in (("HTFS", "HTFS"), ("Hfull", "Hfull")):
                for g, m_ in (("A", zz), ("B", ~zz)):
                    rref = LL["P*"][m_].mean(); row[f"{cn}_vs_P*_{g}_相对"] = (rref - LL[cand][m_].mean()) / rref
                rb = .5 * (LL["P*"][zz].mean() + LL["P*"][~zz].mean()); rc = .5 * (LL[cand][zz].mean() + LL[cand][~zz].mean())
                row[f"{cn}_vs_P*_bal_相对"] = (rb - rc) / rb
            lc.append(row)
    LC = pd.DataFrame(lc)
    summ = dict(纳入任务=len(tasks), 任务总数=24, B=B_BOOT, max_T临界值=c95, partial=partial, 秒=round(time.time() - t0))
    with pd.ExcelWriter(OUT / "R42_INFERENCE.xlsx", engine="openpyxl") as xw:
        pd.DataFrame([summ]).to_excel(xw, sheet_name="0_汇总", index=False)
        pd.DataFrame(metas).to_excel(xw, sheet_name="1_任务完整度", index=False)
        T.to_excel(xw, sheet_name="2_全部对照", index=False)
        TH.to_excel(xw, sheet_name="3_E1_maxT", index=False)
        TT.to_excel(xw, sheet_name="4_eICU_t敏感性", index=False)
        CR.to_excel(xw, sheet_name="5_认证逐单元", index=False)
        CS.to_excel(xw, sheet_name="6_认证汇总", index=False)
        CX.to_excel(xw, sheet_name="7_认证交叉表_描述", index=False)
        LC.to_excel(xw, sheet_name="8_学习曲线", index=False)
    (OUT / "R42_INFERENCE.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main(partial="--partial" in sys.argv)
