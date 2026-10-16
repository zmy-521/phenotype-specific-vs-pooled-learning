# -*- coding: utf-8 -*-
r"""R43 第一部分：R42 冻结结果的只读补齐（GPT 第 48 轮 §3）。**不拟合任何模型**，只读 R42 的 runs/ 产物。

1. 闭环核对（§3.2）：逐单元核对预测产物与**预定单位集合**（不用集合交集）——
   main/cert/lc 的测试集合 = 按 LR 冻结外层划分重算的测试折；每个 seed 五折拼起来 = 任务全体；
   单位唯一；y、z 与冻结数据一致；预测有限且在 [1e-6, 1−1e-6]；产物指纹 = 运行记录；
   cert 的 C 集合 = 按冻结种子重算的认证划分；lc 的子集规模 = 按冻结种子重算的嵌套子集。
2. 辅助指标（§3.1）：各 seed 的完整 OOF 分别计算 Brier、log loss、AUROC、AUPRC、平均预测、观察事件率、
   校准截距（斜率固定 1 的 offset 模型）与校准斜率；逐轨与总体；保留三个 seed 的值与配对差值。
   **不先平均三个 seed 的概率**。
3. 联合重抽复现与扩展（§3.2、§3.3）：与 R42 完全相同的注册表、SeedSequence(20421)、分块抽样顺序
   ⇒ 同一套 replicate 权重；先核对 R42 的全部对照逐项不变，再加：HTFS vs S0*、P*_F vs P*_D（认证的样本成本）、
   log loss 版本的主要对照。导出**逐 replicate 的各臂风险**（全任务人群汇总量），交付后可脱离患者级数据重算区间与 max-T。
4. 认证分解（§3.3）：同一评价对象、同一绝对尺度上 R(P*_D)−R(CERT) = {R(P*_D)−R(P*_F)} + {R(P*_F)−R(CERT)}。
5. 学习曲线（§3.4）：实际训练人数、比例、簇数、逐轨事件数；P*、HTFS、Hfull 各自的绝对 Brier。

**只留本地**（`24_R43_结果补齐与T1/_local_only/`，不打包）：簇级损失汇总（MIMIC 的簇即患者 ⇒ 逐患者派生记录，
按 PhysioNet 2025-09-24 关于 LLM 与在线服务的说明，不交给在线服务）。交付只给其 SHA。

用法：python pgss_r43_readonly.py
输出：24_R43_结果补齐与T1/readonly/R43_READONLY.xlsx、R43_READONLY.json、R43_BOOT_RISKS.parquet
"""
import sys, json, pathlib, time, hashlib
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pgss_threads  # noqa: F401
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import pgss_r42_inference as R42I
from pgss_xgb_run import fsha

RUNS = R42I.RUNS
R43 = HERE.parent / "24_R43_结果补齐与T1"
OUT = R43 / "readonly"; LOCAL = R43 / "_local_only"
SEEDS, OUTCOMES, PHENOS = R42I.SEEDS, R42I.OUTCOMES, R42I.PHENOS
FRACS = (0.25, 0.5, 0.75)
DATA_FP, ENGINE = "1598182eaa10e060", "xgb-1.0"
PLO, PHI = 1e-6, 1 - 1e-6
AUX_ARMS = ["P*", "HTFS", "Hfull", "S0*", "SZ*"]
CERT_ARMS = ["P*@C", "P*_F", "CERT", "HTFS_F"]
PAIRS = [("HTFS", "P*"), ("Hfull", "P*"), ("S0*", "P*"), ("SZ*", "P*"), ("HTFS", "S0*"),
         ("CERT", "P*@C"), ("CERT", "P*_F"), ("P*_F", "P*@C")]
NEW_CONTRASTS = [("E1_vs_S0*", "S0*", "HTFS"), ("P*F_vs_P*D", "P*@C", "P*_F")]
LL_CONTRASTS = [("E1", "P*", "HTFS"), ("E2", "P*", "Hfull"), ("E1_vs_S0*", "S0*", "HTFS"),
                ("CERT_vs_P*D", "P*@C", "CERT"), ("CERT_vs_P*F", "P*_F", "CERT"), ("P*F_vs_P*D", "P*@C", "P*_F")]
LL_ARMS = ["P*", "HTFS", "Hfull", "S0*", "P*@C", "P*_F", "CERT"]


def h16_of_list(xs):
    return hashlib.sha256("|".join(xs).encode()).hexdigest()[:16]


def sha16(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]


# ── 校准诊断：Newton 法（2 参数，无惩罚）────────────────────────
def _logit(p):
    p = np.clip(p, PLO, PHI)
    return np.log(p / (1 - p))


def cal_fit(y, p):
    """返回 (校准截距 CITL[斜率固定 1 的 offset 模型], 校准斜率[logit(y)~a+b·logit(p) 的 b])。不收敛记 NaN。"""
    x = _logit(p); y = np.asarray(y, float)
    a = 0.0
    for _ in range(100):                                  # CITL：offset = x
        mu = 1 / (1 + np.exp(-(a + x))); g = np.sum(y - mu); h = np.sum(mu * (1 - mu))
        step = g / h; a += step
        if abs(step) < 1e-12: break
    else:
        a = np.nan
    th = np.zeros(2); Xd = np.column_stack([np.ones_like(x), x]); ok = False
    for _ in range(100):
        mu = 1 / (1 + np.exp(-(Xd @ th)))
        g = Xd.T @ (y - mu); H = (Xd * (mu * (1 - mu))[:, None]).T @ Xd
        step = np.linalg.solve(H, g); th += step
        if np.max(np.abs(step)) < 1e-10:
            ok = True; break
    return float(a), (float(th[1]) if ok else np.nan)


def metrics(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    pc = np.clip(p, PLO, PHI)
    out = dict(n=int(len(y)), 事件=int(y.sum()), Brier=float(np.mean((y - p) ** 2)),
               logloss=float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
               平均预测=float(p.mean()), 观察事件率=float(y.mean()))
    if 0 < y.sum() < len(y):
        out["AUROC"] = float(roc_auc_score(y, p)); out["AUPRC"] = float(average_precision_score(y, p))
        out["校准截距"], out["校准斜率"] = cal_fit(y, p)
    return out


def logloss_vec(y, p):
    pc = np.clip(p, PLO, PHI)
    return -(y * np.log(pc) + (1 - y) * np.log(1 - pc))


# ── 1. 闭环核对 + 逐 seed 预测 ──────────────────────────────────
def task_pass(db, oc, ph):
    import pgss_r38_bench as BN, pgss_r36_engine as E, pgss_xgb_engine as EN
    from pgss_xgb_pathaudit import task_arrays
    A, names, gs = task_arrays(db, oc, ph)
    uid_all = A["uid"]; pos = pd.Series(np.arange(len(uid_all)), index=uid_all)
    rows, lcrows, per_seed, lc_seed = [], [], {}, {f: {} for f in FRACS}
    for seed in SEEDS:
        folds = E.group_split(A["X"], A["y"], A["z"], gs, BN.K_OUTER, seed)
        parts = []; lc_parts = {f: [] for f in FRACS}
        for fold, (dev, te) in enumerate(folds):
            exp_te = np.sort(uid_all[te]); D = EN.sub(A, dev)
            base = dict(库=db, 结局=oc, 表型=ph, seed=seed, 外层折=fold)

            def check_unit(od, branch, extra=None):
                r = dict(base, 分支=branch, 单元=od.name, 完成=(od / "_DONE").exists(), 失败记录=(od / "_FAILED").exists())
                if not r["完成"]:
                    return r, None
                P = pd.read_parquet(od / "preds_test.parquet"); m = json.loads((od / "meta.json").read_text(encoding="utf-8"))
                u = P.unit_id.to_numpy(np.int64); ii = pos.reindex(u).to_numpy()
                pcols = [c for c in P.columns if c not in ("unit_id", "y", "z")]
                V = P[pcols].to_numpy(float)
                r.update(测试集合一致=bool(np.array_equal(np.sort(u), exp_te)), 单位唯一=bool(P.unit_id.is_unique),
                         y一致=bool(not np.isnan(ii).any() and np.array_equal(A["y"][ii.astype(int)], P.y.to_numpy())),
                         z一致=bool(not np.isnan(ii).any() and np.array_equal(A["z"][ii.astype(int)], P.z.to_numpy(bool))),
                         预测有限=bool(np.isfinite(V).all()), 预测最小=float(np.nanmin(V)), 预测最大=float(np.nanmax(V)),
                         预测在合法范围=bool(np.nanmin(V) >= PLO - 1e-12 and np.nanmax(V) <= PHI + 1e-12),
                         产物指纹一致=bool(all(fsha(od / f) == s for f, s in m["artifacts"].items())),
                         产物数=len(m["artifacts"]), 代码指纹=m["code_sha"], 数据指纹一致=(m["data_fp"] == DATA_FP),
                         引擎一致=(m["engine_version"] == ENGINE), 开发集规模一致=(m["n_dev"] == len(dev)),
                         测试集规模一致=(m["n_test"] == len(te)))
                if extra:
                    r.update(extra(P, m))
                return r, (P, m)

            # main
            r, res = check_unit(RUNS / R42I.tag(db, oc, ph, seed, fold, "main"), "main")
            rows.append(r)
            if res is None:
                raise RuntimeError(f"main 缺失：{r['单元']}")
            P, m = res
            sp = json.loads((RUNS / r["单元"] / "specs.json").read_text(encoding="utf-8"))
            cols = R42I.arm_columns(P, sp)
            # cert
            keep, hold = EN.cert_split(D["g"], seed * 100 + fold + 50_000)
            expC = np.sort(D["uid"][hold])

            def cert_extra(Pc, mc):
                C = pd.read_parquet(RUNS / R42I.tag(db, oc, ph, seed, fold, "cert") / "preds_C.parquet")
                cu = np.sort(C.unit_id.to_numpy(np.int64))
                return dict(认证集合一致=bool(np.array_equal(cu, expC)), 认证集与测试集不交=bool(np.intersect1d(cu, exp_te).size == 0),
                            认证规模一致=bool(mc["n_C"] == len(hold) and mc["n_F"] == len(keep)))
            dc = RUNS / R42I.tag(db, oc, ph, seed, fold, "cert")
            rc, resc = check_unit(dc, "cert", cert_extra)
            rows.append(rc)
            if resc is not None:
                Q = resc[0].set_index("unit_id").loc[P.unit_id]
                cols["CERT"] = Q.certified_program.to_numpy(); cols["P*_F"] = Q.Pstar_F.to_numpy()
                cols["HTFS_F"] = Q.HTFS_F.to_numpy()
            parts.append(pd.DataFrame({"unit_id": P.unit_id, "y": P.y, "z": P.z, **cols}))
            # lc
            for fr in FRACS:
                si = EN.nested_fraction(D["g"], fr, seed * 10 + fold)
                yS, zS = D["y"][si], D["z"][si]
                info = dict(base, 目标比例=fr, 开发集人数=len(dev), 子集人数=len(si), 实际比例=len(si) / len(dev),
                            子集簇数=int(len(np.unique(D["g"][si]))), 开发集簇数=int(len(np.unique(D["g"]))),
                            A事件=int(yS[zS].sum()), A非事件=int((1 - yS[zS]).sum()),
                            B事件=int(yS[~zS].sum()), B非事件=int((1 - yS[~zS]).sum()))

                def lc_extra(Pl, ml):
                    return dict(子集规模与记录一致=bool(ml.get("n_sub") == len(si)),
                                子集簇数与记录一致=bool(ml.get("n_sub_groups") == info["子集簇数"]))
                dl = RUNS / R42I.tag(db, oc, ph, seed, fold, "lc", fr)
                rl, resl = check_unit(dl, "lc", lc_extra)
                rl["目标比例"] = fr; rows.append(rl)
                info["已运行"] = rl["完成"]; lcrows.append(info)
                if resl is not None:
                    Pl, ml = resl
                    spl = json.loads((dl / "specs.json").read_text(encoding="utf-8"))
                    ch = lambda n: "cal" if spl["specs"][n]["cal"] else "raw"
                    lc_parts[fr].append(pd.DataFrame({"unit_id": Pl.unit_id, "y": Pl.y, "z": Pl.z,
                                                      "P*": Pl[f"{spl['pstar']}|{ch(spl['pstar'])}"],
                                                      "HTFS": Pl[f"HTFS|{ch('HTFS')}"], "Hfull": Pl[f"Hfull|{ch('Hfull')}"]}))
        df = pd.concat(parts).set_index("unit_id").sort_index()
        rows.append(dict(库=db, 结局=oc, 表型=ph, seed=seed, 分支="seed全体", 单元="(五折拼接)", 完成=True,
                         测试集合一致=bool(np.array_equal(df.index.to_numpy(np.int64), np.sort(uid_all))),
                         单位唯一=bool(df.index.is_unique)))
        per_seed[seed] = df
        for fr in FRACS:
            if lc_parts[fr]:
                lc_seed[fr][seed] = pd.concat(lc_parts[fr]).set_index("unit_id")
    return A, rows, lcrows, per_seed, lc_seed


# ── 2. 逐 seed 辅助指标 ────────────────────────────────────────
def aux_rows(db, oc, ph, per_seed, cert_common):
    rows, prs = [], []
    for seed, df in per_seed.items():
        y = df.y.to_numpy(float); z = df.z.to_numpy(bool)
        mC = df.index.isin(cert_common)
        d = {a: df[a].to_numpy(float) for a in AUX_ARMS}
        if "CERT" in df:
            d["P*@C"] = np.where(mC, df["P*"].to_numpy(float), np.nan)
            for a in ("CERT", "P*_F", "HTFS_F"):
                d[a] = np.where(mC, df[a].to_numpy(float), np.nan)
        M = {}
        for a, p in d.items():
            for sc, msk in (("A", z), ("B", ~z), ("全体", np.ones_like(z))):
                ok = msk & np.isfinite(p)
                if ok.sum() == 0:
                    continue
                mt = metrics(y[ok], p[ok]); M[(a, sc)] = mt
                rows.append(dict(库=db, 结局=oc, 表型=ph, seed=seed, 臂=a, 范围=sc,
                                 评价对象=("认证共同对象" if a in CERT_ARMS else "全体"), **mt))
            if (a, "A") in M and (a, "B") in M:
                M[(a, "两轨等权")] = {k: .5 * (M[(a, "A")][k] + M[(a, "B")][k]) for k in ("Brier", "logloss")}
                rows.append(dict(库=db, 结局=oc, 表型=ph, seed=seed, 臂=a, 范围="两轨等权",
                                 评价对象=("认证共同对象" if a in CERT_ARMS else "全体"), **M[(a, "两轨等权")]))
        for c, r in PAIRS:
            for sc in ("A", "B", "全体", "两轨等权"):
                if (c, sc) in M and (r, sc) in M:
                    row = dict(库=db, 结局=oc, 表型=ph, seed=seed, 候选=c, 参照=r, 范围=sc)
                    for k in ("Brier", "logloss", "AUROC", "AUPRC", "校准截距", "校准斜率"):
                        if k in M[(c, sc)] and k in M[(r, sc)]:
                            row[f"{k}_候选减参照"] = M[(c, sc)][k] - M[(r, sc)][k]
                    for k in ("Brier", "logloss"):
                        if k in M[(c, sc)]:
                            row[f"{k}_相对改善"] = (M[(r, sc)][k] - M[(c, sc)][k]) / M[(r, sc)][k]
                    prs.append(row)
    return rows, prs


# ── 3. 联合重抽（与 R42 同一套权重）────────────────────────────
def joint_bootstrap(tasks):
    """tasks[k] = (uid, y, z, L_brier, L_logloss)；返回 {(db,oc,ph,metric,arm,scope): 长度 B 的 replicate 风险}。"""
    reps = {}
    ss = dict(zip(("mimic", "eicu"), np.random.SeedSequence(R42I.ROOT_SEED).spawn(2)))
    local_rows = []
    for db in ("mimic", "eicu"):
        keys = [k for k in tasks if k[0] == db]
        cl = {k: R42I.cluster_ids(db, tasks[k][0]) for k in keys}
        reg = np.array(sorted(set().union(*[set(v) for v in cl.values()])), dtype=object)
        slot = {c: i for i, c in enumerate(reg)}; M = len(reg)
        agg = {}
        for k in keys:
            uid, y, z, L, LL = tasks[k]
            s = np.fromiter((slot[c] for c in cl[k]), np.int64, len(uid))
            a = {}
            for met, Ls in (("brier", L), ("logloss", LL)):
                for arm, v in Ls.items():
                    ok = np.isfinite(v)
                    for sc, msk in (("A", z), ("B", ~z)):
                        a[(met, arm, sc, "n")] = np.bincount(s[msk & ok], minlength=M).astype(float)
                        a[(met, arm, sc, "s")] = np.bincount(s[msk & ok], weights=v[msk & ok], minlength=M)
            agg[k] = a
            # 只留本地：簇级损失汇总（宽表，簇编码 = 注册表内的序号，不含原始 ID；只含主要臂）
            W = {f"{met}|{arm}|{sc}|{q}": a[(met, arm, sc, q)] for (met, arm, sc, q) in a if arm in LL_ARMS}
            Wd = pd.DataFrame(W); Wd.insert(0, "簇序号", np.arange(M)); Wd = Wd[Wd.filter(like="|n").sum(axis=1) > 0]
            fp = LOCAL / f"{db}__{k[1]}__{k[2]}.parquet"; Wd.to_parquet(fp, index=False, compression="zstd")
            local_rows.append(dict(库=db, 结局=k[1], 表型=k[2], 簇数=len(Wd), SHA=sha16(fp)))
        rng = np.random.default_rng(ss[db]); done = 0
        while done < R42I.B_BOOT:
            b = min(R42I.CHUNK, R42I.B_BOOT - done)
            draw = rng.integers(0, M, size=(M, b))
            Mt = np.zeros((M, b))
            for j in range(b):
                Mt[:, j] = np.bincount(draw[:, j], minlength=M)
            for k in keys:
                a = agg[k]
                for (met, arm, sc, q) in [x for x in a if x[3] == "n"]:
                    dn = a[(met, arm, sc, "n")] @ Mt; sm = a[(met, arm, sc, "s")] @ Mt
                    reps.setdefault((db, k[1], k[2], met, arm, sc), []).append(np.where(dn > 0, sm / np.where(dn > 0, dn, 1), np.nan))
                    if sc == "B":   # 人口加权
                        dA = a[(met, arm, "A", "n")] @ Mt; sA = a[(met, arm, "A", "s")] @ Mt
                        reps.setdefault((db, k[1], k[2], met, arm, "P"), []).append(
                            np.where(dA + dn > 0, (sA + sm) / np.where(dA + dn > 0, dA + dn, 1), np.nan))
            done += b
            print(f"  {db} 重抽 {done}/{R42I.B_BOOT}", flush=True)
    return {k: np.concatenate(v) for k, v in reps.items()}, pd.DataFrame(local_rows)


def contrast_table(tasks, reps, contrasts, met):
    rows = []
    for (db, oc, ph), (uid, y, z, L, LL) in tasks.items():
        Ls = L if met == "brier" else LL
        Ro = {a: (float(np.nanmean(v[z])), float(np.nanmean(v[~z])), float(np.nanmean(v))) for a, v in Ls.items()}
        for cn, ref, cand in contrasts:
            if ref not in Ls or cand not in Ls:
                continue
            obs = R42I.contrast_vals({a: Ro[a][0] for a in Ro}, {a: Ro[a][1] for a in Ro}, {a: Ro[a][2] for a in Ro}, ref, cand)
            RA = {a: reps[(db, oc, ph, met, a, "A")] for a in (ref, cand)}
            RB = {a: reps[(db, oc, ph, met, a, "B")] for a in (ref, cand)}
            RP = {a: reps[(db, oc, ph, met, a, "P")] for a in (ref, cand)}
            bt = R42I.contrast_vals(RA, RB, RP, ref, cand)
            for sc in ("A", "B", "bal", "pop"):
                dd, rr = obs[sc]; vd, vr = bt[sc]
                Rr = {"A": Ro[ref][0], "B": Ro[ref][1], "bal": .5 * (Ro[ref][0] + Ro[ref][1]), "pop": Ro[ref][2]}[sc]
                row = dict(库=db, 结局=oc, 表型=ph, 指标=met, 对照=cn, 参照=ref, 候选=cand, 尺度=sc,
                           观察绝对差=float(dd), 观察相对改善=float(rr), R参照=float(Rr),
                           相对_P2_5=float(np.nanquantile(vr, .025)), 相对_P97_5=float(np.nanquantile(vr, .975)),
                           绝对_P2_5=float(np.nanquantile(vd, .025)), 绝对_P97_5=float(np.nanquantile(vd, .975)),
                           缺失replicate=int(np.isnan(vr).sum()))
                for m in R42I.MARGINS:
                    row[f"三态_m{m:.3f}"] = ("支持实质收益" if row["相对_P2_5"] > m else
                                           "排除实质收益" if row["相对_P97_5"] < m else "证据不足")
                row["相对0方向"] = "全正" if row["相对_P2_5"] > 0 else "全负" if row["相对_P97_5"] < 0 else "跨零"
                rows.append(row)
    return pd.DataFrame(rows)


def maxT(tasks, reps, met="brier"):
    th = []
    for (db, oc, ph), (uid, y, z, L, LL) in tasks.items():
        Ls = L if met == "brier" else LL
        for g, msk in (("A", z), ("B", ~z)):
            v = (1 - R42I.M_MAIN) * reps[(db, oc, ph, met, "P*", g)] - reps[(db, oc, ph, met, "HTFS", g)]
            obs = (1 - R42I.M_MAIN) * Ls["P*"][msk].mean() - Ls["HTFS"][msk].mean()
            th.append(dict(库=db, 结局=oc, 表型=ph, 轨=g, θ观察=float(obs), θ重抽SD=float(np.nanstd(v)), _v=v))
    TH = pd.DataFrame(th)
    use = TH.θ重抽SD > 0
    Z = np.vstack([(r._v - r.θ观察) / r.θ重抽SD for _, r in TH[use].iterrows()])
    c95 = float(np.nanquantile(np.nanmax(np.abs(Z), axis=0), .95))
    TH["max-T临界值"] = c95; TH["同时下界"] = TH.θ观察 - c95 * TH.θ重抽SD; TH["同时上界"] = TH.θ观察 + c95 * TH.θ重抽SD
    TH["同时判定_相对m"] = np.where(TH.同时下界 > 0, "支持（收益>m）", np.where(TH.同时上界 < 0, "排除（收益<m）", "证据不足"))
    return TH.drop(columns="_v"), c95


def main(smoke=False):
    OUT.mkdir(parents=True, exist_ok=True); LOCAL.mkdir(parents=True, exist_ok=True); t0 = time.time()
    U, LCU, AUX, PR, LCR = [], [], [], [], []
    tasks = {}
    for db in ("mimic", "eicu"):
        for oc in OUTCOMES:
            for ph in PHENOS:
                if smoke and (db, oc, ph) != ("mimic", "OUT_HOSP_MORT", "女性"):
                    continue
                A, rows, lcrows, per_seed, lc_seed = task_pass(db, oc, ph)
                U += rows; LCU += lcrows
                # 与 R42 相同的逐单位 seed 平均损失（主分析单位集合 = 任务全体；断言而非求交集）
                idx = per_seed[SEEDS[0]].index
                for s in SEEDS[1:]:
                    assert per_seed[s].index.equals(idx), "各 seed 的单位集合不同"
                y = per_seed[SEEDS[0]].y.to_numpy(float); z = per_seed[SEEDS[0]].z.to_numpy(bool)
                arms = [c for c in per_seed[SEEDS[0]].columns if c not in ("y", "z")]
                L, LL = {}, {}
                for a in arms:
                    if all(a in d.columns for d in per_seed.values()):
                        L[a] = np.mean([(d[a].to_numpy(float) - y) ** 2 for d in per_seed.values()], axis=0)
                        if a in LL_ARMS:
                            LL[a] = np.mean([logloss_vec(y, d[a].to_numpy(float)) for d in per_seed.values()], axis=0)
                cert_common = idx[np.isfinite(L["CERT"])] if "CERT" in L else idx[:0]
                if "CERT" in L:
                    valid = np.isfinite(L["CERT"])
                    L["P*@C"] = np.where(valid, L["P*"], np.nan); LL["P*@C"] = np.where(valid, LL["P*"], np.nan)
                tasks[(db, oc, ph)] = (idx.to_numpy(np.int64), y, z, L, LL)
                a_, p_ = aux_rows(db, oc, ph, per_seed, cert_common); AUX += a_; PR += p_
                # 学习曲线：各模型自身的绝对 Brier（seed 平均损失；不完整点只在可评价对象上）
                for fr in FRACS + (1.0,):
                    if fr == 1.0:
                        LLc = {a: L[a] for a in ("P*", "HTFS", "Hfull")}; yy, zz = y, z; comp = 1.0; nseed = 3
                    else:
                        ps = lc_seed[fr]
                        if not ps:
                            LCR.append(dict(库=db, 结局=oc, 表型=ph, 目标比例=fr, 完整度=0.0, seed数=0)); continue
                        common = None
                        for d in ps.values():
                            common = d.index if common is None else common.intersection(d.index)
                        common = common.sort_values()
                        yy = ps[next(iter(ps))].loc[common, "y"].to_numpy(float); zz = ps[next(iter(ps))].loc[common, "z"].to_numpy(bool)
                        LLc = {a: np.mean([(d.loc[common, a].to_numpy(float) - yy) ** 2 for d in ps.values()], axis=0)
                               for a in ("P*", "HTFS", "Hfull")}
                        comp = len(common) / len(y); nseed = len(ps)
                    row = dict(库=db, 结局=oc, 表型=ph, 目标比例=fr, 完整度=round(comp, 4), seed数=nseed, 评价人数=len(yy))
                    for a in ("P*", "HTFS", "Hfull"):
                        rA, rB = LLc[a][zz].mean(), LLc[a][~zz].mean()
                        row[f"{a}_A"] = rA; row[f"{a}_B"] = rB; row[f"{a}_两轨等权"] = .5 * (rA + rB)
                    LCR.append(row)
                print(f"{db} {oc} {ph} 完成 {time.time()-t0:.0f}s", flush=True)
    UC = pd.DataFrame(U); LCU = pd.DataFrame(LCU)
    if smoke:
        bad = {c: int((UC[c] == False).sum()) for c in UC.columns if UC[c].dtype == bool or set(UC[c].dropna().unique()) <= {True, False}}  # noqa: E712
        print("闭环核对（False 计数）:", bad); print(pd.DataFrame(AUX).head(12).to_string()); print(pd.DataFrame(LCR).to_string())
        return
    reps, LOC = joint_bootstrap(tasks)
    # R42 复现核对
    T_b = contrast_table(tasks, reps, R42I.CONTRASTS + NEW_CONTRASTS, "brier")
    T_l = contrast_table(tasks, reps, LL_CONTRASTS, "logloss")
    r42 = pd.read_excel(R42I.OUT / "R42_INFERENCE.xlsx", sheet_name="2_全部对照")
    j = T_b.merge(r42, on=["库", "结局", "表型", "对照", "尺度"], suffixes=("", "_R42"))
    rep = dict(R42对照行数=len(r42), 匹配行数=len(j),
               观察相对改善最大差=float((j.观察相对改善 - j.观察相对改善_R42).abs().max()),
               相对下界最大差=float((j.相对_P2_5 - j["相对_P2.5"]).abs().max()),
               相对上界最大差=float((j.相对_P97_5 - j["相对_P97.5"]).abs().max()),
               绝对下界最大差=float((j.绝对_P2_5 - j["绝对_P2.5"]).abs().max()),
               三态不一致=int(sum((j[f"三态_m{m:.3f}"] != j[f"三态_m{m:.3f}_R42"]).sum() for m in R42I.MARGINS)),
               方向不一致=int((j.相对0方向 != j.相对0方向_R42).sum()))
    TH, c95 = maxT(tasks, reps, "brier")
    r42th = pd.read_excel(R42I.OUT / "R42_INFERENCE.xlsx", sheet_name="3_E1_maxT")
    rep.update(maxT临界值=c95, maxT临界值_R42=float(r42th["max-T临界值"].iloc[0]),
               maxT判定不一致=int((TH.sort_values(["库", "结局", "表型", "轨"]).同时判定_相对m.to_numpy()
                                != r42th.sort_values(["库", "结局", "表型", "轨"]).同时判定_相对m.to_numpy()).sum()))
    THl, c95l = maxT(tasks, reps, "logloss")
    # 认证分解（同一评价对象、绝对尺度）
    dec = []
    for (db, oc, ph), (uid, y, z, L, LL) in tasks.items():
        if "CERT" not in L:
            continue
        ok = np.isfinite(L["CERT"])
        for sc, msk in (("A", z & ok), ("B", ~z & ok), ("pop", ok)):
            Rd, Rf, Rc = (float(L[a][msk].mean()) for a in ("P*@C", "P*_F", "CERT"))
            dec.append(dict(库=db, 结局=oc, 表型=ph, 尺度=sc, 评价人数=int(msk.sum()), 纳入比例=round(float(ok.mean()), 4),
                            R_PstarD=Rd, R_PstarF=Rf, R_CERT=Rc, 样本成本项=Rd - Rf, 认证启用项=Rf - Rc, 合计=Rd - Rc))
        a_, b_ = dec[-3], dec[-2]
        dec.append(dict(库=db, 结局=oc, 表型=ph, 尺度="bal", 评价人数=a_["评价人数"] + b_["评价人数"], 纳入比例=a_["纳入比例"],
                        **{k: .5 * (a_[k] + b_[k]) for k in ("R_PstarD", "R_PstarF", "R_CERT", "样本成本项", "认证启用项", "合计")}))
    DEC = pd.DataFrame(dec)
    DEC["恒等式残差"] = DEC.合计 - DEC.样本成本项 - DEC.认证启用项
    for k in ("样本成本项", "认证启用项", "合计"):
        DEC[f"{k}_相对R_PstarD"] = DEC[k] / DEC.R_PstarD
    # 重抽向量导出（汇总量）与本地簇级表
    RP = pd.DataFrame({"|".join(map(str, k)): v for k, v in sorted(reps.items())})
    RP.insert(0, "replicate", np.arange(len(RP)))
    RP.to_parquet(OUT / "R43_BOOT_RISKS.parquet", index=False, compression="zstd")
    AUXD = pd.DataFrame(AUX); PRD = pd.DataFrame(PR); LCRD = pd.DataFrame(LCR)
    chk_cols = [x for x in UC.columns if x not in ("库", "结局", "表型", "seed", "外层折", "分支", "单元", "代码指纹",
                                                   "预测最小", "预测最大", "产物数", "目标比例")]
    UCs = (UC.groupby("分支")[chk_cols].agg(lambda s: f"{int((s == True).sum())}/{int(s.notna().sum())}").reset_index())  # noqa: E712
    summ = dict(任务=len(tasks), 单元记录=len(UC), 代码指纹=sorted(UC.代码指纹.dropna().unique().tolist()),
                预测最小=float(UC.预测最小.min()), 预测最大=float(UC.预测最大.max()),
                **{f"复现_{k}": v for k, v in rep.items()}, logloss_maxT临界值=c95l,
                重抽向量SHA=sha16(OUT / "R43_BOOT_RISKS.parquet"),
                簇级损失本地表SHA=h16_of_list(LOC.SHA.tolist()),
                重抽向量列数=int(RP.shape[1] - 1), 秒=round(time.time() - t0))
    with pd.ExcelWriter(OUT / "R43_READONLY.xlsx", engine="openpyxl") as xw:
        pd.DataFrame([summ]).astype(str).T.reset_index().rename(columns={"index": "项", 0: "值"}).to_excel(xw, sheet_name="0_汇总", index=False)
        UCs.to_excel(xw, sheet_name="1_闭环核对汇总", index=False)
        UC.to_excel(xw, sheet_name="2_闭环核对逐单元", index=False)
        AUXD.to_excel(xw, sheet_name="3_辅助指标逐seed", index=False)
        PRD.to_excel(xw, sheet_name="4_配对差逐seed", index=False)
        T_b.to_excel(xw, sheet_name="5_Brier对照含新增", index=False)
        T_l.to_excel(xw, sheet_name="6_logloss对照", index=False)
        TH.to_excel(xw, sheet_name="7_maxT_Brier复现", index=False)
        THl.to_excel(xw, sheet_name="8_maxT_logloss", index=False)
        DEC.to_excel(xw, sheet_name="9_认证分解", index=False)
        LCU.to_excel(xw, sheet_name="10_学习曲线实际规模", index=False)
        LCRD.to_excel(xw, sheet_name="11_学习曲线绝对风险", index=False)
        LOC.to_excel(xw, sheet_name="12_本地簇级表指纹", index=False)
    (OUT / "R43_READONLY.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, default=float, indent=1))


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
