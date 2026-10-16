# -*- coding: utf-8 -*-
r"""R46 T2 汇总（只读 t2/runs）。方案 R46_T2_PROTOCOL_v1.0.md §6、§8。
比例：分子、分母、比例、Clopper–Pearson 95% 区间、Monte Carlo 标准误；均值：重复间标准误。「边界不确定」单列，不并入任何一类。
表：0 汇总；A 生成与评价规模；B 各臂真值风险；C 候选对真值收益；D 选择器逐轨；E 选择器两轨等权；F 重复级至少一轨不达标启用；
    G MIX 与 POOLMIX；H 认证样本成本；I 与 T1 并列（Q1–Q5）；J 两两排序（Q6）；K 成本；L 失败；M 逐重复记录。
用法：python pgss_r46_t2_summary.py [runs目录 输出目录]（默认 t2/runs → t2/）
"""
import sys, json, pathlib, itertools
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from pgss_r45_r43fix import prop
import pgss_r46_t2 as T2

M = T2.M
T1DIR = HERE.parent / "24_R43_结果补齐与T1" / "t1"


def se(v):
    v = pd.Series(v, dtype=float).dropna()
    return float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan


def pr(k, n, pre):
    d = prop(int(k), int(n), pre)
    p = d[f"{pre}比例"]
    d[f"{pre}MC标准误"] = float(np.sqrt(p * (1 - p) / n)) if n else np.nan
    return d


def load(runs):
    A, P, S, R, F = [], [], [], [], []
    for d in sorted(pathlib.Path(runs).iterdir()):
        if not d.is_dir():
            continue
        if (d / "_FAILED").exists():
            F.append(dict(tag=d.name, 信息=(d / "_FAILED").read_text(encoding="utf-8")[:300]))
        if not (d / "_DONE").exists():
            continue
        m = json.loads((d / "rep.json").read_text(encoding="utf-8"))
        k = dict(条件=m["cond"], rep=m["rep"])
        A.append(pd.read_parquet(d / "arms.parquet").assign(**k)); P.append(pd.read_parquet(d / "pairs.parquet").assign(**k))
        S.append(pd.read_parquet(d / "sel.parquet").assign(**k))
        R.append({kk: v for kk, v in m.items() if kk not in ("cert", "极端值", "追加记录", "最终仍近边界")}
                 | {f"极端_{kk}": v for kk, v in m["极端值"].items()}
                 | {"最终仍近边界项数": len(m["最终仍近边界"]), "条件": m["cond"]})
    cat = lambda L: pd.concat(L, ignore_index=True) if L else pd.DataFrame()
    return cat(A), cat(P), cat(S), pd.DataFrame(R), pd.DataFrame(F)


def order(df):
    return df.assign(_o=df.条件.map({c: i for i, c in enumerate(T2.CONDS)})).sort_values("_o").drop(columns="_o")


def main(runs=None, out=None):
    runs = pathlib.Path(runs) if runs else T2.RUNS
    out = pathlib.Path(out) if out else T2.T2DIR
    A, P, S, R, F = load(runs)
    A["R_bal"] = .5 * (A.R_A + A.R_B)
    # A 生成与评价规模
    rows = []
    for c, h in R.groupby("条件"):
        rows.append(dict(条件=c, 重复=len(h), 开发事件率均值=h.极端_dev_event_rate.mean(), 开发平均p均值=h.极端_dev_mean_p.mean(),
                         评价事件率均值=h.eval_event_rate.mean(), 评价A轨平均p=h.eval_pA.mean(), 评价B轨平均p=h.eval_pB.mean(),
                         A轨Bayes风险=h.eval_bayes_A.mean(), B轨Bayes风险=h.eval_bayes_B.mean(),
                         A轨开发人数均值=h.n_A_dev.mean(), A轨开发事件均值=h.events_A_dev.mean(), B轨开发事件均值=h.events_B_dev.mean(),
                         评价X绝对值最大_中位=h.极端_X_absmax.median(), 评价X绝对值最大_最大=h.极端_X_absmax.max(),
                         评价X绝对值P99_9_均值=h.极端_X_abs_q999.mean(), 评价X绝对值超4比例_均值=h.极端_X_frac_abs_gt4.mean(),
                         线性预测子P0_1_均值=h.极端_lin_q001.mean(), 线性预测子P99_9_均值=h.极端_lin_q999.mean(),
                         线性预测子最小=h.极端_lin_min.min(), 线性预测子最大=h.极端_lin_max.max(),
                         p小于1e6比例_均值=h.极端_p_frac_lt1e6.mean(), p大于1减1e6比例_均值=h.极端_p_frac_gt1m1e6.mean(),
                         评价10万=int((h.n_eval == 100_000).sum()), 评价50万=int((h.n_eval == 500_000).sum()),
                         评价100万=int((h.n_eval == 1_000_000).sum()), 最终仍有近边界项的重复=int((h.最终仍近边界项数 > 0).sum())))
    TA = order(pd.DataFrame(rows))
    TB = order(A.groupby(["条件", "臂"]).agg(重复=("rep", "size"), R_A=("R_A", "mean"), R_B=("R_B", "mean"), R_bal=("R_bal", "mean"),
                                            R_pop=("R_pop", "mean"), R_bal_SE=("R_bal", se)).reset_index())
    # C 候选对
    rows = []
    for (c, pk, g), h in P.groupby(["条件", "候选对", "轨"]):
        nm, n0 = h.近m.astype(bool), h.近0.astype(bool)
        r = dict(条件=c, 候选对=pk, 轨=g, 重复=len(h), 真值相对收益均值=h.G.mean(), SE=se(h.G), P5=h.G.quantile(.05), 中位=h.G.median(),
                 P95=h.G.quantile(.95), 积分SE中位=h.se.median(), m边界不确定=int(nm.sum()), 零边界不确定=int(n0.sum()))
        r.update(pr(int(((h.G >= M) & ~nm).sum()), len(h), "达m_确定_"))
        r.update(pr(int(((h.G < 0) & ~n0).sum()), len(h), "为负_确定_"))
        rows.append(r)
    TC = order(pd.DataFrame(rows))
    # D 选择器逐轨
    rows = []
    for (c, nm, g), h in S.groupby(["条件", "选择器", "轨"]):
        r = dict(条件=c, 选择器=nm, 轨=g, 重复=len(h), 部署相对PstarD改善均值=h.相对PstarD改善.mean(), SE=se(h.相对PstarD改善),
                 P5=h.相对PstarD改善.quantile(.05), 中位=h.相对PstarD改善.median(), P95=h.相对PstarD改善.quantile(.95),
                 判断所依候选对=h.判断所依候选对.iloc[0])
        r.update(pr(int(h.部署实际变差.astype(bool).sum()), len(h), "部署实际变差_"))
        r["部署变差边界不确定"] = int(h.部署变差边界不确定.astype(bool).sum())
        if h.专业化.notna().all():
            sp = h.专业化.astype(bool); G = h.候选对真值相对收益; nmb = h.候选对近m.astype(bool); n0b = h.候选对近0.astype(bool)
            r.update(pr(int(sp.sum()), len(h), "启用_"))
            r.update(pr(int(h.不达标启用.astype(bool).sum()), int(((G < M) & ~nmb).sum()), "不达标启用_条件于确定不达标_"))
            r.update(pr(int(h.不达标启用.astype(bool).sum()), len(h), "不达标启用_占全部重复_"))
            r.update(pr(int(h.实际变差启用.astype(bool).sum()), len(h), "实际变差启用_占全部重复_"))
            r.update(pr(int(h.漏启用.astype(bool).sum()), int(((G >= M) & ~nmb).sum()), "漏启用_条件于确定达标_"))
            r.update(pr(int((sp & (G >= M) & ~nmb).sum()), int(((G >= M) & ~nmb).sum()), "检出_条件于确定达标_"))
            r.update(启用_m边界不确定=int(h.启用_m边界不确定.astype(bool).sum()), 未启用_m边界不确定=int(h.未启用_m边界不确定.astype(bool).sum()),
                     零边界不确定的启用=int((sp & n0b).sum()))
        else:
            r.update(t分布=json.dumps(h.t.value_counts().sort_index().to_dict()))
        rows.append(r)
    TD = order(pd.DataFrame(rows))
    # E 两轨等权部署改善
    S2 = S.pivot_table(index=["条件", "rep", "选择器"], columns="轨", values=["R", "R_PstarD"]).reset_index()
    S2.columns = ["_".join([c for c in col if c]) for col in S2.columns]
    S2["bal改善"] = ((S2.R_PstarD_A + S2.R_PstarD_B) - (S2.R_A + S2.R_B)) / (S2.R_PstarD_A + S2.R_PstarD_B)
    TE = order(S2.groupby(["条件", "选择器"]).bal改善.agg(重复="size", 均值="mean", SE=se, 最小="min",
                                                       P5=lambda v: v.quantile(.05), 中位="median", P95=lambda v: v.quantile(.95),
                                                       最大="max", 为负重复=lambda v: int((v < 0).sum())).reset_index())
    # F 重复级至少一轨不达标启用 / 实际变差启用
    rows = []
    Sb = S[S.专业化.notna()]
    for (c, nm), h in Sb.groupby(["条件", "选择器"]):
        per = h.groupby("rep").agg(错=("不达标启用", lambda v: bool(v.astype(bool).any())), 变差=("实际变差启用", lambda v: bool(v.astype(bool).any())),
                                   不确定=("启用_m边界不确定", lambda v: bool(v.astype(bool).any())))
        r = dict(条件=c, 选择器=nm, 独立重复=len(per))
        r.update(pr(int(per.错.sum()), len(per), "≥1轨不达标启用_")); r.update(pr(int(per.变差.sum()), len(per), "≥1轨实际变差启用_"))
        r["≥1轨m边界不确定启用"] = int(per.不确定.sum())
        rows.append(r)
    TF = order(pd.DataFrame(rows))
    # G MIX 与 POOLMIX
    rows = []
    for (c, nm, g), h in S[S.选择器.isin(["MIX", "POOLMIX"])].groupby(["条件", "选择器", "轨"]):
        r = dict(条件=c, 选择器=nm, 轨=g, 重复=len(h), 风险变化均值=h.相对PstarD改善.mean(), SE=se(h.相对PstarD改善),
                 P5=h.相对PstarD改善.quantile(.05), 中位=h.相对PstarD改善.median(), P95=h.相对PstarD改善.quantile(.95),
                 t为0=int((h.t == 0).sum()), t为1=int((h.t == 1).sum()), 变差边界不确定=int(h.部署变差边界不确定.astype(bool).sum()))
        r.update(pr(int(h.部署实际变差.astype(bool).sum()), len(h), "实际变差_"))
        rows.append(r)
    pv = S2[S2.选择器.isin(["MIX", "POOLMIX"])].pivot_table(index=["条件", "rep"], columns="选择器", values="bal改善").reset_index()
    pv["差"] = pv.MIX - pv.POOLMIX
    TG = pd.concat([order(pd.DataFrame(rows)),
                    order(pv.groupby("条件").agg(重复=("rep", "size"), 风险变化均值=("差", "mean"), SE=("差", se),
                                                 MIX更好重复=("差", lambda v: int((v > 0).sum()))).reset_index()
                          .assign(选择器="MIX − POOLMIX（两轨等权，同重复配对）", 轨="bal"))], ignore_index=True)
    # H 认证样本成本
    piv = A.pivot_table(index=["条件", "rep"], columns="臂", values=["R_A", "R_B"]).reset_index()
    rows = []
    for c, h in piv.groupby("条件"):
        for g in ("A", "B"):
            v = (h[(f"R_{g}", "P*_D")] - h[(f"R_{g}", "P*_F")]) / h[(f"R_{g}", "P*_D")]
            rows.append(dict(条件=c, 轨=g, 重复=len(v), P星F相对P星D改善均值=v.mean(), SE=se(v), 最小=v.min(), 最大=v.max()))
    TH = order(pd.DataFrame(rows))
    # I 与 T1 并列（Q1–Q5）；J 两两排序（Q6）
    TI, TJ = t1_compare(TC, TE, S2)
    TK = order(R.groupby("条件").agg(重复=("rep", "size"), 每重复秒均值=("sec_total", "mean"), 每重复秒最大=("sec_total", "max"),
                                    主分支秒均值=("sec_main", "mean"), 认证分支秒均值=("sec_cert", "mean"),
                                    拟合次数均值=("n_fits", "mean"), 峰值内存MB均值=("peak_rss_mb", "mean"),
                                    峰值内存MB最大=("peak_rss_mb", "max")).reset_index())
    summ = dict(完成=len(R), 失败=len(F), 计划=len(T2.job_list("all")), 条件=T2.CONDS,
                代码指纹=sorted(R.code_sha.unique().tolist()) if len(R) else [], 累计秒=float(R.sec_total.sum()) if len(R) else 0.0,
                评价规模分布=R.n_eval.value_counts().sort_index().to_dict() if len(R) else {})
    out.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out / "R46_T2_SUMMARY.xlsx", engine="openpyxl") as xw:
        pd.DataFrame([summ]).astype(str).T.reset_index().to_excel(xw, sheet_name="0_汇总", index=False)
        for nm, t in (("A_生成与评价规模", TA), ("B_各臂真值风险", TB), ("C_候选对真值收益", TC), ("D_选择器逐轨", TD),
                      ("E_选择器两轨等权", TE), ("F_重复级至少一轨", TF), ("G_MIX与POOLMIX", TG), ("H_认证样本成本", TH),
                      ("I_与T1并列", TI), ("J_两两排序", TJ), ("K_成本", TK), ("L_失败", F), ("M_逐重复记录", R)):
            t.to_excel(xw, sheet_name=nm, index=False)
    (out / "R46_T2_SUMMARY.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, default=float))
    return dict(TA=TA, TB=TB, TC=TC, TD=TD, TE=TE, TF=TF, TG=TG, TH=TH, TI=TI, TJ=TJ, TK=TK, F=F, R=R, summ=summ)


def t1_tables():
    """T1（n=6000、πA=0.15）的候选对与两轨等权部署改善；条件名与 T2 相同（T2-B 无对应）。"""
    B, D = [], []
    for f in ("R43_T1_SUMMARY.xlsx", "R43_T1_EXT_SUMMARY.xlsx"):
        b = pd.read_excel(T1DIR / f, sheet_name="B_候选对真值收益"); d = pd.read_excel(T1DIR / f, sheet_name="D_选择器两轨等权")
        B.append(b[(b.n == 6000) & (b.πA == 0.15)]); D.append(d[(d.n == 6000) & (d.πA == 0.15)])
    return pd.concat(B), pd.concat(D)


def t1_compare(TC, TE, S2):
    B1, D1 = t1_tables()
    pmap = {"D: HTFS_D vs P*_D": "D|P*", "D: HTFS_D vs S0*_D": "D|S0*", "F: HTFS_F vs P*_F": "F|P*"}
    rows = []
    for _, b in B1.iterrows():
        t = TC[(TC.条件 == b.机制) & (TC.候选对 == pmap[b.候选对]) & (TC.轨 == b.轨)]
        if t.empty:
            continue
        t = t.iloc[0]
        rows.append(dict(项="候选对真值相对收益", 条件=b.机制, 对象=pmap[b.候选对], 轨=b.轨, T1重复=int(b.重复), T1均值=b.真值相对收益均值,
                         T1_SE=b.SE, T2重复=int(t.重复), T2均值=t.真值相对收益均值, T2_SE=t.SE, T2下=t.真值相对收益均值 - 1.96 * t.SE,
                         T2上=t.真值相对收益均值 + 1.96 * t.SE, 符号相同=bool(np.sign(b.真值相对收益均值) == np.sign(t.真值相对收益均值)),
                         T1达m比例=b["达m_比例"], T2达m确定比例=t["达m_确定_比例"]))
    for _, d in D1.iterrows():
        t = TE[(TE.条件 == d.机制) & (TE.选择器 == d.选择器)]
        if t.empty:
            continue
        t = t.iloc[0]
        rows.append(dict(项="选择器两轨等权部署改善", 条件=d.机制, 对象=d.选择器, 轨="bal", T1重复=int(d.重复), T1均值=d.均值, T1_SE=d.SE,
                         T2重复=int(t.重复), T2均值=t.均值, T2_SE=t.SE, T2下=t.均值 - 1.96 * t.SE, T2上=t.均值 + 1.96 * t.SE,
                         符号相同=bool(np.sign(d.均值) == np.sign(t.均值))))
    TI = pd.DataFrame(rows)
    # Q6：两两排序（T2 同重复配对差；T1 用均值差）
    rows = []
    for c, h in S2.groupby("条件"):
        w = h.pivot_table(index="rep", columns="选择器", values="bal改善")
        d1 = D1[D1.机制 == c].set_index("选择器").均值
        for a, b in itertools.combinations(sorted(w.columns), 2):
            dd = (w[a] - w[b]).dropna()
            m_, s_ = dd.mean(), se(dd)
            t1d = (d1[a] - d1[b]) if (a in d1.index and b in d1.index) else np.nan
            rev = bool(np.isfinite(t1d) and np.sign(t1d) != np.sign(m_) and (m_ - 1.96 * s_ > 0 or m_ + 1.96 * s_ < 0))
            rows.append(dict(条件=c, 选择器甲=a, 选择器乙=b, T2配对差均值=m_, T2配对差SE=s_, T2下=m_ - 1.96 * s_, T2上=m_ + 1.96 * s_,
                             T1均值差=t1d, T2推翻T1排序=rev))
    TJ = pd.DataFrame(rows)
    return TI, (order(TJ) if len(TJ) else TJ)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        main(sys.argv[1], sys.argv[2])
    else:
        main()
