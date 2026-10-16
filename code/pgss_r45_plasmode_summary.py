# -*- coding: utf-8 -*-
r"""R45 plasmode 汇总（只读 plasmode/runs）。比例同时给分子、分母与 Wilson/Clopper–Pearson 区间；均值给重复间标准误。
δ 之间在同一重复内配对（同一划分与 u），配对差另列。0/20 不写成「错误率 < 5% 已验证」。

表：A 各臂逐轨风险；B 候选对真值收益（D|P*、D|S0*、F|P*、F|S0*；达 m 计数）；C 选择器逐轨（启用/错误启用（拆实际变差）/漏启用/部署相对 P*_D）；
    D 选择器两轨等权部署改善；E 重复级「至少一条不达标轨被启用」；F MIX 与 POOLMIX 对照；G 认证样本成本（P*_F vs P*_D）；
    H δ 配对差（HTFS_D vs P*_D 真值收益）；I 开发选择与病例构成；J 逐重复记录。
输出：27_R45_外部复现与机制核验/plasmode/R45_PLASMODE_SUMMARY.xlsx、.json
"""
import sys, json, pathlib
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
import pgss_r45_plasmode as PM
from pgss_r45_r43fix import prop
M = 0.01


def se(v):
    v = pd.Series(v).dropna()
    return float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan


def main():
    A, S, R = [], [], []
    fails = []
    for d in sorted(PM.RUNS.iterdir()):
        if (d / "_FAILED").exists():
            fails.append(dict(tag=d.name, 信息=(d / "_FAILED").read_text(encoding="utf-8")[:300]))
        if not (d / "_DONE").exists():
            continue
        m = json.loads((d / "rep.json").read_text(encoding="utf-8"))
        a = pd.read_parquet(d / "arms.parquet").assign(δ=m["delta"], rep=m["rep"])
        s = pd.read_parquet(d / "sel.parquet").assign(δ=m["delta"], rep=m["rep"])
        A.append(a); S.append(s)
        R.append({k: v for k, v in m.items() if k not in ("cert", "truth", "生成器")} |
                 {f"G_{k}": v["G"] for k, v in m["truth"].items()} |
                 {f"cert_{ref}_{g}_p": m["cert"][ref][g]["p"] for ref in ("P*", "S0*") for g in ("A", "B")} |
                 {f"cert_{ref}_{g}_启用": m["cert"][ref][g]["specialize"] for ref in ("P*", "S0*") for g in ("A", "B")})
    A = pd.concat(A, ignore_index=True); S = pd.concat(S, ignore_index=True); R = pd.DataFrame(R)
    R["δ"] = R["delta"]  # 冻结后修订 1（R45_PLASMODE_AMENDMENT.json）：rep.json 的键是 delta，汇总按 δ 分组；只补列名，不改计算
    A["R_bal"] = .5 * (A.R_A + A.R_B)
    TA = A.groupby(["δ", "臂"]).agg(重复=("rep", "size"), R_A=("R_A", "mean"), R_B=("R_B", "mean"), R_bal=("R_bal", "mean"),
                                   R_pop=("R_pop", "mean"), R_bal_SE=("R_bal", se)).reset_index()
    # B 候选对
    rows = []
    for pk in ("D|P*", "D|S0*", "F|P*", "F|S0*"):
        for g in ("A", "B"):
            for dl, h in R.groupby("δ"):
                G = h[f"G_{pk}|{g}"]
                r = dict(δ=dl, 候选对=pk, 轨=g, 重复=len(G), 真值相对收益均值=G.mean(), SE=se(G), 最小=G.min(), 最大=G.max(),
                         收益为负=int((G < 0).sum()))
                r.update(prop(int((G >= M).sum()), len(G), "达m_"))
                rows.append(r)
    TB = pd.DataFrame(rows)
    # C 选择器逐轨
    rows = []
    for (dl, nm, g), h in S.groupby(["δ", "选择器", "轨"]):
        r = dict(δ=dl, 选择器=nm, 轨=g, 重复=len(h), 部署相对PstarD改善均值=h.相对PstarD改善.mean(), SE=se(h.相对PstarD改善),
                 带符号差相对均值=h.带符号差_相对两纯策略较优者_相对.mean(), 判断所依候选对=h.判断所依候选对.iloc[0])
        if h.专业化.notna().all():
            sp = h.专业化.astype(bool); G = h.候选对真值相对收益; neg = G < M
            r.update(启用=int(sp.sum()), 不达标重复=int(neg.sum()), 错误启用=int((sp & neg).sum()), 错误启用_实际变差=int((sp & (G < 0)).sum()),
                     达标重复=int((~neg).sum()), 漏启用=int((~sp & ~neg).sum()))
        else:
            r.update(t分布=json.dumps(h.t.value_counts().sort_index().to_dict()))
        rows.append(r)
    TC = pd.DataFrame(rows)
    # D 两轨等权部署
    S2 = S.pivot_table(index=["δ", "rep", "选择器"], columns="轨", values=["R", "R_PstarD"]).reset_index()
    S2.columns = ["_".join([c for c in col if c]) for col in S2.columns]
    S2["bal改善"] = ((S2.R_PstarD_A + S2.R_PstarD_B) - (S2.R_A + S2.R_B)) / (S2.R_PstarD_A + S2.R_PstarD_B)
    TD = S2.groupby(["δ", "选择器"]).bal改善.agg(重复="size", 均值="mean", SE=se, 最小="min", 最大="max").reset_index()
    # E 重复级至少一轨错误启用
    rows = []
    Sb = S[S.专业化.notna()].copy()
    Sb["错启"] = Sb.专业化.astype(bool) & (Sb.候选对真值相对收益 < M)
    Sb["变差启"] = Sb.专业化.astype(bool) & (Sb.候选对真值相对收益 < 0)
    for (dl, nm), h in Sb.groupby(["δ", "选择器"]):
        per = h.groupby("rep").agg(错=("错启", "any"), 变差=("变差启", "any"))
        r = dict(δ=dl, 选择器=nm, 独立重复=len(per))
        r.update(prop(int(per.错.sum()), len(per), "≥1错误启用_")); r.update(prop(int(per.变差.sum()), len(per), "≥1实际变差启用_"))
        rows.append(r)
    TE = pd.DataFrame(rows)
    # F MIX vs POOLMIX（两轨等权部署改善，同一重复配对）
    w = TD[TD.选择器.isin(["MIX", "POOLMIX"])].copy()
    pr = S2[S2.选择器.isin(["MIX", "POOLMIX"])].pivot_table(index=["δ", "rep"], columns="选择器", values="bal改善").reset_index()
    pr["MIX减POOLMIX"] = pr.MIX - pr.POOLMIX
    TF = pr.groupby("δ").agg(重复=("rep", "size"), MIX均值=("MIX", "mean"), POOLMIX均值=("POOLMIX", "mean"),
                             配对差均值=("MIX减POOLMIX", "mean"), 配对差SE=("MIX减POOLMIX", se),
                             MIX更好重复=("MIX减POOLMIX", lambda v: int((v > 0).sum()))).reset_index()
    # G 认证样本成本
    piv = A.pivot_table(index=["δ", "rep"], columns="臂", values=["R_A", "R_B"]).reset_index()
    rows = []
    for dl, h in piv.groupby("δ"):
        for g in ("A", "B"):
            c = (h[(f"R_{g}", "P*_D")] - h[(f"R_{g}", "P*_F")]) / h[(f"R_{g}", "P*_D")]
            rows.append(dict(δ=dl, 轨=g, 重复=len(c), P星F相对P星D改善均值=c.mean(), SE=se(c), 最小=c.min(), 最大=c.max()))
    TG = pd.DataFrame(rows)
    # H δ 配对差：HTFS_D vs P*_D 真值收益（同一重复）
    rows = []
    for g in ("A", "B"):
        pv = R.pivot_table(index="rep", columns="δ", values=f"G_D|P*|{g}")
        for d1 in (0.5, 1.0):
            dd = pv[d1] - pv[0.0]
            rows.append(dict(轨=g, 比较=f"δ={d1} − δ=0", 重复=int(dd.notna().sum()), 收益差均值=dd.mean(), SE=se(dd),
                             为正重复=int((dd > 0).sum())))
    TH = pd.DataFrame(rows)
    TI = R.groupby("δ").agg(重复=("rep", "size"), 截距a=("截距a", "mean"), 开发记录均值=("n_dev", "mean"), 开发患者均值=("n_dev_patients", "mean"),
                            评价记录均值=("n_eval", "mean"), A轨开发人数均值=("n_A_dev", "mean"), A轨开发事件均值=("events_A_dev", "mean"),
                            B轨开发事件均值=("events_B_dev", "mean"), 评价A轨平均p=("eval_pA_mean", "mean"), 评价B轨平均p=("eval_pB_mean", "mean"),
                            每重复秒均值=("sec_total", "mean")).reset_index()
    TI["P*分布"] = [json.dumps(R[R.δ == d].pstar_D.value_counts().to_dict(), ensure_ascii=False) for d in TI.δ]
    TI["ALT分布"] = [json.dumps(R[R.δ == d].ALT.value_counts().to_dict(), ensure_ascii=False) for d in TI.δ]
    cpl = R.groupby("rep").agg(划分指纹数=("dev_split_sha", "nunique"), u指纹数=("u_sha", "nunique"))
    summ = dict(完成=len(R), 失败=len(fails), 计划=len(PM.job_list()), 耦合核对_划分与u在三档δ间相同=bool((cpl.划分指纹数.eq(1) & cpl.u指纹数.eq(1)).all()),
                代码指纹=sorted(R.code_sha.unique().tolist()), 累计秒=float(R.sec_total.sum()))
    with pd.ExcelWriter(PM.PL / "R45_PLASMODE_SUMMARY.xlsx", engine="openpyxl") as xw:
        pd.DataFrame([summ]).astype(str).T.reset_index().to_excel(xw, sheet_name="0_汇总", index=False)
        for nm, t in (("A_各臂风险", TA), ("B_候选对真值收益", TB), ("C_选择器逐轨", TC), ("D_选择器两轨等权", TD),
                      ("E_重复级至少一轨错误启用", TE), ("F_MIX与POOLMIX", TF), ("G_认证样本成本", TG), ("H_δ配对差", TH),
                      ("I_开发选择与构成", TI), ("J_逐重复记录", R), ("K_失败", pd.DataFrame(fails))):
            t.to_excel(xw, sheet_name=nm, index=False)
    (PM.PL / "R45_PLASMODE_SUMMARY.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, default=float))
    return summ


if __name__ == "__main__":
    main()
