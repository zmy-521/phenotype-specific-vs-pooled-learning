# -*- coding: utf-8 -*-
r"""R43 T1 汇总（只读 t1/runs/）。所有比例同时给分子、分母与 Wilson 95% 区间；均值给重复间标准误（Monte Carlo 不确定性）。
20 次重复中 0 次错误**不写成**错误率已控制在 5% 以下：只报告 0/20 及其 Wilson 上界。

用法：python pgss_r43_t1_summary.py [main|ext]
输出：main → t1/R43_T1_SUMMARY.xlsx/json（原 320）；ext → t1/R43_T1_EXT_SUMMARY.xlsx/json（扩展 160，另含各轨病例构成表 I）。
"""
import sys, json, pathlib
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from scipy.stats import norm
import pgss_r43_t1 as T1

M = T1.M
COND = ["机制", "n", "πA"]


def wilson(k, n, a=0.05):
    if n == 0:
        return np.nan, np.nan
    z = norm.ppf(1 - a / 2); p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def prop(k, n):
    lo, hi = wilson(k, n)
    return dict(分子=int(k), 分母=int(n), 比例=(k / n if n else np.nan), Wilson下=lo, Wilson上=hi)


def load(runs):
    A, S, Mt, F = [], [], [], []
    for d in sorted(runs.iterdir()):
        if (d / "_FAILED").exists():
            F.append(dict(tag=d.name, 信息=(d / "_FAILED").read_text(encoding="utf-8")[:300]))
        if not (d / "_DONE").exists():
            continue
        m = json.loads((d / "rep.json").read_text(encoding="utf-8"))
        key = dict(机制=m["scen"], n=m["n"], πA=m["pi"], rep=m["rep"])
        a = pd.read_parquet(d / "arms.parquet"); s = pd.read_parquet(d / "sel.parquet")
        for k, v in key.items():
            a[k] = v; s[k] = v
        A.append(a); S.append(s)
        Mt.append(dict(key, **{k: m[k] for k in ("pstar_D", "pstar_cal", "S0star_D", "HTFS_kA", "HTFS_kB", "SZGFS_k_none",
                                                  "SZGFS_k_bal", "pstar_F", "HTFS_F_kA", "HTFS_F_kB", "n_A_dev", "events_A_dev",
                                                  "events_B_dev", "n_F", "n_C", "sec_total", "sec_main", "sec_cert", "n_fits",
                                                  "code_sha")}, 追加评价=len(m["追加评价"])))
    return pd.concat(A, ignore_index=True), pd.concat(S, ignore_index=True), pd.DataFrame(Mt), pd.DataFrame(F)


def main(mode="main"):
    runs = T1.RUNS if mode == "main" else T1.T1DIR / "runs_ext"
    A, S, Mt, F = load(runs)
    A["R_bal"] = .5 * (A.R_A + A.R_B)
    # T1-A：各臂真值风险（条件 × 臂）
    TA = (A.groupby(COND + ["臂"]).agg(重复=("rep", "size"), R_A=("R_A", "mean"), R_B=("R_B", "mean"),
                                        R_bal=("R_bal", "mean"), R_bal_SE=("R_bal", lambda v: v.std(ddof=1) / np.sqrt(len(v))))
          .reset_index())
    # T1-B：候选对真值相对收益（D 对与 F 对；逐轨）
    pb = []
    piv = A.pivot_table(index=COND + ["rep"], columns="臂", values=["R_A", "R_B"])
    for (sc, n, pi), h in piv.groupby(level=[0, 1, 2]):
        for g in ("A", "B"):
            for pair, (rf, sp) in {"D: HTFS_D vs P*_D": ("P*_D", "HTFS_D"), "D: HTFS_D vs S0*_D": ("S0*_D", "HTFS_D"),
                                   "F: HTFS_F vs P*_F": ("P*_F", "HTFS_F")}.items():
                G = (h[(f"R_{g}", rf)] - h[(f"R_{g}", sp)]) / h[(f"R_{g}", rf)]
                pb.append(dict(机制=sc, n=n, πA=pi, 轨=g, 候选对=pair, 重复=len(G), 真值相对收益均值=G.mean(),
                               SE=G.std(ddof=1) / np.sqrt(len(G)), 最小=G.min(), 最大=G.max(),
                               **{f"达m_{k}": v for k, v in prop(int((G >= M).sum()), len(G)).items()}))
    TB = pd.DataFrame(pb)
    # T1-C：选择器（条件 × 选择器 × 轨）
    sc_rows = []
    for (sc, n, pi, nm, g), h in S.groupby(COND + ["选择器", "轨"]):
        row = dict(机制=sc, n=n, πA=pi, 选择器=nm, 轨=g, 重复=len(h),
                   相对PstarD改善均值=h.相对PstarD改善.mean(), 相对PstarD改善SE=h.相对PstarD改善.std(ddof=1) / np.sqrt(len(h)),
                   regret相对均值=h.regret_相对.mean(), regret相对SE=h.regret_相对.std(ddof=1) / np.sqrt(len(h)),
                   判断所依候选对=h.判断所依候选对.iloc[0])
        if h.专业化.notna().all():
            sp = h.专业化.astype(bool)
            row.update({f"启用_{k}": v for k, v in prop(int(sp.sum()), len(h)).items()})
            neg = h[~h.候选对达m.astype(bool)]; pos = h[h.候选对达m.astype(bool)]
            row.update({f"错误启用_{k}": v for k, v in prop(int(neg.专业化.astype(bool).sum()), len(neg)).items()})
            row.update({f"漏启用_{k}": v for k, v in prop(int((~pos.专业化.astype(bool)).sum()), len(pos)).items()})
        else:
            row.update(混合t分布=json.dumps(h.t.value_counts().sort_index().to_dict()))
        sc_rows.append(row)
    TC = pd.DataFrame(sc_rows)
    # T1-D：程序层面（两轨等权）相对 P*_D 的改善
    S2 = S.pivot_table(index=COND + ["rep", "选择器"], columns="轨", values=["R", "R_PstarD"]).reset_index()
    S2.columns = ["_".join([c for c in col if c]) for col in S2.columns]
    S2["bal改善"] = ((S2.R_PstarD_A + S2.R_PstarD_B) - (S2.R_A + S2.R_B)) / (S2.R_PstarD_A + S2.R_PstarD_B)
    TD = (S2.groupby(COND + ["选择器"]).bal改善.agg(重复="size", 均值="mean", SE=lambda v: v.std(ddof=1) / np.sqrt(len(v)),
                                                  最小="min", 最大="max").reset_index())
    # T1-E：开发选择与面板；T1-F：成本
    TE = []
    for c, h in Mt.groupby(COND):
        TE.append(dict(zip(COND, c), 重复=len(h),
                       Pstar分布=json.dumps(h.pstar_D.value_counts().to_dict(), ensure_ascii=False),
                       Pstar校准比例=h.pstar_cal.mean(),
                       HTFS_kA分布=json.dumps(h.HTFS_kA.value_counts().sort_index().to_dict()),
                       HTFS_kB分布=json.dumps(h.HTFS_kB.value_counts().sort_index().to_dict()),
                       SZGFS_k_none分布=json.dumps(h.SZGFS_k_none.value_counts().sort_index().to_dict()),
                       A轨开发人数均值=h.n_A_dev.mean(), A轨事件均值=h.events_A_dev.mean(), B轨事件均值=h.events_B_dev.mean()))
    TE = pd.DataFrame(TE)
    TF = (Mt.groupby(COND).agg(重复=("rep", "size"), 每重复秒均值=("sec_total", "mean"), 主分支秒均值=("sec_main", "mean"),
                                认证分支秒均值=("sec_cert", "mean"), 拟合次数均值=("n_fits", "mean"), 追加评价重复=("追加评价", lambda v: int((v > 0).sum())))
          .reset_index())
    TI = pd.DataFrame()
    if mode == "ext":                                  # 扩展：各轨病例构成（开发集与评价集），按条件取重复均值
        ex = [json.loads((d / "ext.json").read_text(encoding="utf-8")) for d in sorted(runs.iterdir()) if (d / "ext.json").exists()]
        E = pd.DataFrame(ex).rename(columns={"scen": "机制", "pi": "πA"})
        num = [c for c in E.columns if c not in ("机制", "n", "πA", "rep", "ext_sha") and E[c].dtype != bool]
        TI = E.groupby(COND)[num].mean().reset_index()
        for c in [c for c in E.columns if c.endswith("Z规则逐例一致")]:
            TI[c] = E.groupby(COND)[c].all().to_numpy()
    import pgss_r43_t1_ext as TX
    plan = len(T1.job_list("main")) if mode == "main" else len(TX.job_list("all"))
    summ = dict(批次=("原 T1（320）" if mode == "main" else "T1 扩展（160，探索性）"), 完成重复=len(Mt), 失败=len(F), 计划_本批=plan,
                代码指纹=sorted(Mt.code_sha.unique().tolist()), 累计秒=float(Mt.sec_total.sum()))
    stem = "R43_T1_SUMMARY" if mode == "main" else "R43_T1_EXT_SUMMARY"
    out = T1.T1DIR / f"{stem}.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        pd.DataFrame([summ]).astype(str).T.reset_index().to_excel(xw, sheet_name="0_汇总", index=False)
        TA.to_excel(xw, sheet_name="A_各臂真值风险", index=False)
        TB.to_excel(xw, sheet_name="B_候选对真值收益", index=False)
        TC.to_excel(xw, sheet_name="C_选择器逐轨", index=False)
        TD.to_excel(xw, sheet_name="D_选择器两轨等权", index=False)
        TE.to_excel(xw, sheet_name="E_开发选择与面板", index=False)
        TF.to_excel(xw, sheet_name="F_成本", index=False)
        F.to_excel(xw, sheet_name="G_失败", index=False)
        Mt.to_excel(xw, sheet_name="H_逐重复记录", index=False)
        if len(TI):
            TI.to_excel(xw, sheet_name="I_各轨病例构成", index=False)
    (T1.T1DIR / f"{stem}.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, default=float))
    return dict(TA=TA, TB=TB, TC=TC, TD=TD, TE=TE, TF=TF, F=F, Mt=Mt, TI=TI, summ=summ)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "main")
