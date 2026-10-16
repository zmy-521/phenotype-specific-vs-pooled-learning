# -*- coding: utf-8 -*-
r"""R45 第一部分：R43 T1 汇总层的只读更正（GPT 第 49 轮 §2、§3.2）。不重跑任何重复，只读 t1/runs 与 t1/runs_ext 的逐重复记录。

更正 1（§2.1）候选对表与选择器统计的「真值」统一取自最终 truth（rep.json；接近边界者已追加 40 万评价样本）。
  原 pgss_r43_t1_summary.py 第 63–71 行从 arms.parquet（初始 10 万评价样本）重算候选对收益与达标数，与选择器统计所用的
  最终 truth 口径不同。本表同时列出两种口径与差异，达标数以最终 truth 为准；追加后仍距边界 < 2 个积分标准误的，单列为「边界附近」。
  各臂风险表（arms.parquet）保留初始评价样本口径，表头注明。
更正 2（§2.2）原字段 regret / regret_相对 = R(策略) − min{R(P*_D), R(HTFS_D)}：只与两个纯策略比较，可以为负，不截断；
  改名为「相对两纯策略较优者的带符号风险差」，并报告负值个数。
更正 3（§3.2）新增「同一重复中至少一条不达标轨被启用」的比例（按条件 × 选择器；分子 = 重复数、分母 = 独立重复数；Wilson 与
  Clopper–Pearson 95% 区间），两条轨道不当作两次独立试验，不同机制不合并。另把错误启用拆成「真值收益 < 0（实际变差）」与
  「0 ≤ 收益 < m（未达边界但未变差）」。
另做一致性断言：sel.parquet 的「候选对真值相对收益」逐行等于 rep.json 的 truth[判断所依候选对|轨].G（否则拒绝出表）。

输出：27_R45_外部复现与机制核验/r43fix/R45_R43FIX.xlsx、R45_R43FIX.json
"""
import sys, json, pathlib
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from scipy.stats import beta
import pgss_r43_t1 as T1
from pgss_r43_t1_summary import wilson

M = T1.M
OUT = HERE.parent / "27_R45_外部复现与机制核验" / "r43fix"
COND = ["批次", "机制", "n", "πA"]
PAIR_ARMS = {"D|P*": ("P*_D", "HTFS_D"), "D|S0*": ("S0*_D", "HTFS_D"), "F|P*": ("P*_F", "HTFS_F"), "F|S0*": ("S0*_F", "HTFS_F")}
BINARY = None   # 有二元决定的选择器（运行时从数据确定：专业化非空）


def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def prop(k, n, pre):
    wl, wh = wilson(k, n); cl, ch = cp(k, n)
    return {f"{pre}分子": int(k), f"{pre}分母": int(n), f"{pre}比例": (k / n if n else np.nan),
            f"{pre}Wilson下": wl, f"{pre}Wilson上": wh, f"{pre}CP下": cl, f"{pre}CP上": ch}


def load(batch, runs):
    T, S = [], []
    for d in sorted(runs.iterdir()):
        if not (d / "_DONE").exists():
            continue
        m = json.loads((d / "rep.json").read_text(encoding="utf-8"))
        key = dict(批次=batch, 机制=m["scen"], n=m["n"], πA=m["pi"], rep=m["rep"])
        a = pd.read_parquet(d / "arms.parquet").set_index("臂")
        for pk, (rf, sp) in PAIR_ARMS.items():
            for g in ("A", "B"):
                tr = m["truth"][f"{pk}|{g}"]
                Ri, Rsi = a.loc[rf, f"R_{g}"], a.loc[sp, f"R_{g}"]
                T.append(dict(key, 候选对=pk, 轨=g, G_最终=tr["G"], se_最终=tr["se"], 追加评价=bool(tr["追加评价"]),
                              G_初始=float((Ri - Rsi) / Ri)))
        s = pd.read_parquet(d / "sel.parquet")
        for k, v in key.items():
            s[k] = v
        S.append(s)
    return pd.DataFrame(T), pd.concat(S, ignore_index=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    T0, S0 = load("原T1", T1.RUNS)
    T1e, S1e = load("扩展", T1.T1DIR / "runs_ext")
    T = pd.concat([T0, T1e], ignore_index=True); S = pd.concat([S0, S1e], ignore_index=True)
    # 一致性断言：选择器表所用真值 = rep.json 最终 truth
    tk = T.set_index(COND + ["rep", "候选对", "轨"]).G_最终
    look = S.set_index(COND + ["rep", "判断所依候选对", "轨"]).index
    gt = tk.reindex(look).to_numpy()
    mism = int(np.sum(gt != S.候选对真值相对收益.to_numpy()))
    if mism:
        raise SystemExit(f"选择器表真值与 rep.json truth 不一致 {mism} 行")
    T["达m_最终"] = T.G_最终 >= M; T["达m_初始"] = T.G_初始 >= M
    T["边界附近_最终"] = (T.G_最终 - M).abs() < 2 * T.se_最终
    # 表 1：候选对（最终 truth 为准）
    rows = []
    for c, h in T.groupby(COND + ["轨", "候选对"]):
        r = dict(zip(COND + ["轨", "候选对"], c), 重复=len(h), 真值相对收益均值_最终=h.G_最终.mean(),
                 SE_重复间=h.G_最终.std(ddof=1) / np.sqrt(len(h)), 最小_最终=h.G_最终.min(), 最大_最终=h.G_最终.max(),
                 达m_最终=int(h.达m_最终.sum()), 达m_初始评价=int(h.达m_初始.sum()), 达m口径不同的重复=int((h.达m_最终 != h.达m_初始).sum()),
                 追加评价重复=int(h.追加评价.sum()), 追加后仍在边界附近=int(h.边界附近_最终.sum()),
                 真值相对收益均值_初始评价=h.G_初始.mean())
        lo, hi = wilson(r["达m_最终"], len(h)); r.update(达m_最终_Wilson下=lo, 达m_最终_Wilson上=hi)
        rows.append(r)
    TB = pd.DataFrame(rows)
    # 表 2：选择器逐轨（错误启用拆分）
    S["带符号差_相对两纯策略较优者"] = S.regret; S["带符号差_相对两纯策略较优者_相对"] = S.regret_相对
    bin_sel = sorted(S[S.专业化.notna()].选择器.unique())
    rows = []
    for c, h in S.groupby(COND + ["选择器", "轨"]):
        r = dict(zip(COND + ["选择器", "轨"], c), 重复=len(h),
                 带符号差相对均值=h.带符号差_相对两纯策略较优者_相对.mean(),
                 带符号差相对SE=h.带符号差_相对两纯策略较优者_相对.std(ddof=1) / np.sqrt(len(h)),
                 带符号差为负的重复=int((h.带符号差_相对两纯策略较优者 < 0).sum()),
                 带符号差最小=h.带符号差_相对两纯策略较优者.min(), 判断所依候选对=h.判断所依候选对.iloc[0])
        if c[4] in bin_sel:
            sp = h.专业化.astype(bool); G = h.候选对真值相对收益
            neg = ~(G >= M)
            r.update(启用=int(sp.sum()), 不达标重复=int(neg.sum()), 错误启用=int((sp & neg).sum()),
                     错误启用_实际变差=int((sp & (G < 0)).sum()), 错误启用_未达m未变差=int((sp & (G >= 0) & (G < M)).sum()),
                     达标重复=int((~neg).sum()), 漏启用=int((~sp & ~neg).sum()))
        rows.append(r)
    TC = pd.DataFrame(rows)
    # 表 3：同一重复中至少一条不达标轨被启用（逐条件，不跨机制合并）
    rows = []
    Sb = S[S.选择器.isin(bin_sel)].copy()
    Sb["错启"] = Sb.专业化.astype(bool) & (Sb.候选对真值相对收益 < M)
    Sb["变差启"] = Sb.专业化.astype(bool) & (Sb.候选对真值相对收益 < 0)
    Sb["不达标"] = Sb.候选对真值相对收益 < M
    for c, h in Sb.groupby(COND + ["选择器"]):
        per = h.groupby("rep").agg(错=("错启", "any"), 变差=("变差启", "any"), 有不达标轨=("不达标", "any"))
        r = dict(zip(COND + ["选择器"], c), 独立重复=len(per), 至少一轨不达标的重复=int(per.有不达标轨.sum()))
        r.update(prop(int(per.错.sum()), len(per), "≥1错误启用_"))
        r.update(prop(int(per.变差.sum()), len(per), "≥1实际变差启用_"))
        k2 = int(per[per.有不达标轨].错.sum()); n2 = int(per.有不达标轨.sum())
        r.update(prop(k2, n2, "条件于存在不达标轨_≥1错误启用_"))
        rows.append(r)
    TF = pd.DataFrame(rows)
    # 汇总：候选对表口径差异计数
    diff = TB[TB.达m口径不同的重复 > 0][COND + ["轨", "候选对", "达m_初始评价", "达m_最终", "达m口径不同的重复"]]
    summ = dict(候选对口径不同的单元格=len(diff), 候选对口径不同的重复合计=int(TB.达m口径不同的重复.sum()),
                追加评价重复对合计=int(T.追加评价.sum()), 追加后仍在边界附近合计=int(T.边界附近_最终.sum()),
                选择器表真值一致性断言_不一致行=mism, 带符号差为负的行=int((S.带符号差_相对两纯策略较优者 < 0).sum()),
                选择器表行数=len(S), 有二元决定的选择器=bin_sel,
                **{"FCT|P*_各条件≥1错误启用比例最大": float(TF[TF.选择器 == "FCT|P*"]["≥1错误启用_比例"].max())})
    with pd.ExcelWriter(OUT / "R45_R43FIX.xlsx", engine="openpyxl") as xw:
        pd.DataFrame([summ]).astype(str).T.reset_index().to_excel(xw, sheet_name="0_汇总", index=False)
        diff.to_excel(xw, sheet_name="1_达标数口径差异", index=False)
        TB.to_excel(xw, sheet_name="2_候选对_最终truth", index=False)
        TC.to_excel(xw, sheet_name="3_选择器逐轨_更正", index=False)
        TF.to_excel(xw, sheet_name="4_重复级至少一轨错误启用", index=False)
    (OUT / "R45_R43FIX.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1, default=float))
    print(diff.to_string())
    return dict(TB=TB, TC=TC, TF=TF, diff=diff, summ=summ)


if __name__ == "__main__":
    main()
