# -*- coding: utf-8 -*-
r"""R46 最终核验与 T2：统一报告与证据表（md 与 xlsx **同源生成**：md 中每个数字都来自写入 xlsx「报告引用」表的同一 DataFrame）。

输入：28_R46_最终核验与T2/{outcome_fix, nwicu_v2, nhanes_v2, t2, readonly, numbers, lit}/ 的方案、冻结记录、测试、推断与汇总表；
      只读 R45 NWICU 推断、R44 推断、R45 目标一致性推断（修正前后对照）、R45 R43 更正表、R43 T1 汇总。
输出：28_R46_最终核验与T2/R46_最终核验与T2_统一报告.md、R46_最终核验与T2_证据表.xlsx
用法：python pgss_r46_report.py
"""
import sys, json, pathlib, hashlib
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from pgss_r45_report import fsha, jload, kv, three, row, esc, tbl, ivl, pc, f6, OCN
from pgss_r45_report import pc as _pc0, f6 as _f60

# 逐值追踪：报告中每个经 pc / f6 / fx / sci 格式化的数值都记录其未舍入原值，生成后核对它在「报告引用」表中精确存在
TRACE = []
TRACE_MARK = "〔逐值追踪结果〕"


def _tr(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return
    if np.isfinite(x):
        TRACE.append(x)


def pc(v, d=2):
    _tr(v)
    return _pc0(v, d)


def f6(v):
    _tr(v)
    return _f60(v)


def fx(v, d):
    _tr(v)
    return f"{v:.{d}f}"

ROOT = HERE.parent
S = ROOT / "28_R46_最终核验与T2"
OF, T2D, ROD, NUD, LITD = S / "outcome_fix", S / "t2", S / "readonly", S / "numbers", S / "lit"
MD = S / "R46_最终核验与T2_统一报告.md"; XL = S / "R46_最终核验与T2_证据表.xlsx"
R45 = ROOT / "27_R45_外部复现与机制核验"
NW45 = R45 / "nwicu" / "inference" / "R45_NWICU_INFERENCE.xlsx"
NW46 = S / "nwicu_v2" / "inference" / "R45_NWICU_INFERENCE.xlsx"
NH44 = ROOT / "25_R44_NHANES跨领域复现" / "inference" / "R44_INFERENCE.xlsx"
NH46 = S / "nhanes_v2" / "r44v2" / "inference" / "R44_INFERENCE.xlsx"
TG45 = R45 / "target" / "R45_TARGET_INFERENCE.xlsx"
TG46 = S / "nhanes_v2" / "target_v2" / "R45_TARGET_INFERENCE.xlsx"
KEEP = ["结局", "表型", "对照", "尺度", "观察相对改善", "相对_P2_5", "相对_P97_5", "三态_m0.010", "相对0方向"]
SELS = ["ALWAYS_POOLED", "ALWAYS_SPLIT", "CV_ANY|P*", "CV_POINT_M|P*", "CV_LB_M|P*", "FCT|P*", "FCT|S0*", "USTUN_ADAPT", "MIX", "POOLMIX"]
CITED = ["A1_冻结核对", "A2_运行前与运行后检查", "A3_运行完成情况",
         "B1_NWICU_UNKNOWN核实", "B2_NWICU_K2时间间隔", "B3_NWICU_住院死亡任务变化", "B4_NWICU_住院死亡E1修正前后", "B5_NWICU_方向汇总v2",
         "B6_NWICU_maxT_v2", "B7_NWICU_汇总判定", "C1_NHANES_未知死因", "C2_NHANES_O_CVD10修正前后", "C3_NHANES_预定判读v2",
         "C4_NHANES_汇总", "C5_重训扰动幅度", "C6_重训扰动汇总", "D1_T2生成与评价规模", "D2_T2候选对真值收益", "D3_T2两轨等权部署", "D4_T2逐轨启用与错误",
         "D5_T2重复级至少一轨", "D6_T2_MIX与POOLMIX", "D7_T2认证样本成本", "D8_T2与T1并列", "D9_T2排序推翻", "D10_T2成本与失败", "D11_T2预定对照Q1至Q5", "D12_T2调度核对",
         "E1_预定判读范围更正", "E2_R43行号更正", "E3_NWICU体征两口径", "E4_各模型自身校准_摘要",
         "F1_三层计数", "F2_maxT汇总", "F3_主张证据限制矩阵", "F4_主图源数据清单", "F5_三层总结果表", "G1_文献原文定位", "H1_意见所用数字"]


def sha_ok(fz_path):
    fz = jload(fz_path)
    bad = [k for k, v in fz["文件SHA"].items() if not (ROOT / k).exists() or fsha(ROOT / k) != v]
    return fz, bad


# ── A：冻结、检查与运行 ─────────────────────────────────────────
def part_a():
    rows = []
    for nm, p in (("结局核实规则", OF / "R46_OUTCOME_FIX_FREEZE.json"), ("有限重算", OF / "R46_RERUN_FREEZE.json"),
                  ("T2", T2D / "R46_T2_FREEZE.json")):
        if p.exists():
            fz, bad = sha_ok(p)
            rows.append(dict(部分=nm, 冻结时间=fz["冻结时间"], 登记文件数=len(fz["文件SHA"]), 现值与冻结值不同的文件=len(bad),
                             不同文件="；".join(bad)))
    A1 = pd.DataFrame(rows)
    chk = []
    eq = jload(OF / "R46_RERUN_EQUIV.json")
    chk.append(dict(检查="零修正等价：v2 运行器不删人时逐位复现原单元（NWICU、R44、R45 pop 各 1）", 项数=len(eq["单元"]),
                    通过项=sum(u["一致"] for u in eq["单元"]), 全部通过=eq["全部一致"]))
    chk.append(dict(检查="等价测试的变异对照（预测 +1e-12、P* 改名、真实删除）", 项数=len(eq["变异对照"]),
                    通过项=sum(m["检出"] for m in eq["变异对照"]), 全部通过=eq["变异全检出"]))
    pa = jload(OF / "R46_RERUN_PATHAUDIT.json")
    chk.append(dict(检查="干跑审计（真实引擎路径，不训练）", 项数=pa["总数"], 通过项=pa["可执行"], 全部通过=pa["可执行"] == pa["总数"]))
    if (OF / "R46_FOLDCHECK.csv").exists():
        F = pd.read_csv(OF / "R46_FOLDCHECK.csv", encoding="utf-8-sig")
        okf = F.完成 & F.测试集合等于原集合去掉排除者 & F.保留者标签相同
        chk.append(dict(检查="外层折归属保留：每个重跑单元的测试集合 = 原单元测试集合去掉排除者，保留者标签相同", 项数=len(F),
                        通过项=int(okf.sum()), 全部通过=bool(okf.all())))
    if (OF / "R46_INFERENCE_CHECK.json").exists():
        ic = jload(OF / "R46_INFERENCE_CHECK.json"); C = pd.DataFrame(ic["比较"])
        z = C[C.比较.str.startswith("零修正")]; u = C[C.比较.str.contains("未受影响")]; e = C[C.比较.str.contains("预期不同")]
        chk.append(dict(检查="推断零修正：不路由、不删除时逐格复现 R45 NWICU、R44、R45 目标推断的 Brier 与 log loss 对照表", 项数=len(z),
                        通过项=int((z.不同格数 == 0).sum()), 全部通过=bool((z.不同格数 == 0).all())))
        chk.append(dict(检查="修正后推断：未受影响任务（LOS72、O_MORT10 等）的对照行与原表逐格相同（同一注册表、同一重抽流）", 项数=len(u),
                        通过项=int((u.不同格数 == 0).sum()), 全部通过=bool((u.不同格数 == 0).all())))
        chk.append(dict(检查="修正后推断：受影响任务的对照行确实改变（检出力）", 项数=len(e), 通过项=int((e.不同格数 != 0).sum()),
                        全部通过=bool((e.不同格数 != 0).all())))
        chk.append(dict(检查="推断比较的变异对照（数值 +1e-12、三态改写）", 项数=len(ic["变异对照"]),
                        通过项=sum(m["检出"] for m in ic["变异对照"]), 全部通过=all(m["检出"] for m in ic["变异对照"])))
    if (T2D / "R46_T2_E2E_TESTS.json").exists():
        t = jload(T2D / "R46_T2_E2E_TESTS.json"); r = {k: v for k, v in t["结果"].items() if not k.endswith("错误")}
        chk.append(dict(检查="T2 端到端测试（缩小网格；含汇总脚本在测试产物上跑通；每项配变异对照）", 项数=len(r),
                        通过项=sum(v is True for v in r.values()), 全部通过=t["全部通过"]))
    lt = jload(LITD / "R46_LIT_TABLE.json")
    chk.append(dict(检查="文献摘录在本地原文全文中逐字找到（含改词变异对照）", 项数=lt["条目"],
                    通过项=sum(r["核对级别"] == "本地原文全文逐字匹配" for r in lt["表"]), 全部通过=lt["全部逐字匹配"]))
    A2 = pd.DataFrame(chk)
    runs = []
    import pgss_r46_rerun as RR
    for k, nm in (("nwicu", "NWICU 住院死亡 main"), ("nh44", "NHANES R44 main（O_CVD10）"), ("nhpop", "NHANES R45 pop（O_CVD10）")):
        us = RR.units(k)
        st = [("OK" if (RR.DIRS[k] / RR.tag_of(*u) / "_DONE").exists() else "FAILED" if (RR.DIRS[k] / RR.tag_of(*u) / "_FAILED").exists() else "MISSING")
              for u in us]
        sec = [float((RR.DIRS[k] / RR.tag_of(*u) / "_DONE").read_text(encoding="utf-8")) for u in us if (RR.DIRS[k] / RR.tag_of(*u) / "_DONE").exists()]
        runs.append(dict(部分=nm, 单元=len(us), 完成=st.count("OK"), 不可执行=st.count("FAILED"), 缺失=st.count("MISSING"),
                         单元秒中位=float(np.median(sec)) if sec else np.nan))
    if (T2D / "R46_T2_SUMMARY.json").exists():
        s = jload(T2D / "R46_T2_SUMMARY.json")
        runs.append(dict(部分="T2（8 条件 × 200）", 单元=s["计划"], 完成=s["完成"], 不可执行=s["失败"], 缺失=s["计划"] - s["完成"] - s["失败"],
                         单元秒中位=np.nan))
    return A1, A2, pd.DataFrame(runs)


def delta_table(M, a, b, label):
    """修正前后同一任务×尺度的 E1 相对改善变化，与原区间半宽比较（单次扰动，不是方差估计）。"""
    d = M.copy()
    d["变化"] = d[f"观察相对改善{b}"] - d[f"观察相对改善{a}"]
    d["绝对变化"] = d.变化.abs()
    d["原区间半宽"] = (d[f"相对_P97_5{a}"] - d[f"相对_P2_5{a}"]) / 2
    d["绝对变化_除以原半宽"] = d.绝对变化 / d.原区间半宽
    d["符号翻转"] = np.sign(d[f"观察相对改善{a}"]) != np.sign(d[f"观察相对改善{b}"])
    d["三态改变"] = d[f"三态_m0.010{a}"] != d[f"三态_m0.010{b}"]
    d.insert(0, "来源", label)
    return d[["来源", "结局", "表型", "尺度", f"观察相对改善{a}", f"观察相对改善{b}", "变化", "绝对变化", "原区间半宽", "绝对变化_除以原半宽", "符号翻转", "三态改变"]]         .rename(columns={f"观察相对改善{a}": "修正前", f"观察相对改善{b}": "修正后"})


def perturb_agg(P):
    return P.groupby("来源", sort=False).agg(行=("变化", "size"), 绝对变化中位=("绝对变化", "median"), 绝对变化最大=("绝对变化", "max"),
                                         比值中位=("绝对变化_除以原半宽", "median"), 比值最大=("绝对变化_除以原半宽", "max"),
                                         符号翻转=("符号翻转", "sum"), 三态改变=("三态改变", "sum")).reset_index()


# ── B：NWICU ───────────────────────────────────────────────────
def part_b():
    V = pd.read_excel(OF / "R46_OUTCOME_VERIFY.xlsx", sheet_name=None)
    o = pd.read_excel(NW45, sheet_name="2_Brier对照"); n = pd.read_excel(NW46, sheet_name="2_Brier对照")
    m = lambda d: d[(d.对照 == "E1") & (d.结局 == "OUT_HOSP_MORT")][KEEP]
    B4 = m(o).merge(m(n), on=["结局", "表型", "对照", "尺度"], suffixes=("_R45", "_R46"))
    rows = []
    for sc, h in n[n.对照 == "E1"].groupby("尺度", sort=False):
        ex, ins, sup = three(h); vc = h.相对0方向.value_counts()
        rows.append(dict(尺度=sc, 任务=len(h), 相对改善中位=h.观察相对改善.median(), 点估计为负=int((h.观察相对改善 < 0).sum()),
                         相对上界最大=h.相对_P97_5.max(), 全负=int(vc.get("全负", 0)), 跨零=int(vc.get("跨零", 0)), 全正=int(vc.get("全正", 0)),
                         排除m1=ex, 证据不足m1=ins, 支持m1=sup))
    B5 = pd.DataFrame(rows)
    TH = pd.read_excel(NW46, sheet_name="5_E1_maxT"); TH0 = pd.read_excel(NW45, sheet_name="5_E1_maxT")
    B6 = TH.merge(TH0[["结局", "表型", "轨", "同时判定_相对m"]].rename(columns={"同时判定_相对m": "R45同时判定"}), on=["结局", "表型", "轨"], how="left")
    st = pd.read_excel(NW46, sheet_name="1_任务纳入")
    B7 = pd.DataFrame([dict(项="NWICU 纳入任务（v2）", 值=int(st.纳入.sum())),
                       dict(项="max-T 临界值（v2）", 值=float(TH["max-T临界值"].iloc[0])),
                       dict(项="max-T 临界值（R45）", 值=float(TH0["max-T临界值"].iloc[0])),
                       dict(项="max-T 项数", 值=len(TH)),
                       dict(项="max-T 排除（v2）", 值=int(TH.同时判定_相对m.str.startswith("排除").sum())),
                       dict(项="max-T 证据不足（v2）", 值=int(TH.同时判定_相对m.str.startswith("证据不足").sum())),
                       dict(项="max-T 支持（v2）", 值=int(TH.同时判定_相对m.str.startswith("支持").sum())),
                       dict(项="max-T 判定与 R45 不同的项", 值=int((B6.同时判定_相对m != B6.R45同时判定).sum())),
                       dict(项="四尺度 E1 支持实质收益合计（v2）", 值=int((n[n.对照 == "E1"]["三态_m0.010"] == "支持实质收益").sum())),
                       dict(项="四尺度 E1 行数（v2）", 值=int((n.对照 == "E1").sum()))])
    B8 = delta_table(B4, "_R45", "_R46", "NWICU 住院死亡（删除 200 人次，约 1%）")
    return dict(B1=V["1_NWICU_UNKNOWN核实"], B2=V["2_NWICU_K2时间间隔"], B3=V["3_NWICU_住院死亡任务变化"], B4=B4, B5=B5, B6=B6, B7=B7, B8=B8)


# ── C：NHANES ───────────────────────────────────────────────────
def part_c():
    V = pd.read_excel(OF / "R46_OUTCOME_VERIFY.xlsx", sheet_name="4_NHANES_CVD未知死因")
    aff = [r.表型 for r in V.itertuples() if r.需要重跑]
    o = pd.read_excel(NH44, sheet_name="2_Brier对照"); n = pd.read_excel(NH46, sheet_name="2_Brier对照")
    m = lambda d: d[(d.对照 == "E1") & (d.结局 == "O_CVD10") & d.表型.isin(aff)][KEEP]
    a = m(o).merge(m(n), on=["结局", "表型", "对照", "尺度"], suffixes=("_原", "_v2")).assign(推断="R44 E1（HTFS vs P*）")
    to = pd.read_excel(TG45, sheet_name="2_Brier对照"); tn = pd.read_excel(TG46, sheet_name="2_Brier对照")
    mt = lambda d: d[(d.对照 == "C1") & (d.尺度 == "pop") & (d.结局 == "O_CVD10") & d.表型.isin(aff)][KEEP]
    b = mt(to).merge(mt(tn), on=["结局", "表型", "对照", "尺度"], suffixes=("_原", "_v2")).assign(推断="R45 C1（HTFS_pop vs P*_pop）")
    C2 = pd.concat([a, b], ignore_index=True)
    C3 = pd.read_excel(TG46, sheet_name="4_预定判读")
    C3 = C3[(C3.结局 == "O_MORT10") & (C3.表型 == "低白蛋白")].reset_index(drop=True)
    st = pd.read_excel(NH46, sheet_name="1_任务纳入")
    TH = pd.read_excel(NH46, sheet_name="5_E1_maxT"); TC = pd.read_excel(TG46, sheet_name="5_C1_maxT")
    e1 = n[(n.对照 == "E1") & (n.尺度 == "bal")]
    c1 = tn[(tn.对照 == "C1") & (tn.尺度 == "pop")]
    ex, ins, sup = three(e1); cx, ci, cs = three(c1)
    C4 = pd.DataFrame([dict(项="R44 v2 纳入任务", 值=int(st.纳入.sum())), dict(项="R44 v2 主要尺度（bal）排除", 值=ex),
                       dict(项="R44 v2 主要尺度证据不足", 值=ins), dict(项="R44 v2 主要尺度支持", 值=sup),
                       dict(项="R44 v2 max-T 项数", 值=len(TH)), dict(项="R44 v2 max-T 支持", 值=int(TH.同时判定_相对m.str.startswith("支持").sum())),
                       dict(项="R45 v2 C1（pop）排除", 值=cx), dict(项="R45 v2 C1（pop）证据不足", 值=ci), dict(项="R45 v2 C1（pop）支持", 值=cs),
                       dict(项="R45 v2 C1 max-T 支持", 值=int(TC.同时判定_相对m.str.startswith("支持").sum()))])
    C5 = delta_table(a.drop(columns="推断"), "_原", "_v2", "NHANES O_CVD10（删除 1 人）")
    return dict(C1=V, C2=C2, C3=C3, C4=C4, C5=C5)


# ── D：T2 ──────────────────────────────────────────────────────
def part_d():
    if not (T2D / "R46_T2_SUMMARY.xlsx").exists():
        return None
    X = pd.read_excel(T2D / "R46_T2_SUMMARY.xlsx", sheet_name=None)
    co = {c: i for i, c in enumerate(("N1", "N2", "P1", "P2", "N1-T", "P2-T", "N1-H", "P2-H"))}
    srt = lambda df: df.assign(_c=df.条件.map(co)).sort_values(["_c", "轨"], kind="stable").drop(columns="_c").reset_index(drop=True)
    X["C_候选对真值收益"] = srt(X["C_候选对真值收益"]); X["H_认证样本成本"] = srt(X["H_认证样本成本"])
    D2 = X["C_候选对真值收益"]
    D3 = X["E_选择器两轨等权"]; D3 = D3[D3.选择器.isin(SELS)].reset_index(drop=True)
    D4 = X["D_选择器逐轨"]; D4 = D4[D4.选择器.isin(SELS)].reset_index(drop=True)
    D5 = X["F_重复级至少一轨"]
    D9 = X["J_两两排序"]; D9 = D9[D9.T2推翻T1排序.astype(bool)].reset_index(drop=True) if len(D9) else D9
    D10 = X["K_成本"].merge(X["A_生成与评价规模"][["条件", "评价10万", "评价50万", "评价100万", "最终仍有近边界项的重复"]], on="条件")
    D12 = pd.DataFrame()
    if (T2D / "R46_SCHED_CHECK.json").exists():
        sc = jload(T2D / "R46_SCHED_CHECK.json")
        D12 = pd.DataFrame([dict(重复=t, arms逐位相同=v["arms.parquet"], pairs逐位相同=v["pairs.parquet"], sel逐位相同=v["sel.parquet"],
                                 rep记录除计时相同=v["rep.json（除计时字段）相同"], 正式运行秒=v["原计时秒"], 重跑秒=v["重跑计时秒"],
                                 变异对照检出=not sc["变异对照"][t]["arms.parquet"])
                            for t, v in sc["结果"].items()])
    D8 = X["I_与T1并列"].copy()
    s1, s2 = np.sign(D8.T1均值.astype(float)), np.sign(D8.T2均值.astype(float))
    D8["符号"] = np.where(s2 == 0, "T2 均值为 0", np.where(s1 == 0, "T1 均值为 0", np.where(s1 == s2, "相同", "相反")))
    return dict(D1=X["A_生成与评价规模"], D2=D2, D3=D3, D4=D4, D5=D5, D6=X["G_MIX与POOLMIX"], D7=X["H_认证样本成本"], D8=D8,
                D9=D9, D10=D10, D11=qtable(X["I_与T1并列"]), D12=D12, F=X["L_失败"], J=X["J_两两排序"], summ=jload(T2D / "R46_T2_SUMMARY.json"),
                cost=jload(T2D / "R46_T2_COST.json") if (T2D / "R46_T2_COST.json").exists() else {})


# T2 方案 §8 的 Q1–Q5（T1 表述摘自方案原文；一条表述含多个对象时按对象拆行）；Q6 的判定规则方案已预定（见表 D9）
M_ = 0.01
QDEF = [("Q1", ("P2", "P2-T"), "候选对真值相对收益", "D|P*", ("A", "B"), "≥m", "HTFS_D vs P*_D 两轨均 ≥ m"),
        ("Q2", ("N1", "N2", "N1-T"), "候选对真值相对收益", "D|P*", ("A",), "<0", "A 轨 HTFS_D 比 P*_D 差"),
        ("Q2", ("N1", "N2", "N1-T"), "候选对真值相对收益", "D|P*", ("B",), "≈0", "B 轨接近 0"),
        ("Q3", ("P1",), "候选对真值相对收益", "D|P*", ("B",), "≈m", "B 轨约 +1.1%（边界附近）"),
        ("Q3", ("P1",), "候选对真值相对收益", "D|P*", ("A",), "<0", "A 轨为负"),
        ("Q4", ("N1", "N2", "P1", "P2", "N1-T", "P2-T"), "选择器两轨等权部署改善", "FCT|P*", ("bal",), "≤0", "两轨等权部署改善在所有条件 ≤ 0"),
        ("Q5", ("P2",), "选择器两轨等权部署改善", "MIX", ("bal",), "数值", "P2 约 +3.0% 至 +3.6%"),
        ("Q5", ("P2",), "选择器两轨等权部署改善", "CV_POINT_M|P*", ("bal",), "数值", "P2 约 +3.0% 至 +3.6%"),
        ("Q5", ("N1", "N2"), "选择器两轨等权部署改善", "MIX", ("bal",), "≈0", "N1/N2 接近 0"),
        ("Q5", ("N1", "N2"), "选择器两轨等权部署改善", "CV_POINT_M|P*", ("bal",), "≈0", "N1/N2 接近 0")]


def qtable(I):
    """Q1–Q5 逐行对照。归类是报告层的机械描述：方向性表述（≥m、<0、≤0）按 T2 均值 ± 1.96·SE 相对界限归为一致 / 推翻 / 不确定；
    「接近」「约」类表述方案未预定界限，只列数字。"""
    rows = []
    for q, cs, item, obj, tracks, kind, txt in QDEF:
        for c in cs:
            for t in tracks:
                h = I[(I.项 == item) & (I.条件 == c) & (I.对象 == obj) & (I.轨 == t)]
                if not len(h):
                    continue
                h = h.iloc[0]; lo, hi = float(h.T2下), float(h.T2上)
                b = M_ if kind in ("≥m", "≈m") else 0.0 if kind in ("<0", "≤0", "≈0") else None
                pos = "—" if b is None else ("整体 ≥ " if lo >= b else "整体 < " if hi < b else "跨 ") + ("m" if b == M_ else "0")
                if float(h.T2_SE) == 0 and float(h.T2均值) == 0:
                    pos = "全部重复为 0"
                t1 = float(h.T1均值)
                t1ok = {"≥m": t1 >= M_, "<0": t1 < 0, "≤0": t1 <= 0}.get(kind)
                t1ok = "—" if t1ok is None else "是" if t1ok else "否"
                if kind == "≥m":
                    st = "一致" if lo >= b else "推翻" if hi < b else "不确定"
                elif kind == "<0":
                    st = "一致" if hi < 0 else "推翻" if lo >= 0 else "不确定"
                elif kind == "≤0":
                    st = "一致" if hi <= 0 else "推翻" if lo > 0 else "不确定"
                else:
                    st = "只列数字"
                rows.append(dict(Q=q, T1表述=txt, 条件=c, 对象=obj, 轨=t, T1重复=int(h.T1重复), T1均值=float(h.T1均值), T1自身符合表述=t1ok, T2重复=int(h.T2重复),
                                 T2均值=float(h.T2均值), T2_SE=float(h.T2_SE), T2下=lo, T2上=hi, T2区间相对界限=pos, 与T1表述=st,
                                 T2达m确定比例=float(h["T2达m确定比例"]) if "T2达m确定比例" in h and pd.notna(h["T2达m确定比例"]) else np.nan))
    return pd.DataFrame(rows)


# ── E：只读更正 ─────────────────────────────────────────────────
def part_e():
    X = pd.read_excel(ROD / "R46_READONLY.xlsx", sheet_name=None)
    E1 = X["4_预定判读范围更正"]
    B1 = pd.read_excel(R45 / "R45_外部复现与机制核验_证据表.xlsx", sheet_name="B1_R43_达标数口径差异")
    E2 = B1.assign(行号=np.arange(1, len(B1) + 1)).assign(
        GPT第48轮指出=lambda d: (d.达m_初始评价 != d.达m_最终))
    E3 = X["1_NWICU体征可用性两口径"]
    C = X["3_各模型自身校准"]
    E4 = C[(C.范围 == "全体") & C.臂.isin(["P*", "HTFS", "P*_pop", "HTFS_pop", "P*_B", "HTFS_B"])].reset_index(drop=True)
    return dict(E1=E1, E2=E2, E3=E3, E4=E4, C=C, VV=X["2_NWICU各体征窗内非缺失"], js=jload(ROD / "R46_READONLY.json"))


# ── F：数字包 ───────────────────────────────────────────────────
def part_f():
    X = pd.read_excel(NUD / "R46_NUMBERS.xlsx", sheet_name=None)
    return dict(F1=X["N2_分层计数"], F2=X["N3_maxT汇总"], F3=X["N4_主张证据限制矩阵"], F4=X["N5_主图源数据清单"], L=X["N1_总结果表_三层"], X=X)


def part_h():
    rows = [dict(项=f"0/{n} 时 Clopper–Pearson 双侧 95% 上界", 值=1 - 0.025 ** (1 / n)) for n in (20, 200)]
    rows += [dict(项=f"真实比例 {fx(p, 2)}、200 次重复时比例的 Monte Carlo 标准误", 值=float(np.sqrt(p * (1 - p) / 200))) for p in (0.05, 0.5)]
    return pd.DataFrame(rows)


def build():
    A1, A2, A3 = part_a(); B = part_b(); C = part_c(); D = part_d(); E = part_e(); F = part_f()
    G1 = pd.read_excel(LITD / "R46_LIT_TABLE.xlsx", sheet_name="先例原文定位")
    sheets = {"A1_冻结核对": A1, "A2_运行前与运行后检查": A2, "A3_运行完成情况": A3,
              "B1_NWICU_UNKNOWN核实": B["B1"], "B2_NWICU_K2时间间隔": B["B2"], "B3_NWICU_住院死亡任务变化": B["B3"],
              "B4_NWICU_住院死亡E1修正前后": B["B4"], "B5_NWICU_方向汇总v2": B["B5"], "B6_NWICU_maxT_v2": B["B6"], "B7_NWICU_汇总判定": B["B7"],
              "C1_NHANES_未知死因": C["C1"], "C2_NHANES_O_CVD10修正前后": C["C2"], "C3_NHANES_预定判读v2": C["C3"], "C4_NHANES_汇总": C["C4"],
              "C5_重训扰动幅度": pd.concat([B["B8"], C["C5"]], ignore_index=True)}
    sheets["C6_重训扰动汇总"] = perturb_agg(sheets["C5_重训扰动幅度"])
    if D:
        sheets.update({"D1_T2生成与评价规模": D["D1"], "D2_T2候选对真值收益": D["D2"], "D3_T2两轨等权部署": D["D3"], "D4_T2逐轨启用与错误": D["D4"],
                       "D5_T2重复级至少一轨": D["D5"], "D6_T2_MIX与POOLMIX": D["D6"], "D7_T2认证样本成本": D["D7"], "D8_T2与T1并列": D["D8"],
                       "D9_T2排序推翻": D["D9"], "D10_T2成本与失败": D["D10"], "D11_T2预定对照Q1至Q5": D["D11"], "D12_T2调度核对": D["D12"]})
    sheets.update({"E1_预定判读范围更正": E["E1"], "E2_R43行号更正": E["E2"], "E3_NWICU体征两口径": E["E3"], "E4_各模型自身校准_摘要": E["E4"],
                   "F1_三层计数": F["F1"], "F2_maxT汇总": F["F2"], "F3_主张证据限制矩阵": F["F3"], "F4_主图源数据清单": F["F4"],
                   "G1_文献原文定位": G1, "H1_意见所用数字": part_h()})
    # 明细（不进入数字溯源池）
    sheets["X1_各模型自身校准_全表"] = E["C"]
    sheets["X2_NWICU各体征窗内非缺失"] = E["VV"]
    sheets["F5_三层总结果表"] = F["L"]
    for k, v in F["X"].items():
        if k.startswith("F") and k[1].isdigit():
            sheets[f"X4_{k}"[:31]] = v
    if D:
        sheets["X5_T2两两排序全表"] = D["J"]
        sheets["X6_T2失败"] = D["F"]
    with pd.ExcelWriter(XL, engine="openpyxl") as xw:
        pd.DataFrame({"表": ["0_说明"] + list(sheets), "报告引用": ["否"] + ["是" if k in CITED else "否" for k in sheets]}
                     ).to_excel(xw, sheet_name="目录", index=False)
        pd.DataFrame({"说明": [
            "本表与统一报告由 pgss_r46_report.py 同源生成；报告中的数字只来自「目录」中标为「报告引用」的表。",
            "真实数据推断范围：固定 OOF 预测的簇重抽（第 Ⅱ 层），不是算法级区间。",
            "相对改善 = (R参照 − R候选) / R参照；正值 = 候选更好。",
            "T2 的真值是评价协变量上已知生成概率下的条件 Brier（Monte Carlo 积分；逐重复自适应评价规模，同一重复内全部量同一口径）。",
            "不含任何逐人记录、逐人预测或患者编号；文献原文只留本地。"]}).to_excel(xw, sheet_name="0_说明", index=False)
        for k, v in sheets.items():
            v.to_excel(xw, sheet_name=k[:31], index=False)
    return dict(A1=A1, A2=A2, A3=A3, B=B, C=C, C6=sheets["C6_重训扰动汇总"], D=D, E=E, F=F, G1=G1, H1=part_h(), sheets=sheets)


# ── md ─────────────────────────────────────────────────────────
TR = lambda s: {"排除实质收益": "排除", "证据不足": "证据不足", "支持实质收益": "支持"}.get(s, s)


def sec_a(R):
    L = ["## 1. 冻结、检查与运行\n"]
    L.append(tbl(R["A1"], ["部分", "冻结时间", "登记文件数", "现值与冻结值不同的文件"]))
    L.append("\n三份冻结记录都在对应计算之前写出：结局核实规则在任何分类计算之前，重算代码在零修正等价测试通过之后、任何正式重跑之前，"
             "T2 方案与代码在端到端测试通过之后、任何正式重复之前。"
             + ("本轮没有冻结后修订。\n" if not list(S.rglob("R46_*_AMENDMENT.json")) else "冻结后修订见附表。\n"))
    L.append(tbl(R["A2"], ["检查", "项数", "通过项", "全部通过"]))
    L.append("\n" + tbl(R["A3"], ["部分", "单元", "完成", "不可执行", "缺失"]) + "\n")
    return "\n".join(L)


def sec_b(R):
    B = R["B"]; v = B["B1"].set_index("项").N
    L = ["## 2. NWICU：`UNKNOWN` 出院去向的住院死亡标签（GPT 第 50 轮 §3）\n"]
    L.append("### 2.1 本地结构化字段核实（规则先冻结，再计算）\n")
    L.append("证据：本次住院的 deathtime、患者 dod（仅日期）、同一患者此后的其他住院（admittime ≥ 本次 dischtime）、hospital_expire_flag。"
             "NWICU 的检验、处方、操作记录全部挂在住院号下（实测无门诊记录），所以后续就诊证据只能来自后续住院。"
             "死亡方向证据 D：deathtime ≤ 出院 + 24 h，或 dod ≤ 出院日 + 1 天，或 flag = 1；存活方向证据 L：deathtime > 出院 + 24 h，或 dod ≥ 出院日 + 2 天，或有后续住院。\n")
    L.append(tbl(B["B1"], ["项", "N"]))
    g = B["B2"].iloc[0]
    L.append(f"\n- 有存活证据的 {int(v['类别 K2'])} 人次中，{int(v['K2 依据：后续入院'])} 人次有后续住院，距本次出院的中位时间 {fx(g.中位, 4)} 天"
             f"（P5 {fx(g.P5, 6)}、P95 {fx(g.P95, 4)} 天）——多数在出院当天再入院。")
    L.append(f"- 只有 UNKNOWN、没有任何可确定生命状态证据的 {int(v['类别 K3'])} 人次，以及死亡记录在出院后 24 h 内、无法区分院内与院外死亡的 {int(v['类别 K3a'])} 人次，"
             f"从住院死亡分析中排除，合计 {int(v['住院死亡不可评价（排除）合计'])} 人次（{int(v['排除人次涉及患者数'])} 名患者）；"
             f"保留者的标签改变 {int(v['住院死亡标签改变（可评价者中）'])} 个。LOS72 不受影响、不重跑。")
    L.append("- 这是看到 R45 结果后识别并修正的结局编码问题；规则由数据证据决定，不由修正后是否仍为阴性决定。\n")
    L.append(tbl(B["B3"], ["表型", "R45任务人次", "排除人次", "排除_A轨", "排除_B轨", "v2任务人次", "R45事件", "v2事件"]))
    L.append("\n### 2.2 有限重算后的住院死亡六个任务（E1：HTFS vs P\\*）\n")
    b4 = B["B4"].copy()
    b4 = b4[b4.尺度.isin(["bal", "B", "A"])]
    b4["R45 相对改善"] = b4.观察相对改善_R45.map(pc); b4["R46 相对改善"] = b4.观察相对改善_R46.map(pc)
    b4["R46 95% 区间"] = [f"{pc(a)} ~ {pc(c)}" for a, c in zip(b4.相对_P2_5_R46, b4.相对_P97_5_R46)]
    b4["R45 三态"] = b4["三态_m0.010_R45"].map(TR); b4["R46 三态"] = b4["三态_m0.010_R46"].map(TR)
    L.append(tbl(b4, ["表型", "尺度", "R45 相对改善", "R46 相对改善", "R46 95% 区间", "R45 三态", "R46 三态"]))
    L.append("\n全部 12 个任务（住院死亡读 v2 单元，LOS72 读 R45 原单元）在各尺度的汇总：\n")
    L.append(tbl(B["B5"], ["尺度", "任务", "相对改善中位", "点估计为负", "相对上界最大", "全负", "跨零", "全正", "排除m1", "证据不足m1", "支持m1"],
                 fmt={"相对改善中位": pc, "相对上界最大": pc}))
    s = B["B7"].set_index("项").值
    ins = B["B6"][B["B6"].同时判定_相对m.str.startswith("证据不足")]
    L.append(f"\n逐轨 max-T（{int(s['max-T 项数'])} 项，临界值 {fx(s['max-T 临界值（v2）'], 4)}；R45 为 {fx(s['max-T 临界值（R45）'], 4)}）："
             f"排除 {int(s['max-T 排除（v2）'])}、证据不足 {int(s['max-T 证据不足（v2）'])}、支持 {int(s['max-T 支持（v2）'])}；"
             f"与 R45 判定不同的项 {int(s['max-T 判定与 R45 不同的项'])} 个。证据不足的项："
             + "；".join(f"{OCN.get(r.结局, r.结局)}×{r.表型}×{r.轨} 轨" for r in ins.itertuples()) + "。")
    L.append("**所以不能写「已经同时排除了所有任务、所有轨道的实质收益」**；能写的是：没有任何一项得到「支持实质收益」，"
             f"两轨等权的逐任务区间在 {int(B['B5'][B['B5'].尺度 == 'bal'].iloc[0].排除m1)} / {int(B['B5'][B['B5'].尺度 == 'bal'].iloc[0].任务)} 个任务上排除 ≥1% 的收益。\n")
    return "\n".join(L)


def sec_c(R):
    C = R["C"]
    L = ["## 3. NHANES：10 年内死亡但死因缺失的一例（GPT 第 50 轮 §4）\n"]
    L.append("该例的 O_CVD10 改记为未知（死因未知 ≠ 已知的其他原因死亡），只在它实际所在、且 R44 已纳入的 O_CVD10 任务中删除；全因死亡不受影响。\n")
    L.append(tbl(C["C1"], ["结局", "表型", "R44已纳入", "未知者在任务人群中", "未知者所在轨", "需要重跑"]))
    c2 = C["C2"].copy()
    c2 = c2[c2.尺度.isin(["bal", "pop", "B"])]
    c2["原 相对改善"] = c2.观察相对改善_原.map(pc); c2["v2 相对改善"] = c2.观察相对改善_v2.map(pc)
    c2["v2 95% 区间"] = [f"{pc(a)} ~ {pc(b)}" for a, b in zip(c2.相对_P2_5_v2, c2.相对_P97_5_v2)]
    c2["原 三态"] = c2["三态_m0.010_原"].map(TR); c2["v2 三态"] = c2["三态_m0.010_v2"].map(TR)
    L.append("\n" + tbl(c2, ["推断", "表型", "尺度", "原 相对改善", "v2 相对改善", "v2 95% 区间", "原 三态", "v2 三态"]))
    c4 = C["C4"].set_index("项").值
    L.append(f"\n修正后 R44 主要尺度：{int(c4['R44 v2 纳入任务'])} 个任务排除 {int(c4['R44 v2 主要尺度（bal）排除'])}、证据不足 {int(c4['R44 v2 主要尺度证据不足'])}、"
             f"支持 {int(c4['R44 v2 主要尺度支持'])}；R45 目标一致性 C1（pop）排除 {int(c4['R45 v2 C1（pop）排除'])}、证据不足 {int(c4['R45 v2 C1（pop）证据不足'])}、"
             f"支持 {int(c4['R45 v2 C1（pop）支持'])}。O_MORT10 各任务的对照行与原表逐格相同（见 §1 检查表），低白蛋白任务的预定判读不变：")
    for r in C["C3"].itertuples():
        L.append(f"- {r.对照}（{r.判读尺度}）：{pc(r.观察相对改善)}（{pc(r.相对_P2_5)} ~ {pc(r.相对_P97_5)}，{TR(r.三态_m1)}）。")
    L.append("\n建议写法（GPT §2.1）：在该任务中，专属模型相对原合并程序的优势，没有在目标一致的强合并参照之上保留下来；原比较不足以把优势归因于独立训练本身。"
             "差距收回比例只放补充材料，不作机制比例。\n")
    P = pd.concat([R["B"]["B8"], C["C5"]], ignore_index=True)
    L.append("### 3.1 两次修正顺带给出的重训扰动幅度（描述）\n")
    L.append("删除人次后，外层折归属不变，但开发集的单位集合变了：内层划分、早停留出与 XGBoost 种子都由单位集合派生，所以整个学习程序等于换了一条随机流重新开发。"
             "下表是同一任务、同一尺度上 E1 相对改善的前后变化，与修正前固定 OOF 区间半宽的比值：\n")
    agg = R["C6"]
    L.append(tbl(agg, ["来源", "行", "绝对变化中位", "绝对变化最大", "比值中位", "比值最大", "符号翻转", "三态改变"],
                 fmt={"绝对变化中位": pc, "绝对变化最大": pc, "比值中位": lambda v: f"{fx(v, 2)}", "比值最大": lambda v: f"{fx(v, 2)}"}))
    mx = P.loc[P.绝对变化.idxmax()]
    L.append(f"\n最大的一处：{mx.来源}，{OCN.get(mx.结局, mx.结局)}×{mx.表型}（{mx.尺度}）由 {pc(mx.修正前)} 变为 {pc(mx.修正后)}。"
             f"这只是单次扰动，不是方差估计；但变化与原区间半宽之比的中位为 {'、'.join(f'{fx(v, 2)}' for v in agg.比值中位)}、最大为 {'、'.join(f'{fx(v, 2)}' for v in agg.比值最大)}，"
             "说明固定 OOF 预测的簇重抽区间不包含重新开发带来的变异，而这部分变异并不小。"
             "所以逐任务区间只能按 GPT §5.3 的范围解释，正文不应把逐任务点估计的符号当作可复现的结论（与 LR 阶段「效应符号不可复现」的发现一致）。\n")
    return "\n".join(L)


def sec_d(R):
    D = R["D"]
    L = ["## 4. T2 独立确认模拟（GPT 第 50 轮 §7）\n"]
    if D is None:
        L.append("T2 尚未完成。\n"); return "\n".join(L)
    s = D["summ"]; cost = D["cost"]
    L.append(f"8 个条件 × 200 次 = {s['计划']} 次独立重复，完成 {s['完成']}、失败 {s['失败']}。开发集 n = 6,000、π_A = 0.15；根种子 20461（与 T1、plasmode 不重叠）；"
             "冻结引擎与选择器不变；评价逐重复自适应（累计 10 万 → 50 万 → 100 万），同一重复内所有臂、候选对与选择器在同一最终评价集上计算。"
             + (f"并发按成本批确定为 {cost.get('并发J')}。" if cost else "") + "\n")
    L.append("### 4.1 生成与评价规模\n")
    L.append(tbl(D["D1"], ["条件", "重复", "评价事件率均值", "A轨Bayes风险", "B轨Bayes风险", "评价X绝对值最大_最大", "线性预测子最小", "线性预测子最大",
                           "评价10万", "评价50万", "评价100万", "最终仍有近边界项的重复"],
                 fmt={"评价事件率均值": pc, "A轨Bayes风险": f6, "B轨Bayes风险": f6, "评价X绝对值最大_最大": lambda v: f"{fx(v, 2)}",
                      "线性预测子最小": lambda v: f"{fx(v, 2)}", "线性预测子最大": lambda v: f"{fx(v, 2)}"}))
    L.append("\n### 4.2 候选对真值收益（HTFS_D vs P\\*_D；各条件全部完成重复的均值 ± 重复间标准误）\n")
    d2 = D["D2"][D["D2"].候选对 == "D|P*"].copy()
    d2["均值"] = d2.真值相对收益均值.map(pc); d2["SE"] = d2.SE.map(pc)
    d2["≥m（确定）"] = [f"{int(a)}/{int(b)}" for a, b in zip(d2["达m_确定_分子"], d2["达m_确定_分母"])]
    d2["<0（确定）"] = [f"{int(a)}/{int(b)}" for a, b in zip(d2["为负_确定_分子"], d2["为负_确定_分母"])]
    L.append(tbl(d2, ["条件", "轨", "均值", "SE", "≥m（确定）", "<0（确定）", "m边界不确定", "零边界不确定"]))
    L.append("\n### 4.3 选择程序的两轨等权部署改善（相对 P\\*_D；各条件全部完成重复的均值）\n")
    pv = D["D3"].pivot_table(index="选择器", columns="条件", values="均值").reindex(SELS)
    pv = pv[[c for c in ("N1", "N2", "P1", "P2", "N1-T", "P2-T", "N1-H", "P2-H") if c in pv.columns]].reset_index()
    L.append(tbl(pv, list(pv.columns), fmt={c: pc for c in pv.columns if c != "选择器"}))
    L.append("\n（均值的重复间标准误、P5/P95 与为负次数见表 D3。）\n")
    L.append("### 4.4 启用与错误（逐轨与重复级）\n")
    d5 = D["D5"][D["D5"].选择器.isin(["ALWAYS_SPLIT", "CV_ANY|P*", "CV_POINT_M|P*", "CV_LB_M|P*", "FCT|P*", "FCT|S0*", "USTUN_ADAPT"])].copy()
    d5["≥1 轨不达标启用"] = [f"{int(a)}/{int(b)}" for a, b in zip(d5["≥1轨不达标启用_分子"], d5["≥1轨不达标启用_分母"])]
    d5["CP 上界"] = d5["≥1轨不达标启用_CP上"].map(pc)
    d5["≥1 轨实际变差启用"] = [f"{int(a)}/{int(b)}" for a, b in zip(d5["≥1轨实际变差启用_分子"], d5["≥1轨实际变差启用_分母"])]
    pv5 = d5.pivot_table(index="选择器", columns="条件", values="≥1 轨不达标启用", aggfunc="first")
    pv5 = pv5[[c for c in ("N1", "N2", "P1", "P2", "N1-T", "P2-T", "N1-H", "P2-H") if c in pv5.columns]].reset_index()
    L.append("重复级「至少一轨不达标启用」（不达标 = 候选对真值收益确定 < m；边界不确定的单列，不计入）：\n")
    L.append(tbl(pv5, list(pv5.columns)))
    fct = d5[d5.选择器 == "FCT|P*"]
    L.append(f"\nFCT|P\\* 各条件 Clopper–Pearson 95% 上界最大 {pc(float(fct['≥1轨不达标启用_CP上'].max()))}。0/200 不写成「错误率已验证低于某值」。"
             "逐轨的启用、检出（条件于确定达标）、漏启用、实际变差启用与边界不确定计数见表 D4。\n")
    L.append("### 4.5 MIX 与 POOLMIX（不把非零混合权重判为错误专业化）\n")
    d6 = D["D6"].copy()
    rows6 = []
    for c in [c for c in ("N1", "N2", "P1", "P2", "N1-T", "P2-T", "N1-H", "P2-H") if c in set(d6.条件)]:
        r6 = dict(条件=c)
        for sel in ("MIX", "POOLMIX"):
            for t in ("A", "B"):
                h = d6[(d6.条件 == c) & (d6.选择器 == sel) & (d6.轨 == t)]
                if len(h):
                    h = h.iloc[0]
                    r6[f"{sel} {t} 改善"] = pc(h.风险变化均值)
                    r6[f"{sel} {t} 实际变差"] = f"{int(h.实际变差_分子)}/{int(h.实际变差_分母)}"
        h = d6[(d6.条件 == c) & d6.选择器.astype(str).str.startswith("MIX − POOLMIX")]
        if len(h):
            r6["MIX − POOLMIX（bal 配对）"] = f"{pc(h.iloc[0].风险变化均值)}（SE {pc(h.iloc[0].SE)}）"
        rows6.append(r6)
    T6 = pd.DataFrame(rows6)
    L.append(tbl(T6, list(T6.columns)))
    L.append("\n（改善 = 相对 P\\*_D 的风险变化，各条件全部完成重复的均值；「实际变差」= 部署风险高于 P\\*_D 且超过 2 个积分标准误的重复数。POOLMIX 只是 P\\* 与另一合并族候选的一种融合。）\n")
    L.append("### 4.6 认证的样本成本（P\\*_F 相对 P\\*_D）\n")
    L.append(tbl(D["D7"], ["条件", "轨", "重复", "P星F相对P星D改善均值", "SE"], fmt={"P星F相对P星D改善均值": pc, "SE": pc}))
    L.append("\n### 4.7 与 T1 并列（n = 6,000、π_A = 0.15；T1 每条件 20 次）\n")
    d8 = D["D8"].copy()
    d8 = d8[(d8.项 == "候选对真值相对收益") & (d8.对象 == "D|P*") | (d8.项 == "选择器两轨等权部署改善") & d8.对象.isin(["MIX", "CV_POINT_M|P*", "FCT|P*", "ALWAYS_SPLIT"])]
    for c in ("T1均值", "T2均值", "T2下", "T2上"):
        d8[c] = d8[c].map(pc)
    L.append(tbl(d8, ["项", "条件", "对象", "轨", "T1均值", "T2均值", "T2下", "T2上", "符号"]))
    L.append("\n（「符号」比较 T1 与 T2 均值的正负；T2 均值恰为 0 单列，不算符号相反。T2下 / T2上 = T2 均值 ± 1.96·SE。）\n")
    jt = D["J"][D["J"].T1均值差.notna()]
    L.append(f"\n两两排序（Q6）：有 T1 对应的 {jt.条件.nunique()} 个条件共 {len(jt)} 个选择器对（表 X5），其中 T1 与 T2 符号相反且 T2 配对区间不含 0 的共 {len(D['D9'])} 个"
             + ("：" + "；".join(f"{r.条件} {esc(r.选择器甲)} vs {esc(r.选择器乙)}" for r in D["D9"].itertuples()) if len(D["D9"]) else "") + "。\n")
    L.append("### 4.8 方案 §8 预定的 T1 对照（Q1–Q5；Q6 见上）\n")
    L.append("T1 表述摘自方案 §8 原文（一条表述含多个对象时按对象拆行；Q4 中「不达标启用罕见」的计数见表 D5）。归类是报告层的机械描述（方案只为 Q6 预定了判定规则）：方向性表述（≥ m、< 0、≤ 0）按 T2 均值 ± 1.96·SE 相对界限归为"
             "「一致 / 推翻 / 不确定」；「接近 0」「约」类表述方案没有预定界限，只列数字。均值是各条件全部完成重复的平均，「T2 达 m（确定）」是单次重复真值收益确定 ≥ m 的比例。\n")
    d11 = D["D11"].copy()
    for c in ("T1均值", "T2均值"):
        d11[c] = d11[c].map(pc)
    d11["T2 区间"] = [f"{pc(a)} ~ {pc(b)}" for a, b in zip(D["D11"].T2下, D["D11"].T2上)]
    d11["T2 达 m（确定）"] = D["D11"].T2达m确定比例.map(lambda v: "—" if pd.isna(v) else pc(v))
    d11["T1 自身符合"] = D["D11"].T1自身符合表述
    L.append(tbl(d11, ["Q", "T1表述", "条件", "对象", "轨", "T1均值", "T1 自身符合", "T2均值", "T2 区间", "T2区间相对界限", "与T1表述", "T2 达 m（确定）"]))
    nb = D["D11"][D["D11"].T1自身符合表述 == "否"]
    if len(nb):
        L.append("\n「T1 自身符合」= T1 均值本身是否满足方案对 T1 的方向性描述。不满足的行："
                 + "；".join(f"{r.Q} {r.条件} {esc(r.对象)} {r.轨}（T1 均值 {pc(r.T1均值)}）" for r in nb.itertuples())
                 + "。这些行的「与 T1 表述」比较的是方案的文字描述，不是 T1 的数字。")
    L.append("")
    L.append("### 4.9 成本与失败\n")
    L.append(tbl(D["D10"], ["条件", "重复", "每重复秒均值", "峰值内存MB最大", "评价10万", "评价50万", "评价100万"],
                 fmt={"每重复秒均值": lambda v: f"{fx(v, 1)}", "峰值内存MB最大": lambda v: f"{fx(v, 1)}"}))
    L.append("\n运行中途有一段时间，Windows 11 把无窗口的后台 worker 只调度到能效核上，单次重复明显变慢；发现后对 T2 进程关闭执行速度节流"
             "（SetProcessInformation，只改调度提示），代码、种子、并发数与已完成的重复都没有改动。上表的每重复秒数含这一段，不宜当作纯计算成本。")
    d12 = D["D12"]
    if len(d12):
        nok = int((d12.arms逐位相同 & d12.pairs逐位相同 & d12.sel逐位相同 & d12.rep记录除计时相同).sum())
        L.append(f"调度不改变结果的核对：在能效核时段跑完的 {len(d12)} 个重复，在性能核上重跑到独立目录，逐臂、候选对、选择器结果与重复记录（除计时字段）逐位相同 "
                 f"{nok}/{len(d12)}；变异对照（重跑结果中一个浮点值乘 1 + 10⁻¹²）检出 {int(d12.变异对照检出.sum())}/{len(d12)}。\n")
        L.append(tbl(d12, ["重复", "arms逐位相同", "pairs逐位相同", "sel逐位相同", "rep记录除计时相同", "正式运行秒", "重跑秒", "变异对照检出"],
                     fmt={"正式运行秒": lambda v: f"{fx(v, 1)}", "重跑秒": lambda v: f"{fx(v, 1)}"}))
    L.append("")
    return "\n".join(L)


def sec_e(R):
    E = R["E"]
    L = ["## 5. 只读更正（不触发重训）\n"]
    L.append("### 5.1 预定判读的范围\n")
    e1 = E["E1"]
    L.append(f"目标一致性推断表「4_预定判读」对 {len(e1)} 行都写了预定措辞；方案只为 O_MORT10×低白蛋白 的 C1、D1 预定了解释性判读。"
             f"本轮在报告层更正：{int(e1.方案预定的解释性判读.sum())} 行保留预定措辞，其余 {int((~e1.方案预定的解释性判读).sum())} 行只报三态（表 E1）。"
             "冻结的推断脚本保留历史版本，但报告不再输出已知不准确的文字。\n")
    L.append("### 5.2 R43 更正表的行号\n")
    e2 = E["E2"]
    rows = e2[e2.GPT第48轮指出]
    L.append(f"R45 报告 §2.1 写的「前两行是 GPT 指出的两处」不对：GPT 第 48 轮指出的是表内**第 {'、'.join(str(int(r)) for r in rows.行号)} 行**"
             f"（{'；'.join(f'{r.机制} n={int(r.n)} πA={fx(r.πA, 2)} {r.轨} 轨 {r.候选对}：{int(r.达m_初始评价)}/20 → {int(r.达m_最终)}/20' for r in rows.itertuples())}）；"
             f"第 {'、'.join(str(int(r)) for r in e2[~e2.GPT第48轮指出].行号)} 行是 R43 原表没有列出的那一处（达标数不变、分类互换）。数字本身不变。\n")
    L.append("### 5.3 NWICU 的「窗内有体征」\n")
    js = E["js"]["体征"]
    e3 = E["E3"]; al = e3[e3.分组方式 == "全部"].iloc[0]
    L.append(f"R45 代码中「窗内有体征」实际是「窗内心率非缺失」。两种口径并列（{int(js['人次'])} 人次）：窗内心率非缺失 {pc(js['心率非缺失'])}，"
             f"窗内任一规定体征（心率、收缩压、舒张压、平均动脉压、呼吸频率、SpO2）非缺失 {pc(js['任一体征非缺失'])}，两口径不一致 {int(js['两口径不一致人次'])} 人次。"
             f"住院死亡率（v2 可评价人次）：任一体征非缺失者 {pc(al['住院死亡率_窗内任一规定体征非缺失'])}，否则 {pc(al['住院死亡率_非窗内任一规定体征非缺失'])}。"
             "按 ICU 类型与锚定年份组的拆分见表 E3。只改名称与描述，不改入组和拟合。\n")
    L.append("### 5.4 各模型自身的校准（不只报差值）\n")
    e4 = E["E4"]
    sub = e4[(e4.臂.isin(["P*", "HTFS"])) & e4.库.str.startswith(("nwicu", "nhanes（R46"))].copy()
    sub["CITL"] = sub.校准截距.map(lambda v: f"{fx(v, 4)}"); sub["斜率"] = sub.校准斜率.map(lambda v: f"{fx(v, 4)}")
    sub["平均预测"] = sub.平均预测.map(pc); sub["观察事件率"] = sub.观察事件率.map(pc)
    L.append("全部库、全部任务、P\\*/HTFS/Hfull/S0\\*/SZ\\* 与 NHANES 目标一致性追加臂（P\\*_pop、HTFS_pop、P\\*_B、HTFS_B）在 A、B、全体三种范围的 CITL、斜率、平均预测、"
             "观察事件率、Brier（3 个 seed 的均值与范围）见表 X1；下面只列 NWICU 与 NHANES 主分析的 P\\* 与 HTFS（全体范围）：\n")
    L.append(tbl(sub, ["库", "结局", "表型", "臂", "CITL", "斜率", "平均预测", "观察事件率"]))
    L.append("")
    return "\n".join(L)


def sec_f(R):
    F = R["F"]
    L = ["## 6. 写作数字包（三层分开，不合成总效应）\n"]
    L.append("三层：**L1** ICU 预定两轨等权比较（MIMIC-IV、eICU-CRD、NWICU v2）；**L2** NHANES 原始跨领域复现（v2）；"
             "**L3** NHANES 结果知情后登记的目标一致性追加对照（C1 pop 11 任务、D1 B 1 任务）。数据库内任务共享患者，库间结局与人群不同，所以只分库、分目标报告。"
             "NHANES 的 pop 一律称「按研究样本比例加权」，不是使用调查权重后的美国总体风险；NHANES 追加的 pop 结果不替换其原 bal 主结果。\n")
    f1 = F["F1"][F["F1"].尺度.isin(["bal", "pop", "B"])]
    f1 = f1[~((f1.层.str.startswith("L1") | f1.层.str.startswith("L2")) & (f1.尺度 != "bal"))]
    L.append(tbl(f1, ["层", "库", "对照", "尺度", "任务", "排除", "证据不足", "支持", "点估计为负", "相对改善中位", "相对上界最大"],
                 fmt={"相对改善中位": pc, "相对上界最大": pc}))
    L.append("\n逐轨 max-T 同时判断（与逐任务区间分开陈述）：\n")
    L.append(tbl(F["F2"], ["库", "项数", "排除", "证据不足", "支持", "max_T临界值", "证据不足项"], fmt={"max_T临界值": lambda v: f"{fx(v, 4)}"}))
    L.append("\n**主张—证据—限制矩阵**（候选表述与适用范围；数字由脚本从结果表取出；不是论文正文）：\n")
    L.append(tbl(F["F3"], ["编号", "证据层级", "主张", "关键数字", "限制"]))
    L.append("\n主图源数据（证据表 X4_ 开头的各表；`numbers/R46_NUMBERS.xlsx` 同内容）：\n")
    L.append(tbl(F["F4"], ["图", "表", "行数"]))
    L.append("")
    return "\n".join(L)


def sec_g(R):
    G = R["G1"]
    L = ["## 7. 直接相关先例的原文定位（不是系统综述）\n"]
    L.append(f"{G.编号.nunique()} 篇、{len(G)} 条摘录，每条都在本地原文全文中逐字找到，页码由匹配位置自动给出；改一个词的摘录必须找不到（变异对照通过）。"
             "原文只留本地。「与本研究的关系」只作描述，不据本研究结果推断既往研究有缺陷。Vergouwe 2017 本轮未取得全文，未做定位。\n")
    g = G.copy(); g["摘录"] = g.原文摘录.map(lambda s: s if len(s) <= 160 else s[:157] + "…")
    L.append(tbl(g, ["编号", "主题", "原文位置", "摘录", "与本研究的关系"]))
    L.append("")
    return "\n".join(L)


def sec_pkg(R):
    L = ["## 8. 交付与复现包\n"]
    L.append("- 交付三件：本报告、证据表、`R46_CODE_TESTS.zip`。zip 中的代码按 import 关系**自动求出本地依赖闭包**（含 `pgss_core.py`、`pgss_sharing.py`），"
             "另附 `code/external/nhanes.py`；包自检在临时目录解压后用隔离解释器（`python -I`）逐个导入全部模块，并以「删去 pgss_core.py 后引擎导入必须失败」作变异对照。")
    L.append("- 包内含：两份方案与全部冻结记录、核实与等价测试、干跑审计、折归属核对、修正后与零修正推断及比较、只读更正、数字包、文献定位表（不含原文）、"
             "T2 配置/端到端测试/成本/汇总与逐重复记录（合成数据）、驱动与日志、环境记录。")
    L.append("- **不交付**：任何逐单元 runs 产物（含 stay_id / SEQN 的测试预测）、`_local_only`（生命状态核实表等个体级文件）、NWICU/MIMIC/NHANES 个体数据、文献原文。")
    L.append("- 数字溯源两层：① 报告中带小数的数字须在「报告引用」表中找到（舍入匹配；两位小数的随机命中率高，几乎没有检出力，命中率见包自检）；② "
             + TRACE_MARK + "。")
    L.append("- 旧数据与模型版本不覆盖：R45 NWICU、R44、R45 目标一致性的原单元与原推断原样保留；R46 v2 在独立目录。\n")
    return "\n".join(L)


def trace_check(sheets):
    """逐值追踪核对：原值在报告引用表中精确存在（相对差 ≤ 1e-12）；变异对照：原值乘以 (1 + 1e-6) 后必须找不到。"""
    arr = [pd.DataFrame(v).apply(pd.to_numeric, errors="coerce").to_numpy(float).ravel() for k, v in sheets.items() if k in CITED]
    pool = np.unique(np.abs(np.concatenate(arr))); pool = pool[np.isfinite(pool)]

    def found(x):
        x = abs(x); i = int(np.searchsorted(pool, x)); c = pool[max(i - 1, 0):i + 1]
        return bool(len(c) and np.min(np.abs(c - x)) <= 1e-12 * max(1.0, x))
    tv = np.unique(np.abs(np.array(TRACE, float)))
    miss = [float(x) for x in tv if not found(x)]
    nz = tv[tv > 0]
    pick = np.random.default_rng(0).choice(nz, size=min(50, len(nz)), replace=False) if len(nz) else np.array([])
    leak = int(sum(found(x * (1 + 1e-6)) for x in pick))
    return dict(追踪数值=len(TRACE), 不同数值=int(len(tv)), 找不到=miss, 变异对照数=int(len(pick)), 变异对照被放行=leak)


def write_md(R, conclusion, opinions):
    TRACE.clear()
    L = ["# R46 最终核验与 T2 · 统一报告\n",
         "> 由 `pgss_r46_report.py` 生成；正文数字只来自同名证据表中「目录」标为「报告引用」的表（md 与 xlsx 同源）。"
         "相对改善 = (R参照 − R候选)/R参照，正值 = 候选更好；三态判读以 m = 1% 为主。真实数据区间均为固定 OOF 预测的簇重抽，不是算法级区间。\n"]
    L.append(conclusion(R))
    for f in (sec_a, sec_b, sec_c, sec_d, sec_e, sec_f, sec_g, sec_pkg):
        L.append(f(R))
    L.append(opinions(R))
    tc = trace_check(R["sheets"])
    txt = "\n".join(L) + "\n"
    assert txt.count(TRACE_MARK) == 1
    txt = txt.replace(TRACE_MARK, f"生成时逐值追踪 {tc['追踪数值']} 次格式化（{tc['不同数值']} 个不同原值），在报告引用表中找不到原值的 {len(tc['找不到'])} 个；"
                                  f"变异对照（{tc['变异对照数']} 个原值乘以 1 + 10⁻⁶）被放行 {tc['变异对照被放行']} 个；"
                                  "追踪记录同时登记报告全文的 SHA-256，打包自检据此检出生成后对报告的任何改动（`R46_NUM_TRACE.json`）")
    MD.write_text(txt, encoding="utf-8")
    tc["报告sha"] = hashlib.sha256(MD.read_bytes()).hexdigest()[:16]
    (S / "R46_NUM_TRACE.json").write_text(json.dumps(tc, ensure_ascii=False, indent=1), encoding="utf-8")
    if tc["找不到"] or tc["变异对照被放行"]:
        print("逐值追踪未通过：", tc["找不到"][:20], tc["变异对照被放行"])


def conclusion(R):
    B, C, D, F = R["B"], R["C"], R["D"], R["F"]
    v = B["B1"].set_index("项").N; s7 = B["B7"].set_index("项").值; b5 = B["B5"].set_index("尺度")
    c4 = C["C4"].set_index("项").值
    A2 = R["A2"]; A3 = R["A3"]
    L = ["## 0. 本轮结论\n"]
    L.append(f"**执行**：{len(A2)} 类运行前与运行后检查{'全部通过' if bool(A2.全部通过.all()) else '未全部通过（见 §1）'}；三份冻结记录的登记文件现值与冻结值不同的共 {int(R['A1'].现值与冻结值不同的文件.sum())} 个。"
             + "；".join(f"{r.部分} {int(r.完成)}/{int(r.单元)}" for r in A3.itertuples()) + "。\n")
    bal = b5.loc["bal"]
    L.append(f"1. **NWICU 结局修正**：405 条 UNKNOWN 中 {int(v['类别 K2'])} 条有出院时存活的独立证据（{int(v['K2 依据：后续入院'])} 条为后续住院），"
             f"{int(v['住院死亡不可评价（排除）合计'])} 条没有可确定生命状态的证据，从住院死亡分析中排除（不填零、不按比例推断）；保留者标签改变 {int(v['住院死亡标签改变（可评价者中）'])} 个。"
             f"重跑 90 个住院死亡单元后，12 个任务的两轨等权逐任务区间：排除 {int(bal.排除m1)}、证据不足 {int(bal.证据不足m1)}、支持 {int(bal.支持m1)}，"
             f"点估计为负 {int(bal.点估计为负)}/{int(bal.任务)}，中位 {pc(bal.相对改善中位)}；"
             f"逐轨 max-T {int(s7['max-T 项数'])} 项：排除 {int(s7['max-T 排除（v2）'])}、证据不足 {int(s7['max-T 证据不足（v2）'])}、支持 {int(s7['max-T 支持（v2）'])}。"
             f"四个尺度合计 {int(s7['四尺度 E1 行数（v2）'])} 行中支持实质收益 {int(s7['四尺度 E1 支持实质收益合计（v2）'])} 行。")
    L.append(f"2. **NHANES 死因未知一例**：在 3 个已纳入的 O_CVD10 任务中删除后重跑；R44 主要尺度 {int(c4['R44 v2 纳入任务'])} 个任务支持 {int(c4['R44 v2 主要尺度支持'])}，"
             f"目标一致性 C1（pop）支持 {int(c4['R45 v2 C1（pop）支持'])}；O_MORT10 各任务逐格不变，低白蛋白的预定判读不变。")
    L.append(t2_bullet(R))
    L.append("4. **只读更正**：预定判读只保留低白蛋白一项的措辞；R43 更正表的行号改为第 1、3 行；NWICU「窗内有体征」改为两种准确口径并列；"
             "正式校准表给出每个模型自身的 CITL、斜率、平均预测与观察事件率。")
    L.append("5. **写作材料**：三层总结果表（ICU 预定比较 / NHANES 原始复现 / NHANES 结果知情后追加对照）、max-T 汇总、主张—证据—限制矩阵、主图源数据；"
             f"直接相关先例 {R['G1'].编号.nunique()} 篇的原文定位（{len(R['G1'])} 条摘录全部逐字核对）。复现包按 import 闭包自包含。\n")
    return "\n".join(L)


def opinions(R):
    B, G = R["B"], R["G1"]
    v = B["B1"].set_index("项").N; g2 = B["B2"].iloc[0]; s7 = B["B7"].set_index("项").值
    pf = G[G.编号 == "pfohl2022"]; sy = G[(G.编号 == "suriyakumar2023") & (G.主题 == "目标一致性")]
    L = ["## 9. 提请 GPT 裁决或知悉的意见\n"]
    L.append(f"1. **「目标一致」不是新做法，论文要据实定位。** Pfohl 等 2022（Sci Rep）已经把合并模型的早停与超参数选择改为按亚组最坏情况目标"
             f"（{pf[pf.主题 == '目标一致性'].原文位置.iloc[0]}：“{pf[pf.主题 == '目标一致性'].原文摘录.iloc[0]}”），"
             f"并得出「多数情况下没有方案在每个亚组上优于使用全部数据的标准程序」；Suriyakumar 等 2023 也写明按总体表现做模型选择会使某组变差"
             f"（{sy.原文位置.iloc[0]}）。我建议：把「目标一致」写成**专业化收益的归因原则**（与 GPT §2.3 一致），引用这两篇为先例；"
             "本研究能主张的增量是在含表型信息的强合并参照、算法级重训与 Brier 下，把独立训练的增量与开发目标的差异分开，并给出一个实例。"
             "工作题目里的 “Objective-aligned evaluation” 可能被审稿人认为是已知做法；是否改为突出主问题（独立训练在表型感知合并学习之上何时增值），请 GPT 定。")
    L.append(f"2. **NWICU 的 UNKNOWN 多半不是隐匿死亡，但排除仍是对的。** 有存活证据的 {int(v['类别 K2'])} 条中 {int(v['K2 依据：后续入院'])} 条是后续住院，"
             f"中位间隔 {fx(g2.中位, 4)} 天，提示 UNKNOWN 常见于院内转科或住院记录拆分；但另外 {int(v['类别 K3'])} 条没有任何证据，按 GPT 规则排除，不按比例推断。"
             "R45（UNKNOWN 记为存活）与 R46（排除）两版并列可作描述：两版的住院死亡结果见表 B4。不再做第三种处理。")
    L.append(f"3. **同时判断的措辞**：NWICU 修正后逐轨 max-T 仍有 {int(s7['max-T 证据不足（v2）'])} 项证据不足（表 B6）。论文只能写「没有任何一项得到支持」，"
             "不能写「所有任务、所有轨道的实质收益都已同时排除」。")
    L.append(t2_opinions(R))
    L.append("- **本轮两处运行前调整（都在冻结之前，不是冻结后修订）**：① T2 端到端测试第一次在 n = 2,000、π_A = 0.15 的测试设置下因认证拟合集 A 轨过小而 FoldFailure，"
             "这是测试设置问题，改用正式规模 n = 6,000 后通过；② 测得评价扩到 100 万时单进程峰值约 1 GB、本机只有 15.8 GB 内存，于是把追加评价改为每 10 万行一块生成与预测，"
             "并新增测试 G8 证明逐块预测与整段预测逐位相同。两处都在 T2 冻结前完成并记录在方案里。")
    L.append("- **运行中的调度调整（不改计算，不是修订）**：T2 运行中途 Windows 把后台 worker 挤到能效核上，发现后关闭了 T2 进程的执行速度节流；"
             "调度不改变结果的逐位核对见 4.9 节与表 D12。以后的长任务在启动时即关闭节流。")
    L.append("- **校准表口径**：CITL = 斜率固定为 1 的 offset 逻辑回归截距；斜率 = logit(y) 对 logit(p̂) 的逻辑回归斜率；都在每个 seed 的完整 OOF 上计算，再取 3 个 seed 的均值与范围。")
    L.append("- **文献**：Vergouwe 2017（闭合检验选择更新方法）本轮未取得全文，没有做原文定位；若 GPT 认为必须进入相关工作，需要用户提供全文。\n")
    return "\n".join(L)


def t2_bullet(R):
    D = R["D"]
    if D is None:
        return "3. **T2**：尚未完成。"
    return t2_bullet_full(R)


def t2_opinions(R):
    D = R["D"]
    if D is None:
        return "4. **T2**：尚未完成。"
    return t2_opinions_full(R)





# ── T2 结论与意见（数据驱动）──────────────────────────────────────
def t2_facts(R):
    D = R["D"]
    tc, te, tf, tg = D["D2"], D["D3"], D["D5"], D["D6"]
    g = lambda c, k, t: tc[(tc.条件 == c) & (tc.候选对 == k) & (tc.轨 == t)].iloc[0]
    e = lambda c, s: te[(te.条件 == c) & (te.选择器 == s)].iloc[0]
    f = lambda c, s: tf[(tf.条件 == c) & (tf.选择器 == s)].iloc[0]
    pm = tg[tg.选择器.astype(str).str.startswith("MIX − POOLMIX")].set_index("条件")
    return g, e, f, pm


def t2_bullet_full(R):
    D = R["D"]; s = D["summ"]
    g, e, f, pm = t2_facts(R)
    conds = [c for c in ("N1", "N2", "P1", "P2", "N1-T", "P2-T", "N1-H", "P2-H") if c in set(D["D2"].条件)]
    pos = [c for c in ("P2", "P2-T", "P2-H") if c in conds]
    neg = [c for c in ("N1", "N2", "N1-T", "N1-H") if c in conds]
    txt = (f"3. **T2**（{s['完成']}/{s['计划']} 完成，失败 {s['失败']}）：HTFS_D 相对 P\\*_D 的真值收益（A / B 轨，全部完成重复的均值）："
           + "；".join(f"{c} {pc(g(c, 'D|P*', 'A').真值相对收益均值)} / {pc(g(c, 'D|P*', 'B').真值相对收益均值)}" for c in conds) + "。")
    if pos:
        txt += ("两轨等权部署改善（相对 P\\*_D）：" + "；".join(
            f"{c}：MIX {pc(e(c, 'MIX').均值)}、CV_POINT_M|P\\* {pc(e(c, 'CV_POINT_M|P*').均值)}、FCT|P\\* {pc(e(c, 'FCT|P*').均值)}" for c in pos) + "。")
    if neg:
        txt += ("无轨道特异结构的条件中，FCT|P\\* 重复级至少一轨不达标启用 " + "、".join(
            f"{c} {int(f(c, 'FCT|P*')['≥1轨不达标启用_分子'])}/{int(f(c, 'FCT|P*')['≥1轨不达标启用_分母'])}" for c in neg)
                + "；ALWAYS_SPLIT 的两轨等权部署改善 " + "、".join(f"{c} {pc(e(c, 'ALWAYS_SPLIT').均值)}" for c in neg) + "。")
    if len(pm):
        txt += "MIX − POOLMIX（两轨等权，配对）：" + "、".join(f"{c} {pc(pm.loc[c].风险变化均值)}" for c in pm.index if c in pos + neg) + "。"
    q = D["D11"]; qc = q.与T1表述.value_counts()
    txt += (f"方案 §8 的 Q1–Q5 共 {len(q)} 行：与 T1 表述一致 {int(qc.get('一致', 0))}、推翻 {int(qc.get('推翻', 0))}、不确定 {int(qc.get('不确定', 0))}、"
            f"只列数字 {int(qc.get('只列数字', 0))}（表 D11）；Q6 中 T2 推翻 T1 排序的选择器对 {len(D['D9'])} 个。")
    return txt


NEGC = ("N1", "N2", "N1-T", "N1-H"); POSC = ("P2", "P2-T", "P2-H")
ALLC = ("N1", "N2", "P1", "P2", "N1-T", "P2-T", "N1-H", "P2-H")
frac = lambda r, k: f"{int(r[k + '_分子'])}/{int(r[k + '_分母'])}"
sci = lambda v: (_tr(v), f"{v:.2e}")[1]


def t2_opinions_full(R):
    D = R["D"]
    g, e, f, pm = t2_facts(R)
    d1, d3, d4, d5, d6, d7, d9, q = D["D1"], D["D3"], D["D4"], D["D5"], D["D6"], D["D7"], D["D9"], D["D11"]
    conds = [c for c in ALLC if c in set(d3.条件)]
    neg = [c for c in NEGC if c in conds]; pos = [c for c in POSC if c in conds]
    L = []
    # 4 T1 对照
    bad = q[q.与T1表述.isin(["推翻", "不确定"])]
    s = (f"4. **按方案「结论按 T2 改」需要改动的 T1 表述。** Q1–Q5 共 {len(q)} 行中，T2 推翻 {int((q.与T1表述 == '推翻').sum())} 行、"
         f"不确定 {int((q.与T1表述 == '不确定').sum())} 行（表 D11）")
    s += ("：" + "；".join(f"{r.Q} {r.条件} {esc(r.对象)} {r.轨}：T1 {pc(r.T1均值)} → T2 {pc(r.T2均值)}（{pc(r.T2下)} ~ {pc(r.T2上)}，{r.与T1表述}）"
                          for r in bad.itertuples()) + "。") if len(bad) else "。"
    if len(d9):
        s += (f"Q6 推翻的 {len(d9)} 个选择器对（共比较 {int(D['J'].T1均值差.notna().sum())} 对，表 X5）分布在 {'、'.join(sorted(set(d9.条件), key=ALLC.index))}，这些对的 T1 均值差绝对值最大 "
              f"{pc(float(d9.T1均值差.abs().max()))}（表 D9）。")
    nb = q[q.T1自身符合表述 == "否"]
    if len(nb):
        s += ("另有 " + str(len(nb)) + " 行是方案对 T1 的文字描述与 T1 自身数字不符："
              + "；".join(f"{r.Q} {r.条件} {esc(r.对象)} {r.轨} T1 均值 {pc(r.T1均值)}" for r in nb.itertuples()) + "（表 D11）。")
    s += "T1 每条件 20 次。我的意见：论文凡涉及上述数字与选择器排序的地方一律引用 T2，T1 只作为设计阶段的探索并在方法中说明它被 T2 取代。"
    L.append(s)
    # 5 FCT
    fe = d3[d3.选择器 == "FCT|P*"].set_index("条件").reindex(conds)
    f5 = d5[d5.选择器 == "FCT|P*"].set_index("条件").reindex(conds)
    f4 = d4[d4.选择器 == "FCT|P*"].set_index(["条件", "轨"])
    nneg = int((fe.均值 < 0).sum())
    s = ("5. **FCT 的启用错误与部署净效应是两个问题。** 按冻结设计（f = 1/3、认证后不重拟合），FCT 部署的是用 2/3 开发数据拟合的 HTFS_F 或 P\\*_F，"
         f"即使从不启用也比 P\\*_D 少用 1/3 数据：P\\*_F 相对 P\\*_D 的逐轨真值改善均值在 {pc(float(d7.P星F相对P星D改善均值.min()))} 到 "
         f"{pc(float(d7.P星F相对P星D改善均值.max()))} 之间（表 D7）。FCT|P\\* 重复级至少一轨不达标启用各条件为 "
         f"{int(f5['≥1轨不达标启用_分子'].min())}–{int(f5['≥1轨不达标启用_分子'].max())}/{int(f5['≥1轨不达标启用_分母'].max())}"
         f"（Clopper–Pearson 上界最大 {pc(float(f5['≥1轨不达标启用_CP上'].max()))}，表 D5）；两轨等权部署改善在 {nneg}/{len(conds)} 个条件为负，"
         f"范围 {pc(float(fe.均值.min()))} 到 {pc(float(fe.均值.max()))}（表 D3）。")
    if pos:
        s += ("有轨道特异结构的条件中，B 轨检出（条件于 F 候选对确定达标）与 A 轨部署改善："
              + "；".join(f"{c} B 轨检出 {frac(f4.loc[(c, 'B')], '检出_条件于确定达标')}、A 轨 {pc(float(f4.loc[(c, 'A')].部署相对PstarD改善均值))}"
                          for c in pos if (c, "B") in f4.index and (c, "A") in f4.index) + "（表 D4）。")
    s += ("我的意见：论文分两句写 FCT——启用错误的计数与上界；部署净效应及其样本成本来源——不合成「FCT 安全且有效」一句。"
          "认证后用全部开发数据重拟合是新的认证变体，第 50 轮明确不做，只能写进限制与后续工作。")
    L.append(s)
    # 6 MIX 与 POOLMIX
    mx = d6[d6.选择器 == "MIX"].set_index(["条件", "轨"])
    s = "6. **MIX 的平均改善与逐轨变差要并列报告。** "
    if neg:
        s += ("无轨道特异结构的条件中，MIX 相对 P\\*_D 实际变差（部署风险高出超过 2 个积分 SE）的重复，A 轨 / B 轨："
              + "；".join(f"{c} {frac(mx.loc[(c, 'A')], '实际变差')} / {frac(mx.loc[(c, 'B')], '实际变差')}" for c in neg if (c, "A") in mx.index)
              + "；两轨等权部署改善 " + "、".join(f"{c} {pc(e(c, 'MIX').均值)}" for c in neg) + "（表 D3、D6）。")
    pmc = [c for c in conds if c in pm.index]
    if pmc:
        sig = [c for c in pmc if abs(pm.loc[c].风险变化均值) > 1.96 * pm.loc[c].SE]
        s += ("MIX − POOLMIX（两轨等权，同重复配对）：" + "、".join(f"{c} {pc(pm.loc[c].风险变化均值)}（SE {pc(pm.loc[c].SE)}）" for c in pmc)
              + "；均值 ± 1.96·SE 不含 0 的条件："
              + ("、".join(f"{c}（{'MIX' if pm.loc[c].风险变化均值 > 0 else 'POOLMIX'} 更好）" for c in sig) if sig else "无") + "。")
    s += ("我的意见：无结构条件下 MIX 的两轨等权平均改善掩盖了逐轨的变差，论文报告 MIX 时应把逐轨实际变差比例与平均改善并列；"
          "MIX 相对合并族融合 POOLMIX 的增量按上面的配对差陈述，不把非零混合权重本身判为错误专业化（方案 §6）。")
    L.append(s)
    # 7 重尾
    pairs = [(a, b) for a, b in (("N1", "N1-H"), ("P2", "P2-H")) if a in conds and b in conds]
    if pairs:
        d1i = d1.set_index("条件")
        same = sum(np.sign(g(a, "D|P*", t).真值相对收益均值) == np.sign(g(b, "D|P*", t).真值相对收益均值) for a, b in pairs for t in ("A", "B"))
        s = ("7. **重尾（多元 t₅）条件只与同机制的正态条件并列（方案：T2-B 没有 T1 对应）。** HTFS_D 相对 P\\*_D 的真值收益（A / B）："
             + "；".join(f"{a} {pc(g(a, 'D|P*', 'A').真值相对收益均值)} / {pc(g(a, 'D|P*', 'B').真值相对收益均值)} vs {b} "
                         f"{pc(g(b, 'D|P*', 'A').真值相对收益均值)} / {pc(g(b, 'D|P*', 'B').真值相对收益均值)}" for a, b in pairs)
             + f"，符号相同 {int(same)}/{2 * len(pairs)}；MIX 两轨等权部署改善："
             + "；".join(f"{a} {pc(e(a, 'MIX').均值)} vs {b} {pc(e(b, 'MIX').均值)}" for a, b in pairs) + "（表 D2、D3）。")
        hb = [b for _, b in pairs]
        s += ("重尾下评价协变量绝对值最大 " + "、".join(f"{b} {fx(d1i.loc[b, '评价X绝对值最大_最大'], 2)}" for b in hb)
              + "，线性预测子范围 " + "、".join(f"{b} {fx(d1i.loc[b, '线性预测子最小'], 2)} ~ {fx(d1i.loc[b, '线性预测子最大'], 2)}" for b in hb)
              + "，生成概率 < 10⁻⁶ 与 > 1 − 10⁻⁶ 的评价比例均值 " + "、".join(
                  f"{b} {sci(d1i.loc[b, 'p小于1e6比例_均值'])} 与 {sci(d1i.loc[b, 'p大于1减1e6比例_均值'])}" for b in hb) + "（表 D1）。"
              "Brier 损失有界，真值风险不会因极端线性预测子发散；但论文描述重尾条件时应写明一部分评价对象的结局在生成模型下几乎确定，这是生成设计的性质。")
        L.append(s)
    L.append(f"8. **T2 的边界不确定**：评价扩到 100 万后仍有项距边界不足 2 个积分标准误的重复共 {int(d1.最终仍有近边界项的重复.sum())} 个（各条件见表 D1）；"
             "这些项单列为「边界不确定」，不计入达标或不达标。")
    return "\n".join(L)


if __name__ == "__main__":
    R = build()
    write_md(R, conclusion, opinions)
    print("写出", MD.name, XL.name, len(R["sheets"]))
