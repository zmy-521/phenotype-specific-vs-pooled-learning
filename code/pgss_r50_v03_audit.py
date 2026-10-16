# -*- coding: utf-8 -*-
r"""R50：v0.3 手稿（GPT 共识局部修订）逐数字核对。只读，不改 Word，不拟合、不模拟。

做法：
  · 从 v0.3 Word 抽出正文、摘要与六个图注的每个段落（参考文献列表除外），用同一正则取出全部数字记号；
  · REG 为每个段落按出现顺序登记每个数字的来源（冻结结果表单元格、冻结代码常量、数据定义或图号），脚本从来源取值，
    按文中的小数位格式化后与原文逐字比较；未登记或多登记的数字一律判「未登记」（保证覆盖全部数字）；
  · WORDS 核对以英文单词写出的计数主张（如 "all eight"、"three were inconclusive"）；
  · TERMS 检查统计口径用语（点态区间、max-T、Monte Carlo 均值区间、重复间变异、NHANES 10 年死亡、调查加权）；
  · 变异对照：改文中一个数字 / 删一条登记 / 改一个来源值，核对都必须判为不通过。
来源：R47 冻结数字包、R48 图源与补充材料、R49 只读补齐、R50 O5 计数、R46 证据表（NWICU UNKNOWN 核实）、R44 队列记录、冻结代码。
用法：python pgss_r50_v03_audit.py <v0.3.docx>
"""
import sys, re, json, pathlib, hashlib
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
import docx

ROOT = HERE.parent
X47 = ROOT / "29_R47_写作交接" / "R47_冻结数字包.xlsx"
X48 = ROOT / "30_R48_图稿与补充材料" / "R48_图源与补充材料.xlsx"
X49 = ROOT / "31_R49_只读补齐" / "R49_只读补齐_证据表.xlsx"
X46 = ROOT / "28_R46_最终核验与T2" / "R46_最终核验与T2_证据表.xlsx"
X44 = ROOT / "25_R44_NHANES跨领域复现" / "R44_NHANES跨领域复现_证据表.xlsx"
O5 = ROOT / "32_R50_讨论只读计数"
OUT = ROOT / "33_R50_v03数字核对"
RX = re.compile(r"(?<![A-Za-z_\d.*])[−-]?\d[\d,]*(?:\.\d+)?%?")
_C = {}


def sh(path, sheet):
    k = (str(path), sheet)
    if k not in _C:
        _C[k] = pd.read_excel(path, sheet_name=sheet)
    return _C[k]


def one(df, **kw):
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k] == v).to_numpy()
    h = df[m]
    assert len(h) == 1, (kw, len(h))
    return h.iloc[0]


def src(path):
    return {X47: "R47", X48: "R48", X49: "R49", X46: "R46", X44: "R44"}[path]


# ── 取值函数 ────────────────────────────────────────────────────
def B3(layer, db, c, sc, col):
    d = sh(X47, "B3_分层计数"); return one(d[d.层.str.startswith(layer)], 库=db, 对照=c, 尺度=sc)[col], f"R47 B3_分层计数（{layer}·{db}·{c}·{sc}）.{col}"
def B4(db, col):
    return one(sh(X47, "B4_maxT汇总"), 库=db)[col], f"R47 B4_maxT汇总（{db}）.{col}"
def B5(c, sc, col):
    return one(sh(X47, "B5_NHANES低白蛋白归因"), 对照=c, 尺度=sc)[col], f"R47 B5_NHANES低白蛋白归因（{c}·{sc}）.{col}"
def B6(t, col):
    return one(sh(X47, "B6_NHANES差距收回"), 目标=t)[col], f"R47 B6_NHANES差距收回（目标 {t}）.{col}"
def B1(db, oc, ph, col):
    return one(sh(X47, "B1_数据与任务概况"), 库=db, 结局=oc, 表型=ph)[col], f"R47 B1_数据与任务概况（{db}·{oc}·{ph}）.{col}"
def B1max(db, oc):
    d = sh(X47, "B1_数据与任务概况"); return int(d[(d.库 == db) & (d.结局 == oc)].n.max()), f"R47 B1_数据与任务概况（{db}·{oc}）各任务 n 的最大值"
def B1count(db):
    d = sh(X47, "B1_数据与任务概况"); return int(d[(d.库 == db) & d.纳入].shape[0]), f"R47 B1_数据与任务概况（{db}）纳入任务数"
def C2(cond, track, col):
    return one(sh(X47, "C2_T2候选对真值收益"), 条件=cond, 候选对="D|P*", 轨=track)[col], f"R47 C2_T2候选对真值收益（{cond}·D|P*·{track}）.{col}"
def C3(cond, sel, col="两轨等权部署改善均值"):
    return one(sh(X47, "C3_T2两轨等权部署"), 条件=cond, 选择器=sel)[col], f"R47 C3_T2两轨等权部署（{cond}·{sel}）.{col}"
def C4(cond, sel, track, col):
    return one(sh(X47, "C4_T2逐轨部署与变差"), 条件=cond, 选择器=sel, 轨=track)[col], f"R47 C4_T2逐轨部署与变差（{cond}·{sel}·{track}）.{col}"
def C6b(cond, sel, col):
    return one(sh(X47, "C6b_T2重复级启用错误"), 条件=cond, 选择器=sel)[col], f"R47 C6b_T2重复级启用错误（{cond}·{sel}）.{col}"
def C6b_fct(agg, col):
    d = sh(X47, "C6b_T2重复级启用错误"); d = d[d.选择器 == "FCT|P*"]
    return getattr(d[col], agg)(), f"R47 C6b_T2重复级启用错误（FCT|P*，8 个条件）.{col} 的{ {'min': '最小', 'max': '最大'}[agg] }"
def C5b(cond):
    return abs(one(sh(X47, "C5b_MIX与POOLMIX配对差"), 条件=cond).风险变化均值), f"R47 C5b_MIX与POOLMIX配对差（{cond}）.风险变化均值的绝对值（百分点）"
def C10(delta, sel):
    d = sh(X47, "C10_plasmode探索"); return one(d[d.表 == "选择器两轨等权部署改善"], δ=delta, 选择器=sel).均值, f"R47 C10_plasmode探索（δ={delta}·{sel}）.均值"
def C49(db, oc, ph, col):
    return one(sh(X49, "C2_表型任务"), 数据库=db, 结局=oc, 表型=ph)[col], f"R49 C2_表型任务（{db}·{oc}·{ph}）.{col}"
def C49ratio(db, oc, ph):
    r = one(sh(X49, "C2_表型任务"), 数据库=db, 结局=oc, 表型=ph); return r.表型不可判定 / r.合格队列, f"R49 C2_表型任务（{db}·{ph}）.表型不可判定 / 合格队列"
def C4_49(db, col):
    return one(sh(X49, "C4_计数口径"), 数据库=db)[col], f"R49 C4_计数口径（{db}）.{col}"
def D3(dbl, col):
    return one(sh(X49, "D3_配对差汇总_R43格式"), 数据库=dbl, 候选="HTFS", 参照="P*", 范围="全体")[col], f"R49 D3_配对差汇总_R43格式（{dbl}·HTFS−P*·全体）.{col}"
def S54(col):
    return int(sh(X48, "S54_D1_认证汇总")[col].sum()), f"R48 S54_D1_认证汇总.{col} 合计"
def S58(col):
    return one(sh(X48, "S58_C2_认证分解汇总"), 尺度="bal")[col], f"R48 S58_C2_认证分解汇总（bal）.{col}"
def panel_full(sheet, db=None):
    d = sh(X48, sheet)
    if sheet.startswith("S22"):
        return int(d[(d.分支 == "main") & (d.库 == db) & (d.臂 == "HTFS") & (d.选择 == "kA=23,kB=23")].单元数.sum()), f"R48 {sheet}（main·{db}·HTFS·kA=23,kB=23）.单元数"
    return int(d[(d.项 == "HTFS面板") & (d.Pstar == "kA=23,kB=23")].单元数.sum()), f"R48 {sheet}（NWICU·HTFS面板 kA=23,kB=23）.单元数"
def panel_units(sheet, db=None):
    d = sh(X48, sheet)
    if sheet.startswith("S22"):
        return int(d[(d.分支 == "main") & (d.库 == db) & (d.臂 == "HTFS")].单元数.sum()), f"R48 {sheet}（main·{db}·HTFS）单元数合计"
    return int(d[d.项 == "HTFS面板"].单元数.sum()), f"R48 {sheet}（NWICU·HTFS面板）单元数合计"
def O5d(db, col):
    d = pd.read_csv(O5 / "深度7汇总.csv"); return one(d, 库=db)[col], f"R50 O5 深度7汇总.csv（{db}）.{col}"
def O5h(agg):
    h = pd.read_csv(O5 / "HTFS深度分布_附带.csv"); r = h["6"] / h[[c for c in h.columns if c not in ("库", "轨")]].sum(axis=1)
    return getattr(r, agg)(), f"R50 O5 HTFS深度分布_附带.csv：各库各轨选深度 6 的比例的{ {'min': '最小', 'max': '最大'}[agg] }"
def R46B1(item):
    return int(one(sh(X46, "B1_NWICU_UNKNOWN核实"), 项=item).N), f"R46 B1_NWICU_UNKNOWN核实「{item}」"
def F2count(cond):
    d = sh(X48, "F2_ICU复现"); return int(len(d) if cond == "all" else (d.相对_P97_5 < 0).sum()), "R48 F2_ICU复现 " + ("行数" if cond == "all" else "相对_P97_5 < 0 的任务数")
def F4count(db, oc, frac):
    d = sh(X48, "F4_学习曲线"); return int(d[(d.库 == db) & (d.可画) & (d.比例 == frac) & ((d.结局 == oc) if oc else True)].shape[0]), f"R48 F4_学习曲线（{db}·{oc or '两结局'}·比例 {frac}）可画任务数"
def maxT48():
    return int(len(sh(ROOT / "23_R42_正式主实验" / "R42_正式主实验_证据表.xlsx", "C5_E1_maxT"))), "R42 证据表 C5_E1_maxT 行数（= R48 S18）"


# 冻结代码常量
def eng(name):
    import pgss_xgb_engine as E
    return getattr(E, name)
def t2(name):
    import pgss_r46_t2 as T
    return getattr(T, name)
def code_has(fname, pat):
    return bool(re.search(pat, (HERE / fname).read_text(encoding="utf-8"), flags=re.M))
def CONST(v, where):
    return v, where
def CODE(fname, pat, v, desc):
    return (v if code_has(fname, pat) else f"代码中未找到：{pat}"), f"{fname}：{desc}"


# ── 格式化 ─────────────────────────────────────────────────────
def neg(s):
    return s.replace("-", "−")
def pct(d):
    return lambda v: neg(f"{100 * float(v):.{d}f}%")
def dec(d):
    return lambda v: neg(f"{float(v):.{d}f}")
def intc(v):
    return f"{int(v):,}"
def intp(v):
    return f"{int(v)}"
def same(v):
    return str(v)
def pctint(v):
    return f"{int(round(100 * float(v)))}%"


# ── 登记：段落 → 按出现顺序的数字来源（类型：结果 / 计数 / 设计常量 / 定义 / 记号 / 图号 / 版本）──
def R(tok, kind, getter, fmt):
    return (tok, kind, getter, fmt)


def registry():
    G = {}
    G[4] = [R("36", "计数", lambda: (B1count("mimic")[0] + B1count("eicu")[0] + B1count("nwicu")[0], "R47 B1 三个 ICU 库纳入任务数之和"), intp),
            R("11", "计数", lambda: B1count("nhanes"), intp), R("200", "设计常量", lambda: (t2("REPS"), "pgss_r46_t2.REPS"), intp)]
    G[5] = [R("36", "计数", lambda: F2count("all"), intp), R("95%", "记号", lambda: CONST("95%", "点态 95% 区间（2.5/97.5 百分位，R42 推断）"), same),
            R("−0.84%", "结果", lambda: B3("L1", "mimic", "E1", "bal", "相对改善中位"), pct(2)), R("−0.39%", "结果", lambda: B3("L1", "eicu", "E1", "bal", "相对改善中位"), pct(2)),
            R("−0.96%", "结果", lambda: B3("L1", "nwicu", "E1", "bal", "相对改善中位"), pct(2)), R("3.23%", "结果", lambda: B5("E1", "B", "观察相对改善"), pct(2)),
            R("0.11%", "结果", lambda: B5("D1", "B", "观察相对改善"), pct(2)), R("95%", "记号", lambda: CONST("95%", "点态 95% 区间"), same),
            R("−0.09%", "结果", lambda: B5("D1", "B", "相对_P2_5"), pct(2)), R("0.31%", "结果", lambda: B5("D1", "B", "相对_P97_5"), pct(2)),
            R("−1.48%", "结果", lambda: C2("P1", "A", "真值相对收益均值"), pct(2)), R("1.04%", "结果", lambda: C2("P1", "B", "真值相对收益均值"), pct(2)),
            R("−0.20%", "结果", lambda: C3("P1", "ALWAYS_SPLIT"), pct(2))]
    G[15] = [R("3.1", "版本", lambda: CONST("3.1", "MIMIC-IV 版本号：本地记录依据包名，未经官方校验（13_MIMIC_EICU/MIMIC_EICU_FEASIBILITY.xlsx 0_来源与声明）"), same),
             R("2.0", "版本", lambda: CONST("2.0", "eICU-CRD 版本号：同上，依据包名"), same)]
    G[18] = [R("24", "定义", lambda: CODE("pgss_r35_cohort.py", r"interval '24 hour'", "24", "landmark = intime + 24 h"), same),
             R("24", "定义", lambda: CODE("pgss_r35_cohort.py", r"interval '24 hour'", "24", "landmark = intime + 24 h"), same),
             R("72", "定义", lambda: CODE("pgss_r35_cohort.py", r"> 3\.0", "72", "OUT_LOS72：MIMIC los > 3 天；eICU unitdischargeoffset > 4320 min"), same),
             R("72", "定义", lambda: CODE("pgss_r35_cohort.py", r"> 4320", "72", "同上"), same),
             R("18", "定义", lambda: CODE("pgss_r35_cohort.py", r"age >= 18", "18", "成人 age ≥ 18"), same),
             R("24", "定义", lambda: CODE("pgss_r35_cohort.py", r"> 1440", "24", "eICU landmark 1440 min"), same)]
    G[20] = [R("405", "计数", lambda: R46B1("UNKNOWN 人次（合格队列）"), intp), R("205", "计数", lambda: R46B1("类别 K2"), intp),
             R("200", "计数", lambda: R46B1("住院死亡不可评价（排除）合计"), intp)]
    G[21] = [R("20", "定义", lambda: (str(20) if (sh(X44, "A2_纳排流程").步骤 == "年龄≥20").any() else "未找到", "R44 A2_纳排流程「年龄≥20」"), same),
             R("1999", "定义", lambda: CODE("pgss_r44_nhanes_cohort.py", r'CYCLES = \["1999-2000"', "1999", "CYCLES 首周期"), same),
             R("2008", "定义", lambda: CODE("pgss_r44_nhanes_cohort.py", r'"2007-2008"\]', "2008", "CYCLES 末周期"), same),
             R("10", "定义", lambda: CODE("pgss_r44_nhanes_cohort.py", r"HORIZON_M = 120", "10", "HORIZON_M = 120 个月"), same),
             R("10", "定义", lambda: CODE("pgss_r44_nhanes_cohort.py", r"HORIZON_M = 120", "10", "同上"), same)]
    G[23] = [R("65", "定义", lambda: CODE("pgss_r35_qc.py", r"age >= 65", "65", "phenotypes：年龄 ≥ 65"), same),
             R("12", "定义", lambda: CODE("pgss_r35_qc.py", r"hgb < 12\.0", "12", "女 Hb < 12"), same), R("13", "定义", lambda: CODE("pgss_r35_qc.py", r"hgb < 13\.0", "13", "男 Hb < 13"), same),
             R("60", "定义", lambda: CODE("pgss_r35_qc.py", r"eg < 60", "60", "eGFR < 60"), same), R("1.73", "记号", lambda: CONST("1.73", "单位 mL/min/1.73 m²"), same),
             R("2021", "定义", lambda: CODE("pgss_r35_qc.py", r"def ckd_epi_2021", "2021", "CKD-EPI 2021"), same),
             R("3.5", "定义", lambda: CODE("pgss_r35_qc.py", r"alb < 3\.5", "3.5", "白蛋白 < 3.5"), same),
             R("30", "定义", lambda: CODE("pgss_r35_qc.py", r"bmi >= 30", "30", "BMI ≥ 30"), same),
             R("1", "记号", lambda: CONST("1", "Z = 1"), same), R("24", "定义", lambda: CONST("24", "24-hour landmark"), same)]
    G[24] = [R("23", "设计常量", lambda: (max(eng("K_GRID")), "pgss_xgb_engine.K_GRID 最大值（ICU 23 项池）"), intp),
             R("21", "计数", lambda: (len(json.loads((ROOT / "25_R44_NHANES跨领域复现" / "data" / "COHORT.json").read_text(encoding="utf-8"))["特征"]), "R44 data/COHORT.json「特征」列表长度"), intp),
             R("23", "设计常量", lambda: (max(eng("K_GRID")), "pgss_xgb_engine.K_GRID 最大值"), intp)]
    kg = lambda i: (lambda: (eng("K_GRID")[i], f"pgss_xgb_engine.K_GRID[{i}]"))
    dt = lambda i: (lambda: (sorted({g["max_depth"] for g in eng("GRID_TRACK")})[i], f"pgss_xgb_engine.GRID_TRACK 深度[{i}]"))
    G[28] = [R("5,", "设计常量", kg(0), lambda v: f"{v},"), R("8,", "设计常量", kg(1), lambda v: f"{v},"), R("12,", "设计常量", kg(2), lambda v: f"{v},"),
             R("16", "设计常量", kg(3), intp), R("23", "设计常量", kg(4), intp),
             R("2,", "设计常量", dt(0), lambda v: f"{v},"), R("3,", "设计常量", dt(1), lambda v: f"{v},"), R("4", "设计常量", dt(2), intp), R("6", "设计常量", dt(3), intp),
             R("1", "设计常量", lambda: (min({g["min_child_weight"] for g in eng("GRID_TRACK")}), "GRID_TRACK min_child_weight 最小"), intp),
             R("10", "设计常量", lambda: (max({g["min_child_weight"] for g in eng("GRID_TRACK")}), "GRID_TRACK min_child_weight 最大"), intp),
             R("7", "设计常量", lambda: (sorted({g["max_depth"] for g in eng("GRID_POOLED")} - {g["max_depth"] for g in eng("GRID_TRACK")})[0], "GRID_POOLED 相对 GRID_TRACK 多出的深度"), intp),
             R("7", "设计常量", lambda: (7, "同上"), intp),
             R("36", "计数", lambda: O5d("mimic", "深度7"), intp), R("180", "计数", lambda: O5d("mimic", "单元"), intp),
             R("19", "计数", lambda: O5d("eicu", "深度7"), intp), R("180", "计数", lambda: O5d("eicu", "单元"), intp),
             R("6", "设计常量", lambda: (max(g["max_depth"] for g in eng("GRID_TRACK")), "GRID_TRACK 最大深度"), intp),
             R("18.3%", "计数", lambda: O5h("min"), pct(1)), R("30.6%", "计数", lambda: O5h("max"), pct(1)),
             R("0.05", "设计常量", lambda: (eng("PARAM_BASE")["eta"], "PARAM_BASE.eta"), dec(2)),
             R("0.8", "设计常量", lambda: (eng("PARAM_BASE")["subsample"] if eng("PARAM_BASE")["subsample"] == eng("PARAM_BASE")["colsample_bytree"] else -1, "PARAM_BASE.subsample = colsample_bytree"), dec(1)),
             R("1", "设计常量", lambda: (eng("PARAM_BASE")["reg_lambda"], "PARAM_BASE.reg_lambda"), lambda v: f"{v:g}"),
             R("2,000", "设计常量", lambda: (eng("MAX_ROUNDS"), "MAX_ROUNDS"), intc), R("50", "设计常量", lambda: (eng("PATIENCE"), "PATIENCE"), intp)]
    G[30] = [R("2,", "记号", lambda: CONST("2,", "R_bal = (R_A + R_B)/2"), same), R("100", "记号", lambda: CONST("100", "100 × 相对改善"), same),
             R("1%", "设计常量", lambda: (t2("M"), "m = 0.01（pgss_r46_t2.M；pgss_xgb_run.M）"), pctint),
             R("0.5%", "设计常量", lambda: CODE("pgss_r42_inference.py", r"m ∈ \{0\.5%,1%,2%\}", "0.5%", "三态判定 m ∈ {0.5%, 1%, 2%}"), same),
             R("2%", "设计常量", lambda: CODE("pgss_r42_inference.py", r"m ∈ \{0\.5%,1%,2%\}", "2%", "同上"), same)]
    G[31] = [R("2,000", "设计常量", lambda: CODE("pgss_r42_inference.py", r"B_BOOT, CHUNK, ROOT_SEED = 2000", 2000, "B_BOOT"), intc),
             R("95%", "记号", lambda: CONST("95%", "点态 95% 区间"), same)]
    G[33] = [R("10", "定义", lambda: CONST("10", "10 年全因死亡（HORIZON_M = 120）"), same),
             R("11", "计数", lambda: B3("L3", "nhanes", "C1", "pop", "任务"), intp)]
    lf = lambda i: (lambda: (eng("LC_FRACS")[i], f"pgss_xgb_engine.LC_FRACS[{i}]"))
    G[35] = [R("25%", "设计常量", lf(0), pctint), R("50%", "设计常量", lf(1), pctint), R("75%", "设计常量", lf(2), pctint)]
    G[36] = [R("1", "记号", lambda: CONST("1", "(1 − m)"), same), R("0.01", "设计常量", lambda: CODE("pgss_xgb_run.py", r"(?m)^M = 0\.01", 0.01, "M"), dec(2)),
             R("0.025", "设计常量", lambda: CODE("pgss_xgb_run.py", r"ALPHA_TRACK = 0\.025", 0.025, "ALPHA_TRACK"), dec(3))]
    G[38] = [R("200", "设计常量", lambda: (t2("REPS"), "pgss_r46_t2.REPS"), intp), R("6,000", "设计常量", lambda: (t2("N_DEV"), "pgss_r46_t2.N_DEV"), intc),
             R("15%", "设计常量", lambda: (t2("PI"), "pgss_r46_t2.PI"), pctint)]
    cum = lambda i: (lambda: (int(np.cumsum(t2("STAGES"))[i]), f"pgss_r46_t2.STAGES 累计[{i}]"))
    G[39] = [R("1", "记号", lambda: CONST("1", "p*(1 − p*)"), same), R("100,000", "设计常量", cum(0), intc), R("500,000", "设计常量", cum(1), intc),
             R("1,000,000", "设计常量", cum(2), intc)]
    tg = lambda i, f: (lambda: (t2("T_GRID")[i], f"pgss_r46_t2.T_GRID[{i}]"))
    G[40] = [R("1%", "设计常量", lambda: (t2("M"), "m"), pctint), R("0,", "设计常量", tg(0, None), lambda v: f"{v:g},"), R("0.25", "设计常量", tg(1, None), dec(2)),
             R("0.50", "设计常量", tg(2, None), dec(2)), R("0.75", "设计常量", tg(3, None), dec(2)), R("1", "设计常量", tg(4, None), lambda v: f"{v:g}")]
    G[42] = [R("20", "计数", lambda: (int(sh(X47, "C10_plasmode探索").重复.unique()[0]) if sh(X47, "C10_plasmode探索").重复.nunique() == 1 else -1, "R47 C10_plasmode探索.重复（每 δ）"), intp)]
    G[46] = [R("36", "计数", lambda: F2count("all"), intp), R("11", "计数", lambda: B1count("nhanes"), intp),
             R("67,000", "计数", lambda: C4_49("MIMIC-IV", "合格单位数"), intc), R("53,539", "计数", lambda: C4_49("MIMIC-IV", "合格患者数"), intc),
             R("109,104", "计数", lambda: C4_49("eICU-CRD", "合格单位数"), intc), R("94,920", "计数", lambda: C4_49("eICU-CRD", "合格患者数"), intc),
             R("207", "计数", lambda: C4_49("eICU-CRD", "合格医院数"), intp), R("19,965", "计数", lambda: B1max("nwicu", "OUT_LOS72"), intc),
             R("19,765", "计数", lambda: B1max("nwicu", "OUT_HOSP_MORT"), intc), R("23,513", "计数", lambda: B1max("nhanes", "O_MORT10"), intc),
             R("10", "定义", lambda: CONST("10", "10 年"), same), R("23,512", "计数", lambda: B1max("nhanes", "O_CVD10"), intc)]
    G[47] = [R("18,516", "计数", lambda: C49("MIMIC-IV", "OUT_HOSP_MORT", "低白蛋白", "任务n"), intc), R("67,000", "计数", lambda: C49("MIMIC-IV", "OUT_HOSP_MORT", "低白蛋白", "合格队列"), intc),
             R("50,348", "计数", lambda: C49("eICU-CRD", "OUT_HOSP_MORT", "低白蛋白", "任务n"), intc), R("109,104", "计数", lambda: C49("eICU-CRD", "OUT_HOSP_MORT", "低白蛋白", "合格队列"), intc),
             R("72.4%", "计数", lambda: C49ratio("MIMIC-IV", "OUT_HOSP_MORT", "低白蛋白"), pct(1)), R("53.9%", "计数", lambda: C49ratio("eICU-CRD", "OUT_HOSP_MORT", "低白蛋白"), pct(1)),
             R("58.4%", "计数", lambda: C49ratio("MIMIC-IV", "OUT_HOSP_MORT", "肥胖"), pct(1)),
             R("16.7%", "计数", lambda: C49("MIMIC-IV", "OUT_HOSP_MORT", "低白蛋白", "事件率"), pct(1)), R("10.7%", "计数", lambda: C49("MIMIC-IV", "OUT_HOSP_MORT", "女性", "事件率"), pct(1))]
    G[48] = [R("10", "定义", lambda: CONST("10", "10 年心血管死亡"), same), R("1", "图号", lambda: CONST("1", "Figure 1"), same)]
    G[49] = [R("1%", "设计常量", lambda: (t2("M"), "m"), pctint)]
    G[51] = [R("2", "图号", lambda: CONST("2", "Figure 2"), same), R("−0.84%", "结果", lambda: B3("L1", "mimic", "E1", "bal", "相对改善中位"), pct(2)),
             R("−0.39%", "结果", lambda: B3("L1", "eicu", "E1", "bal", "相对改善中位"), pct(2)), R("−0.96%", "结果", lambda: B3("L1", "nwicu", "E1", "bal", "相对改善中位"), pct(2)),
             R("95%", "记号", lambda: CONST("95%", "点态 95% 区间"), same), R("36", "计数", lambda: F2count("upper<0"), intp)]
    G[52] = [R("48", "计数", maxT48, intp), R("1%", "设计常量", lambda: (t2("M"), "m"), pctint), R("22", "计数", lambda: B4("nwicu", "排除"), intp), R("24", "计数", lambda: B4("nwicu", "项数"), intp)]
    G[53] = [R("1", "计数", lambda: S54("专业化"), intp), R("712", "计数", lambda: S54("单元"), intp), R("−0.61%", "结果", lambda: S58("合计相对中位"), pct(2)), R("24", "计数", lambda: S58("任务"), intp)]
    G[54] = list(G[20])
    G[56] = [R("11", "计数", lambda: B3("L2", "nhanes", "E1", "bal", "任务"), intp), R("1%", "设计常量", lambda: (t2("M"), "m"), pctint),
             R("−0.71%", "结果", lambda: B3("L2", "nhanes", "E1", "bal", "相对改善中位"), pct(2)), R("10", "定义", lambda: CONST("10", "10 年"), same),
             R("275", "计数", lambda: B1("nhanes", "O_MORT10", "低白蛋白", "A轨n"), intp), R("21,680", "计数", lambda: B1("nhanes", "O_MORT10", "低白蛋白", "B轨n"), intc),
             R("3.23%", "结果", lambda: B5("E1", "B", "观察相对改善"), pct(2)), R("95%", "记号", lambda: CONST("95%", "点态 95% 区间"), same),
             R("2.80%", "结果", lambda: B5("E1", "B", "相对_P2_5"), pct(2)), R("3.69%", "结果", lambda: B5("E1", "B", "相对_P97_5"), pct(2)),
             R("3.03%", "结果", lambda: B5("E1", "pop", "观察相对改善"), pct(2)), R("−2.02%", "结果", lambda: B5("E1", "bal", "观察相对改善"), pct(2)),
             R("−7.36%", "结果", lambda: B5("E1", "bal", "相对_P2_5"), pct(2)), R("2.91%", "结果", lambda: B5("E1", "bal", "相对_P97_5"), pct(2))]
    G[57] = [R("0.11%", "结果", lambda: B5("D1", "B", "观察相对改善"), pct(2)), R("−0.09%", "结果", lambda: B5("D1", "B", "相对_P2_5"), pct(2)),
             R("0.31%", "结果", lambda: B5("D1", "B", "相对_P97_5"), pct(2)), R("3", "图号", lambda: CONST("3", "Figure 3"), same),
             R("0.081046", "结果", lambda: B6("B", "R_P*（两轨等权开发）"), dec(6)), R("0.078508", "结果", lambda: B6("B", "R_P*_目标"), dec(6)),
             R("0.078420", "结果", lambda: B6("B", "R_HTFS_目标"), dec(6))]
    G[58] = [R("3.03%", "结果", lambda: B5("E1", "pop", "观察相对改善"), pct(2)), R("−0.26%", "结果", lambda: B5("C1", "pop", "观察相对改善"), pct(2)),
             R("−0.55%", "结果", lambda: B5("C1", "pop", "相对_P2_5"), pct(2)), R("0.04%", "结果", lambda: B5("C1", "pop", "相对_P97_5"), pct(2)),
             R("0.079513", "结果", lambda: B6("pop", "R_P*_目标"), dec(6)), R("0.079718", "结果", lambda: B6("pop", "R_HTFS_目标"), dec(6)),
             R("1%", "设计常量", lambda: (t2("M"), "m"), pctint)]
    G[59] = [R("11", "计数", lambda: B3("L3", "nhanes", "C1", "pop", "任务"), intp), R("1%", "设计常量", lambda: (t2("M"), "m"), pctint)]
    G[60] = [R("−0.0040", "结果", lambda: D3("MIMIC-IV + eICU-CRD（= R43 B3）", "AUROC差中位"), dec(4)), R("−0.0062", "结果", lambda: D3("NWICU（v2）", "AUROC差中位"), dec(4)),
             R("−0.0013", "结果", lambda: D3("NHANES（v2）", "AUROC差中位"), dec(4)), R("−0.0064", "结果", lambda: D3("MIMIC-IV + eICU-CRD（= R43 B3）", "AUPRC差中位"), dec(4)),
             R("−0.0086", "结果", lambda: D3("NWICU（v2）", "AUPRC差中位"), dec(4)), R("−0.0033", "结果", lambda: D3("NHANES（v2）", "AUPRC差中位"), dec(4))]
    G[62] = [R("4", "图号", lambda: CONST("4", "Figure 4"), same), R("12", "计数", lambda: (min(F4count("mimic", None, f)[0] for f in (0.25, 0.5, 0.75, 1.0)), "R48 F4_学习曲线 MIMIC 各比例可画任务数的最小值"), intp),
             R("25%", "设计常量", lf(0), pctint), R("50%", "设计常量", lf(1), pctint), R("75%", "设计常量", lf(2), pctint), R("100%", "记号", lambda: CONST("100%", "完整开发集"), same)]
    G[63] = [R("25%", "设计常量", lf(0), pctint), R("50%", "设计常量", lf(1), pctint), R("75%", "设计常量", lf(2), pctint),
             R("12", "计数", lambda: F4count("eicu", None, 1.0), intp), R("100%", "记号", lambda: CONST("100%", "完整开发集"), same)]
    G[65] = [R("200", "设计常量", lambda: (t2("REPS"), "REPS"), intp), R("6,000", "设计常量", lambda: (t2("N_DEV"), "N_DEV"), intc),
             R("0.15", "设计常量", lambda: (t2("PI"), "PI"), dec(2)), R("5", "图号", lambda: CONST("5", "Figure 5"), same)]
    G[66] = [R("−3.72%", "结果", lambda: C2("N1", "A", "真值相对收益均值"), pct(2)), R("0.05%", "结果", lambda: C2("N1", "B", "真值相对收益均值"), pct(2)),
             R("−1.84%", "结果", lambda: C3("N1", "ALWAYS_SPLIT"), pct(2)), R("−3.08%", "结果", lambda: C2("N2", "A", "真值相对收益均值"), pct(2)),
             R("−0.04%", "结果", lambda: C2("N2", "B", "真值相对收益均值"), pct(2))]
    G[67] = [R("−1.48%", "结果", lambda: C2("P1", "A", "真值相对收益均值"), pct(2)), R("1.04%", "结果", lambda: C2("P1", "B", "真值相对收益均值"), pct(2)),
             R("0.96%", "结果", lambda: C2("P1", "B", "下"), pct(2)), R("1.11%", "结果", lambda: C2("P1", "B", "上"), pct(2)), R("1%", "设计常量", lambda: (t2("M"), "m"), pctint),
             R("−0.20%", "结果", lambda: C3("P1", "ALWAYS_SPLIT"), pct(2)), R("−0.35%", "结果", lambda: C3("P1", "ALWAYS_SPLIT", "下"), pct(2)),
             R("−0.05%", "结果", lambda: C3("P1", "ALWAYS_SPLIT", "上"), pct(2))]
    G[68] = [R("1.93%", "结果", lambda: C2("P2", "A", "真值相对收益均值"), pct(2)), R("5.05%", "结果", lambda: C2("P2", "B", "真值相对收益均值"), pct(2)),
             R("3.15%", "结果", lambda: C3("P2", "ALWAYS_SPLIT"), pct(2)), R("3.57%", "结果", lambda: C3("P2-T", "ALWAYS_SPLIT"), pct(2)),
             R("3.21%", "结果", lambda: C3("P2-H", "ALWAYS_SPLIT"), pct(2)), R("49", "计数", lambda: C4("P2", "ALWAYS_SPLIT", "A", "部署实际变差_分子"), intp),
             R("200", "计数", lambda: C4("P2", "ALWAYS_SPLIT", "A", "部署实际变差_分母"), intp)]
    G[70] = [R("1%", "设计常量", lambda: (t2("M"), "m"), pctint), R("0", "计数", lambda: C6b_fct("min", "≥1轨不达标启用_分子"), intp),
             R("2", "计数", lambda: C6b_fct("max", "≥1轨不达标启用_分子"), intp), R("200", "计数", lambda: C6b_fct("max", "≥1轨不达标启用_分母"), intp),
             R("3.57%", "结果", lambda: C6b_fct("max", "≥1轨不达标启用_CP上"), pct(2))]
    G[71] = [R("−0.14%", "结果", lambda: C3("P2-H", "FCT|P*"), pct(2)), R("−0.44%", "结果", lambda: C3("P2-H", "FCT|P*", "下"), pct(2)),
             R("0.15%", "结果", lambda: C3("P2-H", "FCT|P*", "上"), pct(2)), R("−0.48%", "结果", lambda: C3("P2", "FCT|P*"), pct(2)),
             R("3.15%", "结果", lambda: C3("P2", "ALWAYS_SPLIT"), pct(2))]
    G[72] = [R("2.16%", "结果", lambda: C3("P2", "CV_LB_M|P*"), pct(2)), R("2.07%", "结果", lambda: C3("P2-T", "CV_LB_M|P*"), pct(2)), R("2.01%", "结果", lambda: C3("P2-H", "CV_LB_M|P*"), pct(2)),
             R("3,", "计数", lambda: C6b("P2", "CV_LB_M|P*", "≥1轨不达标启用_分子"), lambda v: f"{int(v)},"), R("4", "计数", lambda: C6b("P2-T", "CV_LB_M|P*", "≥1轨不达标启用_分子"), intp),
             R("5", "计数", lambda: C6b("P2-H", "CV_LB_M|P*", "≥1轨不达标启用_分子"), intp), R("200", "计数", lambda: C6b("P2", "CV_LB_M|P*", "≥1轨不达标启用_分母"), intp)]
    G[73] = [R("3.41%", "结果", lambda: C3("P2", "MIX"), pct(2)), R("3.68%", "结果", lambda: C3("P2-T", "MIX"), pct(2)), R("3.41%", "结果", lambda: C3("P2-H", "MIX"), pct(2)),
             R("0.08%", "结果", lambda: C3("N1", "MIX"), pct(2)), R("−0.09%", "结果", lambda: C4("N1", "MIX", "A", "部署相对PstarD改善均值"), pct(2)),
             R("64", "计数", lambda: C4("N1", "MIX", "A", "部署实际变差_分子"), intp), R("200", "计数", lambda: C4("N1", "MIX", "A", "部署实际变差_分母"), intp),
             R("0.12", "结果", lambda: C5b("N2"), lambda v: f"{100 * float(v):.2f}")]
    G[74] = [R("2.30%", "结果", lambda: C10(1.0, "ALWAYS_SPLIT"), pct(2)), R("2.38%", "结果", lambda: C10(1.0, "MIX"), pct(2)), R("−0.64%", "结果", lambda: C10(1.0, "FCT|P*"), pct(2)),
             R("−0.61%", "结果", lambda: C10(0.0, "ALWAYS_SPLIT"), pct(2)), R("0.05%", "结果", lambda: C10(0.0, "MIX"), pct(2)), R("−0.67%", "结果", lambda: C10(0.0, "FCT|P*"), pct(2))]
    G[77] = [R("36", "计数", lambda: F2count("all"), intp)]
    G[78] = [R("3.15%", "结果", lambda: C3("P2", "ALWAYS_SPLIT"), pct(2)), R("3.57%", "结果", lambda: C3("P2-T", "ALWAYS_SPLIT"), pct(2)),
             R("3.21%", "结果", lambda: C3("P2-H", "ALWAYS_SPLIT"), pct(2)), R("6", "图号", lambda: CONST("6", "Figure 6a"), same)]
    G[80] = [R("0.11%", "结果", lambda: B5("D1", "B", "观察相对改善"), pct(2)), R("1%", "设计常量", lambda: (t2("M"), "m"), pctint)]
    G[90] = [R("23", "设计常量", lambda: (max(eng("K_GRID")), "ICU 23 项池"), intp),
             R("168", "计数", lambda: panel_full("S22_C8_开发选择分布", "mimic"), intp), R("180", "计数", lambda: panel_units("S22_C8_开发选择分布", "mimic"), intp),
             R("174", "计数", lambda: panel_full("S22_C8_开发选择分布", "eicu"), intp), R("180", "计数", lambda: panel_units("S22_C8_开发选择分布", "eicu"), intp),
             R("151", "计数", lambda: panel_full("S27_9_开发选择分布"), intp), R("180", "计数", lambda: panel_units("S27_9_开发选择分布"), intp),
             R("24", "定义", lambda: CONST("24", "24-hour landmark"), same), R("72", "定义", lambda: CONST("72", "72-hour"), same)]
    G[92] = [R("6,000", "设计常量", lambda: (t2("N_DEV"), "N_DEV"), intc), R("15%", "设计常量", lambda: (t2("PI"), "PI"), pctint), R("1%", "设计常量", lambda: (t2("M"), "m"), pctint)]
    G[114] = [R("1", "图号", lambda: CONST("1", "Figure 1"), same)]
    G[115] = [R("1%", "设计常量", lambda: (t2("M"), "m"), pctint)]
    G[116] = [R("2", "图号", lambda: CONST("2", "Figure 2"), same)]
    G[117] = [R("36", "计数", lambda: F2count("all"), intp), R("95%", "记号", lambda: CONST("95%", "点态 95% 区间"), same), R("100", "记号", lambda: CONST("100", "100 ×"), same),
              R("1%", "设计常量", lambda: (t2("M"), "m"), pctint), R("72", "定义", lambda: CONST("72", "72 h"), same), R("24", "定义", lambda: CONST("24", "24 h"), same),
              R("72", "定义", lambda: CONST("72", "72 h"), same)]
    G[118] = [R("3", "图号", lambda: CONST("3", "Figure 3"), same)]
    G[119] = [R("95%", "记号", lambda: CONST("95%", "点态 95% 区间"), same)]
    G[120] = [R("4", "图号", lambda: CONST("4", "Figure 4"), same)]
    G[121] = [R("25%", "设计常量", lf(0), pctint), R("50%", "设计常量", lf(1), pctint), R("100%", "记号", lambda: CONST("100%", "完整开发集"), same)]
    G[122] = [R("5", "图号", lambda: CONST("5", "Figure 5"), same)]
    G[123] = [R("200", "设计常量", lambda: (t2("REPS"), "REPS"), intp), R("1.96", "记号", lambda: CONST("1.96", "均值 ± 1.96 Monte Carlo SE（pgss_r48_sources.Z）"), same),
              R("5", "记号", lambda: CONST("5", "第 5 百分位"), same), R("95", "记号", lambda: CONST("95", "第 95 百分位"), same),
              R("6,000", "设计常量", lambda: (t2("N_DEV"), "N_DEV"), intc), R("0.15", "设计常量", lambda: (t2("PI"), "PI"), dec(2))]
    G[124] = [R("6", "图号", lambda: CONST("6", "Figure 6"), same)]
    G[125] = [R("1.96", "记号", lambda: CONST("1.96", "均值 ± 1.96 Monte Carlo SE"), same), R("200", "设计常量", lambda: (t2("REPS"), "REPS"), intp),
              R("1%", "设计常量", lambda: (t2("M"), "m"), pctint)]
    return G


# 以英文单词写出的计数主张：(段落, 原文片段, 冻结核对, 说明)
def words():
    c3 = sh(X47, "C3_T2两轨等权部署"); f = c3[c3.选择器 == "FCT|P*"]
    W = [(5, "worse than full-development pooled learning in all eight conditions", bool((f.两轨等权部署改善均值 < 0).all() and len(f) == 8), "R47 C3 FCT|P* 8 个条件点估计均 < 0"),
         (5, "mean intervals were below zero in seven", int((f.上 < 0).sum()) == 7, "R47 C3 FCT|P* 上限 < 0 的条件数 = 7"),
         (28, "one seventh of the allowed training groups", eng("ES_FOLDS") == 7, "pgss_xgb_engine.ES_FOLDS = 7"),
         (28, "three training-internal folds", eng("CAL_FOLDS") == 3, "pgss_xgb_engine.CAL_FOLDS = 3"),
         (31, "five group-separated folds under each of three frozen random seeds", code_has("pgss_r38_bench.py", r"K_OUTER = 5") and code_has("pgss_r38_bench.py", r"SEEDS = \[20261003, 20261004, 20261005\]"), "pgss_r38_bench.K_OUTER = 5；SEEDS 3 个"),
         (36, "one third of groups formed a certification sample", code_has("pgss_xgb_engine.py", r"def cert_split\(g_dev, seed, f_inv=3\)"), "cert_split f_inv = 3"),
         (38, "eight prespecified conditions", len(t2("CONDS")) == 8, "pgss_r46_t2.CONDS 长度 8"),
         (42, "Three levels of track-dependent structure", sh(X47, "C10_plasmode探索").δ.nunique() == 3, "R47 C10 δ 三个水平"),
         (46, "six phenotypes for each of two outcomes", sh(X47, "B1_数据与任务概况").query("库 == 'mimic'").表型.nunique() == 6 and sh(X47, "B1_数据与任务概况").query("库 == 'mimic'").结局.nunique() == 2, "R47 B1"),
         (52, "all comparisons excluded an improvement of at least 1%", int(B4("mimic", "排除")[0]) + int(B4("eicu", "排除")[0]) == 48, "R47 B4 mimic 24 + eicu 24 排除"),
         (52, "two remained inconclusive", int(B4("nwicu", "证据不足")[0]) == 2, "R47 B4 nwicu 证据不足 2"),
         (53, "(eICU-CRD, track A)", bool(((sh(X48, "S54_D1_认证汇总").专业化 > 0) == ((sh(X48, "S54_D1_认证汇总").库 == "eicu") & (sh(X48, "S54_D1_认证汇总").轨 == "A"))).all()), "R48 S54 唯一启用为 eicu A"),
         (53, "the enabling component had a median of zero", float(S58("认证启用项相对中位")[0]) == 0.0, "R48 S58 bal 认证启用项相对中位 = 0"),
         (53, "reduced-fitting-sample component had the same median relative improvement", float(S58("样本成本项相对中位")[0]) == float(S58("合计相对中位")[0]), "R48 S58 bal 样本成本项 = 合计"),
         (56, "Eight of the 11 evaluable tasks excluded a 1% improvement and three were inconclusive", (int(B3("L2", "nhanes", "E1", "bal", "排除")[0]), int(B3("L2", "nhanes", "E1", "bal", "证据不足")[0])) == (8, 3), "R47 B3 L2 bal 排除 8、证据不足 3"),
         (59, "the additional sample-proportion-aligned comparisons excluded a 1% improvement on that scale", int(B3("L3", "nhanes", "C1", "pop", "排除")[0]) == 11, "R47 B3 L3 C1 pop 排除 11"),
         (63, "four hospital-mortality tasks and all six ICU-stay tasks", (F4count("eicu", "OUT_HOSP_MORT", 0.75)[0], F4count("eicu", "OUT_LOS72", 0.75)[0]) == (4, 6), "R48 F4 eICU 75%"),
         (63, "no task had all outer folds evaluable at the 25% or 50% targets", F4count("eicu", None, 0.25)[0] == 0 and F4count("eicu", None, 0.5)[0] == 0, "R48 F4 eICU 25%/50% 无可画任务"),
         (71, "negative in all eight conditions; mean intervals were below zero in seven", bool((f.两轨等权部署改善均值 < 0).all()) and int((f.上 < 0).sum()) == 7, "R47 C3 FCT|P*"),
         (97, "three original NHANES comparisons remained inconclusive", int(B3("L2", "nhanes", "E1", "bal", "证据不足")[0]) == 3, "R47 B3 L2 bal 证据不足 3")]
    return pd.DataFrame(W, columns=["段落", "原文片段", "一致", "冻结依据"])


# 统计口径用语检查
def terms(P):
    rows = []
    def flag(n, pat, why):
        for p in P:
            if p["n"] == n or n is None:
                for m in re.finditer(pat, p["text"], flags=re.I):
                    s = p["text"][max(0, m.start() - 70): m.end() + 70]
                    rows.append(dict(段落=p["n"], 原文片段=s.replace("\n", " "), 关注点=why))
    flag(None, r"confidence interval", "「confidence interval」：需确认不是在称呼模拟均值区间或 P5–P95")
    flag(None, r"\b(1|one)[- ]year\b", "是否把 NHANES 10 年死亡写成 1 年")
    flag(None, r"population[- ]weighted", "R48 起统一为 sample-proportion-weighted")
    flag(None, r"survey[- ]weight", "调查加权的表述")
    flag(None, r"simultaneous", "max-T 同时判断的表述")
    flag(None, r"Monte Carlo", "模拟均值区间的表述")
    flag(None, r"percentile", "重复间变异的表述")
    return pd.DataFrame(rows)


def load(docx_path):
    d = docx.Document(docx_path)
    P = [dict(n=i + 1, style=p.style.name, text=p.text) for i, p in enumerate(d.paragraphs)]
    ref0 = next(p["n"] for p in P if p["text"].strip() == "References")
    leg0 = next(p["n"] for p in P if p["text"].strip() == "Figure legends")
    return P, ref0, leg0


def tokens(text):
    t = re.sub(r"\[[\d,–\- ]+\]", " ", text)
    return [(m.group(0), m.start()) for m in RX.finditer(t)], t


def audit(P, ref0, leg0, G):
    rows, bad_cov = [], []
    for p in P:
        if ref0 <= p["n"] < leg0:
            continue
        toks, t = tokens(p["text"])
        reg = G.get(p["n"], [])
        if [x for x, _ in toks] != [r[0] for r in reg]:
            bad_cov.append(dict(段落=p["n"], 文中数字=" ".join(x for x, _ in toks), 登记=" ".join(r[0] for r in reg)))
        for i, (tok, pos) in enumerate(toks):
            if i >= len(reg):
                rows.append(dict(段落=p["n"], 原文片段=t[max(0, pos - 60): pos + 40], 数字=tok, 类型="—", 来源="未登记", 冻结值="—", 状态="未登记"))
                continue
            _, kind, getter, fmt = reg[i]                  # 文中数字与登记不同时仍取冻结值比较 ⇒ 判「不一致」并给出冻结值
            try:
                v, where = getter()
                shown = fmt(v)
                ok = shown == tok
                status = "一致" if ok else "不一致"
                if kind == "版本":
                    status = "来源不足（依据包名，需外部核验）"
            except Exception as e:
                shown, where, status = "—", f"取值失败：{type(e).__name__}: {e}"[:200], "来源不足"
            rows.append(dict(段落=p["n"], 原文片段=t[max(0, pos - 60): pos + 40], 数字=tok, 类型=kind, 来源=where, 冻结值=shown, 状态=status))
    return pd.DataFrame(rows), pd.DataFrame(bad_cov, columns=["段落", "文中数字", "登记"])


def main(docx_path):
    OUT.mkdir(parents=True, exist_ok=True)
    P, ref0, leg0 = load(docx_path)
    G = registry()
    A, cov = audit(P, ref0, leg0, G)
    W = words(); T = terms(P)
    # 变异对照
    import copy
    Pm = copy.deepcopy(P); i = next(k for k, p in enumerate(Pm) if p["n"] == 56); Pm[i]["text"] = Pm[i]["text"].replace("3.23%", "3.24%", 1)
    m1 = audit(Pm, ref0, leg0, G)[0]
    Gm = dict(G); Gm[53] = G[53][:-1]
    m2 = audit(P, ref0, leg0, Gm)
    Gm3 = dict(G); Gm3[57] = list(G[57]); tok, kind, getter, fmt = Gm3[57][4]
    Gm3[57][4] = (tok, kind, lambda: (getter()[0] + 1e-6, "变异：来源值 + 1e-6"), fmt)
    m3 = audit(P, ref0, leg0, Gm3)[0]
    mut = pd.DataFrame([dict(变异="文中 [P56] 的 3.23% 改为 3.24%", 被检出=bool((m1.状态 != "一致").sum() > (A.状态 != "一致").sum())),
                        dict(变异="删去 [P53] 最后一条登记", 被检出=bool(len(m2[1]) > len(cov))),
                        dict(变异="[P57] 0.081046 的来源值 + 1e-6", 被检出=bool((m3.状态 != "一致").sum() > (A.状态 != "一致").sum()))])
    summ = A.groupby("状态").size().rename("数字个数").reset_index()
    with pd.ExcelWriter(OUT / "R50_v03_数字核对.xlsx", engine="openpyxl") as xw:
        summ.to_excel(xw, sheet_name="0_汇总", index=False)
        A.to_excel(xw, sheet_name="1_逐数字核对", index=False)
        W.to_excel(xw, sheet_name="2_文字计数主张", index=False)
        T.to_excel(xw, sheet_name="3_统计口径用语", index=False)
        cov.to_excel(xw, sheet_name="4_覆盖缺口", index=False)
        mut.to_excel(xw, sheet_name="5_变异对照", index=False)
    rec = dict(v03_SHA16=hashlib.sha256(pathlib.Path(docx_path).read_bytes()).hexdigest()[:16], 段落=len(P), 审计数字=len(A),
               状态=summ.set_index("状态").数字个数.to_dict(), 文字主张一致=f"{int(W.一致.sum())}/{len(W)}", 覆盖缺口=len(cov),
               变异对照全部检出=bool(mut.被检出.all()), 脚本SHA16=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()[:16])
    (OUT / "R50_v03_数字核对.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 90)
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    print(A[A.状态 != "一致"].to_string()); print(W[~W.一致].to_string()); print(cov.to_string()); print(mut.to_string())
    print(T.to_string())
    return rec


if __name__ == "__main__":
    main(sys.argv[1])
