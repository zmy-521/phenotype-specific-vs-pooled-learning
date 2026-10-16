# -*- coding: utf-8 -*-
r"""R54 补充材料 v0.1 的各组表格定义（ST1–ST15、SM1、SM2）。每个函数只读冻结表、只做筛选/排序/英文化/格式化，
不计算新的统计量；返回印刷用三线表、长表与来源登记。"""
import json
import numpy as np, pandas as pd
import pgss_r54_supp_common as C
from pgss_r54_supp_common import sheet, tr, english, fnum, fpct, fci, fnd, DB, OC, PH, SC, TS, D0, FEAT

ODB = {"MIMIC-IV": 0, "eICU-CRD": 1, "NWICU": 2, "NHANES": 3}
OOC = {"OUT_HOSP_MORT": 0, "OUT_LOS72": 1, "O_MORT10": 2, "O_CVD10": 3}
OPH = {k: i for i, k in enumerate(["女性", "年龄≥65", "贫血", "eGFR降低", "低白蛋白", "肥胖"])}
OSC = {"bal": 0, "A": 1, "B": 2, "pop": 3}


def order(df, db="库", oc="结局", ph="表型", sc=None):
    d = df.copy()
    keys = []
    if db in d:
        d["_db"] = d[db].map(lambda x: ODB.get(DB.get(x, x), 9)); keys.append("_db")
    if oc in d:
        d["_oc"] = d[oc].map(lambda x: OOC.get(x, 9)); keys.append("_oc")
    if ph in d:
        d["_ph"] = d[ph].map(lambda x: OPH.get(x, 9)); keys.append("_ph")
    if sc and sc in d:
        d["_sc"] = d[sc].map(lambda x: OSC.get(x, 9)); keys.append("_sc")
    return d.sort_values(keys, kind="stable").drop(columns=keys).reset_index(drop=True)


def T(id_, group, title, df, num_cols, legend, src, src_frames, fidelity="full", long=(), remarks=""):
    vals = []
    for f in src_frames:
        for c in f.columns:
            v = pd.to_numeric(f[c], errors="coerce").to_numpy(float)
            vals.append(v[np.isfinite(v)])
    return dict(id=id_, group=group, title=title, df=df, num_cols=num_cols, legend=legend, src=src,
                src_vals=np.concatenate(vals) if vals else np.array([]), fidelity=fidelity, long=list(long), remarks=remarks)


TRI = lambda s: TS.get(s, s)


# ── SM1 ──────────────────────────────────────────────────────────
SM1_TEXT = [
    "Supplementary Methods 1. Cohort assembly, prediction landmark and outcome definitions",
    "ICU analysis units. In MIMIC-IV, the analysis unit was the first ICU stay of each hospital admission (earliest ICU entry time within the admission). "
    "In eICU-CRD, it was the first ICU unit stay of each hospital stay (unit visit number 1). In NWICU, it was the first ICU stay of each hospital admission. "
    "A patient could contribute more than one hospital admission; distinct-patient counts are reported separately (Table 1 and Supplementary Table ST1.1).",
    "Landmark and eligibility. Prediction was made 24 hours after ICU entry. A stay was eligible only if it was still in the ICU at the landmark "
    "(MIMIC-IV: ICU exit time later than ICU entry plus 24 hours; eICU-CRD: unit discharge offset greater than 1,440 minutes), the patient was aged at least 18 years "
    "with known age (eICU-CRD ages recorded as '> 89' were coded as 90), and no in-hospital death was recorded at or before the landmark. "
    "MIMIC-IV additionally required a usable death time without death–discharge timing conflict; eICU-CRD required a recorded hospital discharge status.",
    "Predictor information window. In MIMIC-IV (final frozen data version), vital signs, temperature and laboratory values were restricted to records with charting time at or after ICU entry, "
    "charting time before the landmark and storage time before the landmark; the latest charting time was used, the latest stored version was taken for the same item, "
    "and values from different items recorded at that same charting time were averaged. Baseline height and weight used the eligible record closest to ICU entry among those stored before the landmark. "
    "In eICU-CRD, observations with offsets from 0 to less than 1,440 minutes after unit admission were used; because the eICU-CRD laboratory table records only the result time, "
    "eICU-CRD laboratory values follow a result-time rather than a sampling-time window. This cross-database asymmetry is documented and was not altered.",
    "Outcomes. In-hospital death was death after the landmark during the same hospital admission (MIMIC-IV: hospital expiry flag with death time after the landmark, "
    "using the discharge time when the death time was missing; eICU-CRD: hospital discharge status 'Expired' with discharge offset greater than 1,440 minutes). "
    "Prolonged ICU stay was a total ICU length of stay greater than 72 hours measured from ICU entry (MIMIC-IV: ICU length of stay greater than 3 days; "
    "eICU-CRD: unit discharge offset greater than 4,320 minutes). It is not a new 72-hour window beginning at the landmark.",
    "NWICU and NHANES. NWICU followed the same landmark structure; in-hospital death used the corrected outcome status, in which 200 stays with unresolvable discharge status "
    "were excluded from that outcome only (Supplementary Table ST14.1). NHANES included nonpregnant adults aged at least 20 years from the 1999–2008 cycles with a completed "
    "examination and linkable public-use mortality follow-up; outcomes were 10-year all-cause and 10-year cardiovascular death from the examination date, and one death with "
    "unknown underlying cause was excluded from the cardiovascular outcome (Supplementary Table ST14.2). NHANES analyses were not survey-weighted.",
    "Counts at each step are shown in Supplementary Figure S1. For MIMIC-IV and eICU-CRD, the intermediate counts come from the frozen cohort-construction log; "
    "the first and final counts were reproduced exactly from the frozen analysis data, but the intermediate steps could not be recomputed (Supplementary Table SM1.1).",
]
C1B_EN = {("MIMIC-IV", "1→2"): ("Not in the ICU at the landmark: ICU exit at or before entry plus 24 h, or missing exit time (includes exit or death within 24 h)", "Run log; no finer breakdown"),
          ("MIMIC-IV", "2→3"): ("Age <18 years or unknown", "Run log; unknown age 0 among all analysis units"),
          ("MIMIC-IV", "3→4"): ("In-hospital death at or before the landmark", "Logically derived: the two other exclusion reasons (timing conflict, unreliable source) were 0 among all analysis units"),
          ("eICU-CRD", "1→2"): ("Not in the unit at the landmark: discharge offset ≤1,440 min or missing (includes discharge or death within 24 h)", "Run log; no finer breakdown"),
          ("eICU-CRD", "2→3"): ("Age <18 years or unknown", "Run log; breakdown within this step not recoverable"),
          ("eICU-CRD", "3→4"): ("In-hospital death before the landmark, or missing hospital discharge status", "Not separable: both reasons were nonzero and per-record reasons were not saved; not imputed")}


def sm1():
    c = sheet("R49", "C1b_各步排除原因")
    rows = [{"Database": r.数据库, "Step": r.步骤.replace("→", " → "), "Excluded": fnum(r.排除), "Reason": C1B_EN[(r.数据库, r.步骤)][0],
             "Breakdown and basis": C1B_EN[(r.数据库, r.步骤)][1]} for r in c.itertuples()]
    return T("SM1.1", "SM1", "Supplementary Table SM1.1. Exclusions at each MIMIC-IV and eICU-CRD cohort step and their documented reasons",
             pd.DataFrame(rows), ["Excluded"], ["Step numbers follow Supplementary Figure S1, panels A and B. Reasons are taken from the frozen cohort-construction code and run log; no reason was imputed."],
             [("R49", "C1b_各步排除原因", len(c))], [c[["排除"]]])


# ── ST1 ──────────────────────────────────────────────────────────
def st1():
    b = sheet("R47", "B1_数据与任务概况"); c2 = sheet("R49", "C2_表型任务"); t3 = sheet("R51", "T3_表型可判定覆盖率")
    b["数据库"] = b.库.map(DB)
    m = b.merge(c2[["数据库", "结局", "表型", "任务患者数"]], on=["数据库", "结局", "表型"], how="left") \
         .merge(t3[["数据库", "结局", "表型", "不可判定比例"]], on=["数据库", "结局", "表型"], how="left")
    m = order(m)
    rows = []
    for r in m.itertuples():
        rows.append({"Database": DB[r.库], "Outcome": OC[r.结局], "Phenotype": PH[r.表型], "Evaluable": "Yes" if r.纳入 else "No",
                     "n": fnum(r.n), "Events (%)": f"{fnum(r.事件)} ({fpct(r.事件率, 1)})", "Track A n (% of task)": f"{fnum(r.A轨n)} ({fpct(r.A轨占比, 1)})",
                     "Track A events (%)": f"{fnum(r.A轨事件)} ({fpct(r.A轨事件率, 1)})", "Track B n": fnum(r.B轨n),
                     "Track B events (%)": f"{fnum(r.B轨事件)} ({fpct(r.B轨事件率, 1)})", "Not ascertainable (%)": fpct(r.不可判定比例, 1),
                     "Distinct patients": fnum(r.任务患者数) if pd.notna(r.任务患者数) else ("= n" if r.库 == "nhanes" else "NA")})
    df = pd.DataFrame(rows)
    leg = ["n is the number of evaluable observation units in the task (ICU stays or NHANES participants) after outcome-specific exclusions; percentages of events are within the task or track. "
           "Not ascertainable: proportion of outcome-evaluable units without the phenotype-defining measurement; they were excluded from that task, not coded phenotype-negative.",
           "Distinct patients are available for MIMIC-IV and eICU-CRD from the frozen task registry; NHANES units are individual participants; NWICU per-task distinct-patient counts were not saved (NA).",
           "The NHANES 10-year cardiovascular-death × low-albumin task was not evaluable under the frozen program (single-class early-stopping hold-out within outer folds) and was not analyzed."]
    return T("ST1.1", "ST1", "Supplementary Table ST1.1. Evaluable sample size, events and phenotype-track composition for every outcome–phenotype task", df,
             ["n", "Events (%)", "Track A n (% of task)", "Track A events (%)", "Track B n", "Track B events (%)", "Not ascertainable (%)", "Distinct patients"], leg,
             [("R47", "B1_数据与任务概况", len(b)), ("R49", "C2_表型任务", len(c2)), ("R51", "T3_表型可判定覆盖率", len(t3))],
             [b[["n", "事件", "事件率", "A轨n", "A轨占比", "A轨事件", "A轨事件率", "B轨n", "B轨事件", "B轨事件率"]], c2[["任务患者数"]], t3[["不可判定比例"]]],
             long=[("ST1_tasks_full", english(order(m.drop(columns=["数据库"])), drop=["来源"]), "R47 B1 + R49 C2 + R51 T3")])


# ── ST2 ──────────────────────────────────────────────────────────
PHDEF = [("Female", "Recorded sex female", "Sex", "Yes (sex)", "Sex missing or not F/M: not ascertainable"),
         ("Age ≥65 y", "Age at least 65 years", "Age", "Yes (age)", "Age required"),
         ("Anemia", "Hemoglobin <12 g/dL in women or <13 g/dL in men", "Hemoglobin, sex", "Yes (hemoglobin, sex)", "Hemoglobin and known sex required"),
         ("Reduced eGFR", "eGFR <60 mL/min/1.73 m² by the 2021 race-free CKD-EPI creatinine equation", "Serum creatinine, age, sex", "Inputs yes; eGFR itself not a predictor",
          "Creatinine, age and known sex required"),
         ("Low albumin", "Serum albumin <3.5 g/dL", "Albumin", "No (outside the predictor pool)", "Albumin required"),
         ("Obesity", "Body mass index ≥30 kg/m² (recomputed from height and weight; ICU values restricted to 10–80 kg/m²)", "Height, weight (NHANES: recorded BMI, computed from height and weight when missing)",
          "No (outside the predictor pool)", "BMI required")]


def st2():
    rows = [dict(zip(["Phenotype (track A)", "Definition", "Defining measurements", "Defining measurement in predictor pool", "Ascertainability"], r)) for r in PHDEF]
    t1 = T("ST2.1", "ST2", "Supplementary Table ST2.1. Phenotype definitions, defining measurements and their relation to the predictor pool", pd.DataFrame(rows), [],
           ["Track A comprises phenotype-positive units (Z = 1) and track B all other ascertainable units. Values at the 24-hour ICU landmark (ICU sources) or at the NHANES examination were used. "
            "Definitions are identical across sources; NHANES BMI used the recorded examination value (computed from height and weight when missing) without the ICU range restriction."],
           [("R49", "C3_定义与landmark", len(sheet("R49", "C3_定义与landmark"))), ("R48", "S10_A3_表型与任务", len(sheet("R48", "S10_A3_表型与任务")))], [])
    import pyarrow.parquet as pq
    qc = pd.read_excel(C.ROOT / "16_R35_FIX" / "R35_DATA_QC.xlsx", sheet_name="4_最终特征池")     # 冻结运行代码读取的特征池定义
    pool = [{"性别(F指示)": "性别F"}.get(v, v) for v in qc.特征]
    cols = {}
    for k, p in (("MIMIC", C.V5 / "R38_mimic_X.parquet"), ("eICU", C.V5 / "R38_eicu_X.parquet"),
                 ("NWICU", C.ROOT / "27_R45_外部复现与机制核验" / "nwicu" / "data" / "X.parquet"), ("NHANES", C.ROOT / "25_R44_NHANES跨领域复现" / "data" / "X.parquet")):
        cols[k] = [c for c in pq.read_schema(p).names if not c.startswith("__")] + (["性别F"] if k in ("MIMIC", "eICU") else [])   # ICU 性别由 META 构造
    assert all(f in cols["MIMIC"] and f in cols["eICU"] for f in pool), "特征池成员不在 data v5"
    rows = [{"Predictor": FEAT.get(f, f), "MIMIC-IV / eICU-CRD": "Yes", "NWICU": "Yes" if f in cols["NWICU"] else "No", "NHANES": "Yes" if f in cols["NHANES"] else "No"} for f in pool]
    t2 = T("ST2.2", "ST2", "Supplementary Table ST2.2. Shared predictor pool by data source", pd.DataFrame(rows), [],
           [f"The frozen ICU learning pool comprised {len(pool)} predictors; NHANES lacked peripheral oxygen saturation and respiratory rate. "
            "Albumin, body mass index and derived eGFR were not predictor columns; they were used only for phenotype definition. Listed from the column schemas of the frozen analysis files."],
           [("data v5 / NWICU / NHANES frozen X", "column schema", len(pool))], [])
    return [t1, t2]


# ── ST3 ──────────────────────────────────────────────────────────
UNITS = [("Age", "years", "Years; eICU-CRD '> 89' coded 90", "Years", "RIDAGEYR; top-coded 85 (1999–2006) and 80 (2007–2008)", "Comparable"),
         ("Hemoglobin", "g/dL", "g/dL in both databases (verified mapping)", "g/dL and gm/dL only", "g/dL (all cycles)", "Comparable"),
         ("Serum creatinine", "mg/dL", "mg/dL in both databases", "mg/dL only", "mg/dL; 1999–2000 Selvin-calibrated; 2005–2006 not recalibrated in primary data (ST15)", "Comparable"),
         ("eGFR", "mL/min/1.73 m²", "CKD-EPI 2021 from creatinine, age, sex", "Same", "Same, reconstructed from the frozen program", "Comparable"),
         ("Albumin", "g/dL", "g/dL in both databases", "g/dL and gm/dL only", "g/dL (all cycles)", "Comparable"),
         ("Body mass index", "kg/m²", "Recomputed from height and weight; 10–80", "Height/weight converted from inches/ounces; 10–80", "Recorded BMI (computed when missing); no range restriction", "Comparable"),
         ("RDW", "%", "% in both databases", "% and percent only", "% (all cycles)", "Comparable"),
         ("Platelet count", "×10³/µL", "K/µL in both databases", "Equivalent 10³/µL notations only", "1000 cells/µL (1999–2002 label misprinted as %)", "Comparable"),
         ("Chloride", "mmol/L", "mEq/L in both databases", "mEq/L and mmol/L only", "mmol/L (all cycles)", "Comparable")]


def st3():
    t4 = sheet("R51", "T4_单位核实")
    t_units = T("ST3.1", "ST3", "Supplementary Table ST3.1. Measurement units and cross-source comparability of baseline variables",
                pd.DataFrame([dict(zip(["Variable", "Unit", "MIMIC-IV / eICU-CRD", "NWICU", "NHANES", "Comparability"], r)) for r in UNITS]), [],
                ["Basis: verified variable mapping and physiological range filters (MIMIC-IV/eICU-CRD), a full scan of NWICU laboratory unit strings, and official NHANES variable labels by cycle. "
                 "No value was converted or imputed for description."],
                [("R51", "T4_单位核实", len(t4))], [])
    L = sheet("R51", "T1_基线长表")
    L = L[~L.变量.str.startswith("Phenotype")]
    rows = []
    for v in L.变量.unique():
        r = {"Variable": v}
        for db in ("MIMIC-IV", "eICU-CRD", "NWICU", "NHANES"):
            x = L[(L.数据库 == db) & (L.变量 == v)].iloc[0]
            r[db] = f"{fnum(x.观测n)} ({fpct(x.缺失比例, 1)})" if pd.notna(x.观测n) else "NA"
        rows.append(r)
    t_obs = T("ST3.2", "ST3", "Supplementary Table ST3.2. Observed n (missing, %) of baseline variables by data source", pd.DataFrame(rows),
              ["MIMIC-IV", "eICU-CRD", "NWICU", "NHANES"],
              ["Denominators are eligible observation units (Table 1). Missing values were not imputed for description. For Female, the observed n is the number with known sex."],
              [("R51", "T1_基线长表", len(L))], [L[["观测n", "缺失比例"]]],
              long=[("ST3_baseline_long", english(L.drop(columns=["小数位"]), drop=["备注"]), "R51 T1")])
    s6 = sheet("R48", "S6_E13_NWICU_特征覆盖")
    t_nw = T("ST3.3", "ST3", "Supplementary Table ST3.3. NWICU predictor coverage and distribution in the eligible cohort",
             pd.DataFrame([{"Variable": FEAT.get(r.变量, tr(r.变量)), "Observed (%)": fpct(r.有值比例, 1), "P1": fnum(r.P1, 2), "Median": fnum(r.中位数, 2), "P99": fnum(r.P99, 2)}
                           for r in s6.itertuples()]), ["Observed (%)", "P1", "Median", "P99"],
             ["Values in native units after the frozen NWICU source adaptation (Supplementary data table ST3_NWICU_adaptation)."],
             [("R48", "S6_E13_NWICU_特征覆盖", len(s6))], [s6[["有值比例", "P1", "中位数", "P99"]]],
             long=[("ST3_NWICU_adaptation", english(sheet("R48", "S4_E4_NWICU_来源适配与QC")), "R48 S4"),
                   ("ST3_NWICU_missing_pattern", english(sheet("R48", "S5_E5_NWICU_缺失模式")), "R48 S5")])
    miss = json.loads(C.NHCOH.read_text(encoding="utf-8"))["特征缺失率"]
    mdf = pd.DataFrame({"变量": list(miss), "缺失率": list(miss.values())})
    t_nh = T("ST3.4", "ST3", "Supplementary Table ST3.4. NHANES predictor missingness in the eligible cohort (n = 23,513)",
             pd.DataFrame([{"Predictor": FEAT.get(r.变量, r.变量), "Missing (%)": fpct(r.缺失率, 2)} for r in mdf.itertuples()]), ["Missing (%)"],
             ["From the frozen NHANES cohort record written before any model was run. Predictor-level missingness for the full MIMIC-IV and eICU-CRD pools is not available as a frozen summary table (PENDING_SOURCE)."],
             [("R44 data/COHORT.json", "特征缺失率", len(mdf))], [mdf[["缺失率"]]], remarks="MIMIC-IV/eICU-CRD 全特征缺失率：PENDING_SOURCE")
    return [t_units, t_obs, t_nw, t_nh]


# ── ST4 ──────────────────────────────────────────────────────────
def st4():
    s22 = sheet("R48", "S22_C8_开发选择分布"); s27 = sheet("R48", "S27_9_开发选择分布"); s33 = sheet("R48", "S33_9_开发选择分布")
    p = s22[(s22.分支 == "main") & (s22.臂 == "P*")][["库", "选择", "单元数"]]
    p = pd.concat([p, s27[s27.项 == "P*选择"].assign(库="nwicu").rename(columns={"Pstar": "选择"})[["库", "选择", "单元数"]],
                   s33[s33.项 == "P*选择"].assign(库="nhanes").rename(columns={"Pstar": "选择"})[["库", "选择", "单元数"]]])
    piv = p.pivot_table(index="选择", columns="库", values="单元数", aggfunc="sum").reindex(columns=["mimic", "eicu", "nwicu", "nhanes"])
    rows = [{"P* selection (family | weighting | calibration)": k, **{DB[c]: fnum(v) if pd.notna(v) else "—" for c, v in r.items()}} for k, r in piv.iterrows()]
    t1 = T("ST4.1", "ST4", "Supplementary Table ST4.1. Development-selected pooled reference P* across outer-fold units", pd.DataFrame(rows),
           ["MIMIC-IV", "eICU-CRD", "NWICU", "NHANES"],
           ["Counts are outer-fold development units (3 seeds × 5 folds × evaluable tasks) in the main branch. Families: S0 (common pool), SZ (with Z), SZ-GFS (global feature selection + Z), "
            "SZ-union (union of track-specific selections + Z); weighting none or track-balanced; cal = training-only cross-fitted Platt recalibration within tracks, raw = uncalibrated. "
            "— means the option was never selected in that source."],
           [("R48", "S22_C8_开发选择分布", len(s22)), ("R48", "S27_9_开发选择分布", len(s27)), ("R48", "S33_9_开发选择分布", len(s33))], [p[["单元数"]]],
           long=[("ST4_selection_ICU_branches", english(s22), "R48 S22"), ("ST4_selection_NWICU", english(s27), "R48 S27"), ("ST4_selection_NHANES", english(s33), "R48 S33")])
    pd_ = pd.read_csv(C.R50DIR / "P星深度分布.csv", encoding="utf-8-sig"); hd = pd.read_csv(C.R50DIR / "HTFS深度分布_附带.csv", encoding="utf-8-sig")
    d7 = pd.read_csv(C.R50DIR / "深度7汇总.csv", encoding="utf-8-sig")
    rows = []
    # 按列名（深度值本身）取数；不用 itertuples——数字列名会被改成位置序号（曾因此把 3/4/6/7 标成 1–4）
    for _, r in pd_.iterrows():
        rows.append({"Database": DB[r["库"]], "Model": "P* (pooled; grid includes depth 7)", **{f"Depth {k}": fnum(r[k]) for k in pd_.columns if k != "库"}})
    for _, r in hd.iterrows():
        rows.append({"Database": DB[r["库"]], "Model": f"HTFS track {r['轨']} (grid maximum depth 6)", **{f"Depth {k}": fnum(r[k]) for k in hd.columns if k not in ("库", "轨")}})
    df = pd.DataFrame(rows).fillna("—")
    depth_cols = sorted([c for c in df.columns if c.startswith("Depth")], key=lambda c: int(c.split()[1]))
    assert set(depth_cols) == {f"Depth {k}" for k in (2, 3, 4, 6, 7)}, depth_cols
    df = df[["Database", "Model"] + depth_cols]
    n_mimic, n_eicu = int(d7[d7.库 == "mimic"].单元.iloc[0]), int(d7[d7.库 == "eicu"].单元.iloc[0])
    t2 = T("ST4.2", "ST4", "Supplementary Table ST4.2. Selected maximum tree depth of P* and of the track-specific specialists (MIMIC-IV and eICU-CRD)", df,
           depth_cols,
           [f"Counts of outer-fold units ({fnum(n_mimic)} per database). P* selected depth 7 in {fnum(d7[d7.库 == 'mimic'].深度7.iloc[0])}/{fnum(n_mimic)} MIMIC-IV and "
            f"{fnum(d7[d7.库 == 'eicu'].深度7.iloc[0])}/{fnum(n_eicu)} eICU-CRD units; depth 7 was available only to pooled candidates and depth 2 was selected by P* in no unit. "
            "Specialist counts are per track; '—' means the depth was not in that grid or never selected."],
           [("R50 O5", "P星深度分布.csv / HTFS深度分布_附带.csv / 深度7汇总.csv", len(pd_) + len(hd))], [pd_.drop(columns=["库"]), hd.drop(columns=["库", "轨"]), d7[["深度7", "单元"]]])
    return [t1, t2]


# ── ST5 ──────────────────────────────────────────────────────────
def e1_rows(frame):
    return order(frame[(frame.对照 == "E1") & (frame.尺度 == "bal")])


def st5():
    s15 = sheet("R48", "S15_B5_Brier对照含新增"); s23 = sheet("R48", "S23_2_Brier对照")
    e = pd.concat([e1_rows(s15), e1_rows(s23)], ignore_index=True)
    rows = [{"Database": DB[r.库], "Outcome": OC[r.结局], "Phenotype": PH[r.表型], "Relative improvement (%)": fpct(r.观察相对改善, 2, True),
             "Pointwise 95% interval (%)": fci(r.相对_P2_5, r.相对_P97_5), "Interval vs 0": D0[r.相对0方向], "m = 1% judgment": TRI(r["三态_m0.010"])} for _, r in e.iterrows()]
    t1 = T("ST5.1", "ST5", "Supplementary Table ST5.1. Primary E1 contrast (HTFS vs P*) on the balanced Brier scale for the 36 ICU tasks", pd.DataFrame(rows),
           ["Relative improvement (%)", "Pointwise 95% interval (%)"],
           ["Relative improvement = 100 × [R(P*) − R(HTFS)]/R(P*); positive values favour the specialists. Balanced scale = (R_A + R_B)/2. Intervals are pointwise percentile intervals from joint cluster resampling "
            "(2,000 draws) of fixed out-of-fold predictions averaged over three seeds; they do not include redevelopment variability. Excluded: upper limit <1%; Supported: lower limit >1%; otherwise Inconclusive.",
            "NWICU in-hospital death uses the corrected outcome status. Track-specific and sample-weighted scales are in the data table ST5_E1_all_scales."],
           [("R48", "S15_B5_Brier对照含新增", len(s15)), ("R48", "S23_2_Brier对照", len(s23))], [e[["观察相对改善", "相对_P2_5", "相对_P97_5"]]], fidelity="shortened for print (balanced scale; all scales in data table)",
           long=[("ST5_E1_all_scales", english(order(pd.concat([s15[s15.对照 == "E1"], s23[s23.对照 == "E1"]]), sc="尺度")), "R48 S15 + S23 (E1 rows)")])
    s18 = sheet("R48", "S18_C5_E1_maxT"); s26 = sheet("R48", "S26_5_E1_maxT")
    mt = pd.concat([order(s18), order(s26)], ignore_index=True)
    rows = [{"Database": DB[r.库], "Outcome": OC[r.结局], "Phenotype": PH[r.表型], "Track": r.轨, "θ observed": fnum(r.θ观察, 5),
             "Simultaneous band": f"{fnum(r.同时下界, 5)} to {fnum(r.同时上界, 5)}", "Judgment": TRI(r.同时判定_相对m)} for _, r in mt.iterrows()]
    t2 = T("ST5.2", "ST5", "Supplementary Table ST5.2. Track-specific max-T simultaneous judgments of the 1% specialist-gain margin (ICU sources)", pd.DataFrame(rows),
           ["θ observed", "Simultaneous band"],
           ["θ = (1 − m) R_track(P*) − R_track(HTFS) with m = 1%; Excluded when the simultaneous upper limit is below 0 (a ≥1% gain is excluded), Supported when the lower limit exceeds 0. "
            "MIMIC-IV, eICU-CRD and NWICU are separate families of 24 track-level comparisons (critical values in data table ST5_maxT_summary)."],
           [("R48", "S18_C5_E1_maxT", len(s18)), ("R48", "S26_5_E1_maxT", len(s26))], [mt[["θ观察", "同时下界", "同时上界"]]],
           long=[("ST5_maxT_ICU", english(mt), "R48 S18 + S26"), ("ST5_maxT_summary", english(sheet("R47", "B4_maxT汇总")), "R47 B4")])
    b3 = sheet("R47", "B3_分层计数")
    LAY = {"L1 ICU 预定两轨等权比较": "L1 ICU (prespecified)", "L2 NHANES 原始跨领域复现": "L2 NHANES original (prespecified)",
           "L3 NHANES 结果知情后目标一致性追加对照": "L3 NHANES objective alignment (post-result)"}
    rows = [{"Evidence level": LAY[r.层], "Database": DB[r.库], "Contrast": r.对照, "Scale": SC[r.尺度], "Tasks": fnum(r.任务), "Excluded": fnum(r.排除), "Inconclusive": fnum(r.证据不足),
             "Supported": fnum(r.支持), "Entirely <0": fnum(r.全负), "Crosses 0": fnum(r.跨零), "Entirely >0": fnum(r.全正), "Median relative improvement (%)": fpct(r.相对改善中位, 2, True),
             "Maximum upper limit (%)": fpct(r.相对上界最大, 2, True)} for r in b3.itertuples()]
    t3 = T("ST5.3", "ST5", "Supplementary Table ST5.3. Summary of pointwise judgments for the primary and objective-aligned contrasts by evidence level, database and scale", pd.DataFrame(rows),
           ["Tasks", "Excluded", "Inconclusive", "Supported", "Entirely <0", "Crosses 0", "Entirely >0", "Median relative improvement (%)", "Maximum upper limit (%)"],
           ["Counts of tasks; pointwise intervals with m = 1%. L3 rows are explanatory analyses specified after the L2 results (C1: sample-weighted objective, D1: track-B objective, single task)."],
           [("R47", "B3_分层计数", len(b3))], [b3[["任务", "排除", "证据不足", "支持", "全负", "跨零", "全正", "相对改善中位", "相对上界最大"]]])
    b4 = sheet("R47", "B4_maxT汇总")
    rows = [{"Family": DB.get(r.库, r.库), "Comparisons": fnum(r.项数), "Excluded": fnum(r.排除), "Inconclusive": fnum(r.证据不足), "Supported": fnum(r.支持),
             "max-T critical value": fnum(r.max_T临界值, 3), "Inconclusive comparisons": tr(r.证据不足项) if pd.notna(r.证据不足项) else "—"} for r in b4.itertuples()]
    t4 = T("ST5.4", "ST5", "Supplementary Table ST5.4. Track-specific max-T families: counts and critical values", pd.DataFrame(rows),
           ["Comparisons", "Excluded", "Inconclusive", "Supported", "max-T critical value"],
           ["Each database forms a separate family; families are not pooled. The NHANES family includes the single supported comparison (low albumin, track B; see ST8.2)."],
           [("R47", "B4_maxT汇总", len(b4))], [b4[["项数", "排除", "证据不足", "支持", "max_T临界值"]]])
    return [t1, t2, t3, t4]


# ── ST6 ──────────────────────────────────────────────────────────
ARMDEF = {"P*": "development-selected pooled reference", "HTFS": "specialists with track-specific feature selection", "Hfull": "specialists on the full pool",
          "HGFS": "specialists on a global feature panel", "S0*": "best S0 pooled candidate", "SZ*": "best SZ pooled candidate", "SZ-GFS*": "best SZ-GFS candidate",
          "SZ-union*": "best SZ-union candidate", "SZ*raw": "SZ*, uncalibrated", "SZ*cal": "SZ*, recalibrated", "Hfull_raw": "Hfull, uncalibrated", "Hfull_cal": "Hfull, recalibrated",
          "P*@C": "P* developed on all development data, evaluated on certification-common units", "P*_F": "P* developed on the fitting split only", "CERT": "fit–certify–test deployment"}


def summary_rows(s, metric="Brier"):
    d = s[s.指标 == metric].copy()
    d["_c"] = d.对照.map(lambda x: CONTRASTS_ORDER.index(x) if x in CONTRASTS_ORDER else 99); d["_s"] = d.尺度.map(OSC)
    d = d.sort_values(["_c", "_s"]).drop(columns=["_c", "_s"])
    return d, [{"Contrast": r.对照, "Scale": SC[r.尺度], "Tasks": fnum(r.任务), "Median relative improvement (%)": fpct(r.相对改善中位, 2, True),
                "Maximum upper limit (%)": fpct(r.相对上界最大, 2, True), "Entirely <0": fnum(r.全负), "Crosses 0": fnum(r.跨零), "Entirely >0": fnum(r.全正),
                "Excluded": fnum(r.排除m1), "Inconclusive": fnum(r.证据不足m1), "Supported": fnum(r.支持m1)} for r in d.itertuples()]


CONTRASTS_ORDER = ["E1", "E2", "E2′raw", "E2′cal", "E3", "E4", "E5", "E6", "E6′", "E7", "E1_vs_S0*", "CERT_vs_P*D", "CERT_vs_P*F", "P*F_vs_P*D"]


def st6():
    s15 = sheet("R48", "S15_B5_Brier对照含新增")
    cd = s15[["对照", "参照", "候选"]].drop_duplicates()
    cd["_c"] = cd.对照.map(lambda x: CONTRASTS_ORDER.index(x)); cd = cd.sort_values("_c")
    rows = [{"Contrast": r.对照, "Reference": r.参照, "Candidate": r.候选, "Reference description": ARMDEF.get(r.参照, ""), "Candidate description": ARMDEF.get(r.候选, "")} for r in cd.itertuples()]
    t0 = T("ST6.1", "ST6", "Supplementary Table ST6.1. Definitions of the prespecified contrasts", pd.DataFrame(rows), [],
           ["Each contrast compares a candidate with a reference on the same units; positive relative improvements favour the candidate. E1 is the primary contrast. "
            "Certification contrasts (CERT, P*_F) were evaluated in MIMIC-IV and eICU-CRD only."],
           [("R48", "S15_B5_Brier对照含新增", len(s15))], [])
    s16 = sheet("R48", "S16_B7_对照方向汇总"); s25 = sheet("R48", "S25_4_对照方向汇总")
    d16, r16 = summary_rows(s16); d25, r25 = summary_rows(s25)
    nums = ["Tasks", "Median relative improvement (%)", "Maximum upper limit (%)", "Entirely <0", "Crosses 0", "Entirely >0", "Excluded", "Inconclusive", "Supported"]
    leg = ["Counts of tasks by interval direction relative to 0 and by the m = 1% three-state judgment; medians and maxima are across tasks. Pointwise intervals of fixed out-of-fold predictions. "
           "Log-loss summaries and every task-level row (all scales and margins 0.5%, 1% and 2%) are in the data tables."]
    t1 = T("ST6.2", "ST6", "Supplementary Table ST6.2. All prespecified Brier contrasts in MIMIC-IV and eICU-CRD (24 tasks): direction and three-state counts", pd.DataFrame(r16), nums, leg,
           [("R48", "S16_B7_对照方向汇总", len(s16))], [d16[["任务", "相对改善中位", "相对上界最大", "全负", "跨零", "全正", "排除m1", "证据不足m1", "支持m1"]]],
           fidelity="shortened for print (Brier rows; log-loss rows in data table)",
           long=[("ST6_ICU_summary_all", english(s16), "R48 S16"), ("ST6_ICU_Brier_all_rows", english(order(s15, sc="尺度")), "R48 S15"),
                 ("ST6_ICU_logloss_all_rows", english(order(sheet("R48", "S19_B6_logloss对照"), sc="尺度")), "R48 S19"),
                 ("ST6_ICU_maxT_logloss", english(order(sheet("R48", "S20_B8_maxT_logloss"))), "R48 S20"),
                 ("ST6_eICU_t_sensitivity", english(order(sheet("R48", "S21_C7_eICU_t敏感性"))), "R48 S21"),
                 ("ST6_ICU_magnitude_bal", english(sheet("R48", "S17_C4_对照幅度汇总_bal")), "R48 S17")])
    t2 = T("ST6.3", "ST6", "Supplementary Table ST6.3. All prespecified Brier contrasts in NWICU (12 tasks): direction and three-state counts", pd.DataFrame(r25), nums, leg,
           [("R48", "S25_4_对照方向汇总", len(s25))], [d25[["任务", "相对改善中位", "相对上界最大", "全负", "跨零", "全正", "排除m1", "证据不足m1", "支持m1"]]],
           fidelity="shortened for print (Brier rows; log-loss rows in data table)",
           long=[("ST6_NWICU_summary_all", english(s25), "R48 S25"), ("ST6_NWICU_Brier_all_rows", english(order(sheet("R48", "S23_2_Brier对照"), sc="尺度")), "R48 S23"),
                 ("ST6_NWICU_logloss_all_rows", english(order(sheet("R48", "S24_3_logloss对照"), sc="尺度")), "R48 S24")])
    return [t0, t1, t2]


# ── ST7 ──────────────────────────────────────────────────────────
def st7():
    s29 = sheet("R48", "S29_2_Brier对照")
    e = order(s29[s29.对照 == "E1"], sc="尺度")
    rows = [{"Outcome": OC[r.结局], "Phenotype": PH[r.表型], "Scale": SC[r.尺度], "Relative improvement (%)": fpct(r.观察相对改善, 2, True),
             "Pointwise 95% interval (%)": fci(r.相对_P2_5, r.相对_P97_5), "Interval vs 0": D0[r.相对0方向], "m = 1% judgment": TRI(r["三态_m0.010"])} for _, r in e.iterrows()]
    t1 = T("ST7.1", "ST7", "Supplementary Table ST7.1. NHANES primary E1 contrast (HTFS vs P*) on all four scales for the 11 evaluable tasks", pd.DataFrame(rows),
           ["Relative improvement (%)", "Pointwise 95% interval (%)"],
           ["Balanced is the primary scale. Sample-weighted risk weights records equally within the analysed task; it is not NHANES survey weighting. "
            "The 10-year cardiovascular-death analyses use the corrected cause-of-death status. The cardiovascular-death × low-albumin task was not evaluable."],
           [("R48", "S29_2_Brier对照", len(s29))], [e[["观察相对改善", "相对_P2_5", "相对_P97_5"]]], fidelity="shortened for print (E1 rows; all contrasts in data table)",
           long=[("ST7_NHANES_Brier_all_rows", english(order(s29, sc="尺度")), "R48 S29"), ("ST7_NHANES_logloss_all_rows", english(order(sheet("R48", "S30_3_logloss对照"), sc="尺度")), "R48 S30"),
                 ("ST7_NHANES_summary", english(sheet("R48", "S31_4_对照方向汇总")), "R48 S31")])
    s32 = sheet("R48", "S32_5_E1_maxT"); mt = order(s32)
    rows = [{"Outcome": OC[r.结局], "Phenotype": PH[r.表型], "Track": r.轨, "θ observed": fnum(r.θ观察, 5),
             "Simultaneous band": f"{fnum(r.同时下界, 5)} to {fnum(r.同时上界, 5)}", "Judgment": TRI(r.同时判定_相对m)} for _, r in mt.iterrows()]
    t2 = T("ST7.2", "ST7", "Supplementary Table ST7.2. NHANES track-specific max-T simultaneous judgments (22 comparisons)", pd.DataFrame(rows), ["θ observed", "Simultaneous band"],
           ["Definitions as in Supplementary Table ST5.2; one family of 22 track-level comparisons."], [("R48", "S32_5_E1_maxT", len(s32))], [mt[["θ观察", "同时下界", "同时上界"]]])
    return [t1, t2]


# ── ST8 ──────────────────────────────────────────────────────────
B5LAYER = {"L3 结果知情后追加对照（运行前登记）": "L3 post-result comparison (registered before computation)", "L3 追加对照的分解比较（描述）": "L3 decomposition (descriptive)",
           "L2 原始跨领域复现（预定）": "L2 original comparison (prespecified)"}


def st8():
    s39 = sheet("R48", "S39_4_预定判读")
    o = order(s39); o = o.assign(_k=o.对照.map({"C1": 0, "D1": 1})).sort_values("_k", kind="stable").drop(columns="_k").reset_index(drop=True)
    rows = [{"Contrast": r.对照, "Outcome": OC[r.结局], "Phenotype": PH[r.表型], "Judgment scale": SC[r.判读尺度], "Relative improvement (%)": fpct(r.观察相对改善, 2, True),
             "Pointwise 95% interval (%)": fci(r.相对_P2_5, r.相对_P97_5), "m = 1% judgment": TRI(r.三态_m1)} for _, r in o.iterrows()]
    t1 = T("ST8.1", "ST8", "Supplementary Table ST8.1. NHANES objective-alignment analysis: prespecified judgments", pd.DataFrame(rows), ["Relative improvement (%)", "Pointwise 95% interval (%)"],
           ["Specified after the original NHANES results were inspected and registered before computation; explanatory, not confirmatory. "
            "C1: HTFS_pop vs P*_pop, both developed for sample-weighted risk (11 tasks). D1: HTFS_B vs P*_B, both developed for track-B risk (low-albumin task only). "
            "Specialist hyperparameters and panels were frozen; only their calibration choice followed the objective."],
           [("R48", "S39_4_预定判读", len(s39))], [o[["观察相对改善", "相对_P2_5", "相对_P97_5"]]],
           long=[("ST8_alignment_Brier_all", english(order(sheet("R48", "S37_2_Brier对照"), sc="尺度")), "R48 S37"),
                 ("ST8_alignment_logloss_all", english(order(sheet("R48", "S38_3_logloss对照"), sc="尺度")), "R48 S38"),
                 ("ST8_alignment_selection", english(sheet("R48", "S42_9_开发选择分布")), "R48 S42")])
    b5 = sheet("R47", "B5_NHANES低白蛋白归因")
    rows = [{"Evidence layer": B5LAYER[r.证据层], "Contrast": r.对照, "Reference": r.参照, "Candidate": r.候选, "Scale": SC[r.尺度], "Objective-matched scale": "Yes" if r.与目标一致的尺度 else "No",
             "Reference risk": fnum(r.R参照, 4), "Relative improvement (%)": fpct(r.观察相对改善, 2, True), "Pointwise 95% interval (%)": fci(r.相对_P2_5, r.相对_P97_5),
             "m = 1% judgment": TRI(r.三态_m1)} for r in b5.itertuples()]
    t2 = T("ST8.2", "ST8", "Supplementary Table ST8.2. Low-albumin 10-year all-cause-death task: original, objective-aligned and decomposition contrasts", pd.DataFrame(rows),
           ["Reference risk", "Relative improvement (%)", "Pointwise 95% interval (%)"],
           ["C2/C3 and D2/D3 decompose C1 and D1 into the change of pooled development objective and the remaining specialist difference; descriptive only. "
            "They do not separate early stopping, hyperparameters, feature panel and calibration."],
           [("R47", "B5_NHANES低白蛋白归因", len(b5))], [b5[["R参照", "观察相对改善", "相对_P2_5", "相对_P97_5"]]])
    s40 = sheet("R48", "S40_5_C1_maxT"); m40 = order(s40)
    t3 = T("ST8.3", "ST8", "Supplementary Table ST8.3. Max-T simultaneous judgments for C1 on the sample-weighted scale (11 tasks)",
           pd.DataFrame([{"Outcome": OC[r.结局], "Phenotype": PH[r.表型], "θ observed": fnum(r.θ观察, 5), "Simultaneous band": f"{fnum(r.同时下界, 5)} to {fnum(r.同时上界, 5)}",
                          "Judgment": TRI(r.同时判定_相对m)} for _, r in m40.iterrows()]), ["θ observed", "Simultaneous band"],
           ["θ = (1 − m) R_pop(P*_pop) − R_pop(HTFS_pop), m = 1%."], [("R48", "S40_5_C1_maxT", len(s40))], [m40[["θ观察", "同时下界", "同时上界"]]])
    s41 = sheet("R48", "S41_6_差距收回"); m41 = order(s41)
    t4 = T("ST8.4", "ST8", "Supplementary Table ST8.4. Recovery of the original pooled–specialist gap by objective-aligned pooled development",
           pd.DataFrame([{"Objective": {"pop": "Sample-weighted", "B": "Track B"}.get(r.目标, r.目标), "Outcome": OC[r.结局], "Phenotype": PH[r.表型], "Scale": SC[r.尺度],
                          "R(P*)": fnum(r.R_Pstar, 5), "R(P*_objective)": fnum(r.R_Pstar_q, 5), "R(HTFS_objective)": fnum(r.R_HTFS_q, 5), "R(HTFS)": fnum(r.R_HTFS, 5),
                          "Proportion recovered": fnum(r.收回比例_相对v10差距, 3)} for _, r in m41.iterrows()]),
           ["R(P*)", "R(P*_objective)", "R(HTFS_objective)", "R(HTFS)", "Proportion recovered"],
           ["Proportion recovered = [R(P*) − R(P*_objective)] / [R(P*) − R(HTFS)]; descriptive. Values above 1 mean the aligned pooled program improved beyond the original specialist advantage."],
           [("R48", "S41_6_差距收回", len(s41))], [s41[["R_Pstar", "R_Pstar_q", "R_HTFS_q", "R_HTFS", "收回比例_相对v10差距"]]])
    return [t1, t2, t3, t4]


# ── ST9 ──────────────────────────────────────────────────────────
def st9():
    d3 = sheet("R49", "D3_配对差汇总_R43格式")
    h = d3[(d3.候选 == "HTFS") & (d3.参照 == "P*") & (d3.范围 != "两轨等权")].copy()      # 两轨等权尺度上 AUROC/AUPRC 无定义（来源为空）
    h["_db"] = h.数据库.map(lambda x: {"MIMIC-IV + eICU-CRD": 0, "NWICU": 2, "NHANES": 3}.get(tr(x), 9)); h["_s"] = h.范围.map({"全体": 0, "两轨等权": 1, "A": 2, "B": 3}); h = h.sort_values(["_db", "_s"])
    rows = [{"Database": tr(r.数据库), "Scope": SC.get(r.范围, r.范围), "Task × seed units": fnum(r.任务乘seed),
             "Median ΔAUROC": fnum(r.AUROC差中位, 4), "ΔAUROC > 0": fnum(r.AUROC差为正), "Median ΔAUPRC": fnum(r.AUPRC差中位, 4), "ΔAUPRC > 0": fnum(r.AUPRC差为正),
             "Median Δintercept": fnum(r.校准截距差中位, 4), "Median Δslope": fnum(r.校准斜率差中位, 4), "Δslope > 0": fnum(r.校准斜率差为正)} for r in h.itertuples()]
    t1 = T("ST9.1", "ST9", "Supplementary Table ST9.1. Discrimination and calibration differences, HTFS minus P*, across tasks and seeds", pd.DataFrame(rows),
           ["Task × seed units", "Median ΔAUROC", "ΔAUROC > 0", "Median ΔAUPRC", "ΔAUPRC > 0", "Median Δintercept", "Median Δslope", "Δslope > 0"],
           ["MIMIC-IV and eICU-CRD are summarized jointly, as in the frozen summary. Each unit is one task under one seed's complete out-of-fold predictions; the three seeds reuse the same patients and are not independent samples. Δ = HTFS − P*. "
            "Whole-task AUROC/AUPRC do not substitute for the balanced Brier primary contrast. Cross-seed means and ranges for every model and task are in the data tables."],
           [("R49", "D3_配对差汇总_R43格式", len(d3))], [h[["任务乘seed", "AUROC差中位", "AUROC差为正", "AUPRC差中位", "AUPRC差为正", "校准截距差中位", "校准斜率差中位", "校准斜率差为正"]]],
           fidelity="shortened for print (HTFS vs P*; all pairs in data table)",
           long=[("ST9_paired_summary_all", english(d3, drop=["说明"]), "R49 D3"), ("ST9_model_cross_seed", english(sheet("R49", "D1_各模型跨seed汇总")), "R49 D1"),
                 ("ST9_paired_cross_seed", english(sheet("R49", "D2_配对差跨seed汇总")), "R49 D2"), ("ST9_ICU_per_seed_metrics", english(sheet("R48", "S43_B1_辅助指标逐seed")), "R48 S43"),
                 ("ST9_ICU_per_seed_pairs", english(sheet("R48", "S44_B2_配对差逐seed")), "R48 S44"), ("ST9_NWICU_per_seed_metrics", english(sheet("R48", "S46_6_辅助指标逐seed")), "R48 S46"),
                 ("ST9_NWICU_per_seed_pairs", english(sheet("R48", "S47_7_配对差逐seed")), "R48 S47"), ("ST9_NHANES_per_seed_metrics", english(sheet("R48", "S48_6_辅助指标逐seed")), "R48 S48"),
                 ("ST9_NHANES_per_seed_pairs", english(sheet("R48", "S49_7_配对差逐seed")), "R48 S49"), ("ST9_self_calib_summary", english(sheet("R48", "S50_E4_各模型自身校准_摘要")), "R48 S50"),
                 ("ST9_self_calibration_full", english(sheet("R48", "S51_X1_各模型自身校准_全表")), "R48 S51"),
                 ("ST9_alignment_arms_calib", english(sheet("R49", "D5_NHANES目标一致性追加臂"), drop=["性质", "来源"]), "R49 D5")])
    d4 = sheet("R49", "D4_校准诊断汇总_R43格式")
    d4 = d4.assign(_db=d4.数据库.map(lambda x: {"MIMIC-IV + eICU-CRD": 0, "NWICU": 2, "NHANES": 3}.get(tr(x), 9))).sort_values(["_db"], kind="stable")
    rows = [{"Database": tr(r["数据库"]), "Arm": r["臂"], "Units": fnum(r["校准截距_count"]), "Intercept, mean": fnum(r["校准截距_mean"], 3),
             "Intercept, 5–95%": f"{fnum(r['校准截距_5%'], 3)} to {fnum(r['校准截距_95%'], 3)}", "Slope, mean": fnum(r["校准斜率_mean"], 3),
             "Slope, 5–95%": f"{fnum(r['校准斜率_5%'], 3)} to {fnum(r['校准斜率_95%'], 3)}"} for _, r in d4.iterrows()]
    t2 = T("ST9.2", "ST9", "Supplementary Table ST9.2. Calibration intercept and slope of the selected models across task × seed units", pd.DataFrame(rows),
           ["Units", "Intercept, mean", "Intercept, 5–95%", "Slope, mean", "Slope, 5–95%"],
           ["Logistic calibration of each model's own whole-task out-of-fold predictions; summaries over task × seed units."],
           [("R49", "D4_校准诊断汇总_R43格式", len(d4))], [d4[["校准截距_count", "校准截距_mean", "校准截距_5%", "校准截距_95%", "校准斜率_mean", "校准斜率_5%", "校准斜率_95%"]]])
    return [t1, t2]


# ── ST10 ─────────────────────────────────────────────────────────
def st10():
    s53 = sheet("R48", "S53_D0_认证纳入"); m = order(s53)
    t1 = T("ST10.1", "ST10", "Supplementary Table ST10.1. Units entering the real-data fit–certify–test comparison (MIMIC-IV and eICU-CRD)",
           pd.DataFrame([{"Database": DB[r.库], "Outcome": OC[r.结局], "Phenotype": PH[r.表型], "Main units": fnum(r.main_units), "Certification units": fnum(r.cert_units),
                          "Proportion certified (%)": fpct(r.认证比较纳入单位比例, 1)} for _, r in m.iterrows()]), ["Main units", "Certification units", "Proportion certified (%)"],
           ["A certification unit is one task × seed × outer fold for which the certification branch was executable."], [("R48", "S53_D0_认证纳入", len(s53))],
           [s53[["main_units", "cert_units", "认证比较纳入单位比例"]]], long=[("ST10_certification_per_unit", english(order(sheet("R48", "S56_D3_认证逐单元"))), "R48 S56")])
    s54 = sheet("R48", "S54_D1_认证汇总"); s55 = sheet("R48", "S55_D2_认证交叉表_描述")
    rows = [{"Database": DB[r.库], "Track": r.轨, "Track certifications": fnum(r.单元), "Specialist enabled": fnum(r.专业化)} for r in s54.itertuples()]
    t2 = T("ST10.2", "ST10", "Supplementary Table ST10.2. Real-data certification decisions by database and track", pd.DataFrame(rows), ["Track certifications", "Specialist enabled"],
           [f"In total, {fnum(s54.专业化.sum())} of {fnum(s54.单元.sum())} evaluable track-level certification decisions enabled a specialist. Decisions overlap across tasks, seeds, folds and patients; "
            "the count is descriptive, not an error rate or a safety guarantee. Cross-tabulation with test-fold realized gains is in data table ST10_cross_tab."],
           [("R48", "S54_D1_认证汇总", len(s54))], [s54[["单元", "专业化"]]], long=[("ST10_cross_tab", english(s55), "R48 S55")])
    s58 = sheet("R48", "S58_C2_认证分解汇总")
    rows = [{"Scale": SC[r.尺度], "Tasks": fnum(r.任务), "Sample-cost term, median (%)": fpct(r.样本成本项相对中位, 2, True), "Enabling term, median (%)": fpct(r.认证启用项相对中位, 2, True),
             "Total, median (%)": fpct(r.合计相对中位, 2, True), "Maximum |enabling term|": fnum(r.认证启用项绝对最大, 6), "Maximum |identity residual|": fnum(r.恒等式残差绝对最大, 6)} for r in s58.itertuples()]
    t3 = T("ST10.3", "ST10", "Supplementary Table ST10.3. Decomposition of the deployed certification risk relative to full-development P*", pd.DataFrame(rows),
           ["Tasks", "Sample-cost term, median (%)", "Enabling term, median (%)", "Total, median (%)", "Maximum |enabling term|", "Maximum |identity residual|"],
           ["Relative changes versus R(P*_D) on records jointly evaluable for the certification comparison; negative values mean higher risk than full-development P*. "
            "Total = sample-cost term (P*_F vs P*_D) + enabling term (CERT vs P*_F). Task-level rows are in data table ST10_decomposition."],
           [("R48", "S58_C2_认证分解汇总", len(s58))], [s58[["任务", "样本成本项相对中位", "认证启用项相对中位", "合计相对中位", "认证启用项绝对最大", "恒等式残差绝对最大"]]],
           long=[("ST10_decomposition", english(order(sheet("R48", "S57_C1_认证分解"), sc="尺度")), "R48 S57")])
    return [t1, t2, t3]


# ── ST11 ─────────────────────────────────────────────────────────
def st11():
    s60 = sheet("R48", "S60_E2_学习曲线汇总"); s61 = sheet("R48", "S61_D2_学习曲线实际比例汇总")
    m = s60.merge(s61, left_on=["库", "比例"], right_on=["库", "目标比例"], how="left").assign(_db=lambda d: d.库.map(lambda x: ODB[DB[x]])).sort_values(["_db", "比例"])
    rows = [{"Database": DB[r.库], "Target training share": fnum(r.比例, 2), "Actual share, median (range)": f"{fnum(r.实际比例中位, 3)} ({fnum(r.实际比例最小, 3)} to {fnum(r.实际比例最大, 3)})",
             "Tasks": fnum(r.任务数), "Tasks evaluable in all folds": fnum(r.全部外层折可评价任务), "Tasks evaluable in any fold": fnum(r.有任一折可评价任务),
             "Median balanced HTFS vs P* (%)": fpct(r.HTFS_vs_P星_bal中位_全可评价任务, 2, True)} for r in m.itertuples()]
    t1 = T("ST11.1", "ST11", "Supplementary Table ST11.1. Learning-curve evaluability and median balanced specialist gain by training share", pd.DataFrame(rows),
           ["Target training share", "Actual share, median (range)", "Tasks", "Tasks evaluable in all folds", "Tasks evaluable in any fold", "Median balanced HTFS vs P* (%)"],
           ["Only evaluable points are reported; no crossing threshold is inferred. Medians are over tasks evaluable in all outer folds. Task-level values are in data table ST11_learning_curve."],
           [("R48", "S60_E2_学习曲线汇总", len(s60)), ("R48", "S61_D2_学习曲线实际比例汇总", len(s61))],
           [s60[["比例", "任务数", "全部外层折可评价任务", "有任一折可评价任务", "HTFS_vs_P星_bal中位_全可评价任务"]], s61[["实际比例中位", "实际比例最小", "实际比例最大"]]],
           long=[("ST11_learning_curve", english(order(sheet("R48", "S59_E1_学习曲线"))), "R48 S59"), ("ST11_absolute_risk", english(order(sheet("R48", "S62_D3_学习曲线绝对风险"))), "R48 S62")])
    s63 = sheet("R48", "S63_D4_绝对风险汇总"); m63 = order(s63).sort_values(["结局", "目标比例"], kind="stable") if False else order(s63)
    rows = [{"Database": DB[r.库], "Outcome": OC[r.结局], "Target training share": fnum(r.目标比例, 2), "Tasks": fnum(r.同组任务), "Phenotypes": ", ".join(PH[x] for x in str(r.任务).split(",")), "Median balanced risk, P*": fnum(r.P星两轨等权中位, 4),
             "Median balanced risk, HTFS": fnum(r.HTFS两轨等权中位, 4), "Median balanced risk, Hfull": fnum(r.Hfull两轨等权中位, 4)} for _, r in m63.iterrows()]
    t2 = T("ST11.2", "ST11", "Supplementary Table ST11.2. Median absolute balanced Brier risk by training share", pd.DataFrame(rows),
           ["Target training share", "Tasks", "Median balanced risk, P*", "Median balanced risk, HTFS", "Median balanced risk, Hfull"],
           ["Absolute risks on the common evaluated units of each task; medians across the listed tasks (those evaluable at the share) within database and outcome."], [("R48", "S63_D4_绝对风险汇总", len(s63))],
           [s63[["目标比例", "同组任务", "P星两轨等权中位", "HTFS两轨等权中位", "Hfull两轨等权中位"]]],
           long=[("ST11_risk_decrease_25_100", english(order(sheet("R48", "S64_D5_25到100下降"))), "R48 S64")])
    return [t1, t2]


# ── ST12 ─────────────────────────────────────────────────────────
T2DESC = {"N1": ("Gaussian, compound symmetric (23 dims)", "Bernoulli(0.15), independent of X", "No track difference"),
          "N2": ("Gaussian, compound symmetric (23 dims)", "Bernoulli(0.15), independent of X", "Intercept difference only"),
          "P1": ("Gaussian, compound symmetric (23 dims)", "Bernoulli(0.15), independent of X", "Phenotype interaction (Z × X6)"),
          "P2": ("Gaussian, compound symmetric (23 dims)", "Bernoulli(0.15), independent of X", "Track-specific nonlinear sparse structures"),
          "N1-T": ("Gaussian, compound symmetric (23 dims)", "Threshold of X8 (85th percentile)", "No track difference"),
          "P2-T": ("Gaussian, compound symmetric (23 dims)", "Threshold of X8 (85th percentile)", "Track-specific nonlinear sparse structures"),
          "N1-H": ("Multivariate t5 (heavy-tailed)", "Bernoulli(0.15), independent of X", "No track difference"),
          "P2-H": ("Multivariate t5 (heavy-tailed)", "Bernoulli(0.15), independent of X", "Track-specific nonlinear sparse structures")}
ORDC = ["N1", "N2", "P1", "P2", "N1-T", "P2-T", "N1-H", "P2-H"]
ORDS = ["ALWAYS_POOLED", "ALWAYS_SPLIT", "CV_ANY|P*", "CV_ANY|S0*", "CV_POINT_M|P*", "CV_POINT_M|S0*", "CV_LB_M|P*", "CV_LB_M|S0*", "FCT|P*", "FCT|S0*", "USTUN_ADAPT", "MIX", "POOLMIX"]


def oc2(df, cond="条件", sel=None, trk=None):
    d = df.copy(); d["_c"] = d[cond].map(ORDC.index); keys = ["_c"]
    if sel:
        d["_s"] = d[sel].map(lambda x: ORDS.index(x) if x in ORDS else 99); keys.append("_s")
    if trk:
        keys.append(trk)
    return d.sort_values(keys, kind="stable").drop(columns=[k for k in keys if k.startswith("_")]).reset_index(drop=True)


def st12():
    c1 = oc2(sheet("R47", "C1_T2条件与设计"))
    rows = [{"Condition": r.条件, "Covariates": T2DESC[r.条件][0], "Phenotype Z": T2DESC[r.条件][1], "Outcome structure": T2DESC[r.条件][2], "Development n": fnum(r.开发集n),
             "π_A": fnum(r.π_A, 2), "Replicates": fnum(r.完成重复), "Mean development event rate": fnum(r.开发事件率均值, 3),
             "Bayes risk, track A": fnum(r.A轨Bayes风险, 4), "Bayes risk, track B": fnum(r.B轨Bayes风险, 4)} for r in c1.itertuples()]
    t1 = T("ST12.1", "ST12", "Supplementary Table ST12.1. Known-truth simulation (T2) conditions and design", pd.DataFrame(rows),
           ["Development n", "π_A", "Replicates", "Mean development event rate", "Bayes risk, track A", "Bayes risk, track B"],
           ["Exact outcome functions are given in Supplementary Methods 2. True risks were evaluated on independent covariate draws (adaptively 100,000 to 1,000,000 per replicate)."],
           [("R47", "C1_T2条件与设计", len(c1))], [c1[["开发集n", "π_A", "完成重复", "开发事件率均值", "A轨Bayes风险", "B轨Bayes风险"]]],
           long=[("ST12_design_full", english(c1, drop=["模块", "协变量", "Z", "结局函数"]), "R47 C1")])
    c2 = oc2(sheet("R47", "C2_T2候选对真值收益"), trk="候选对")
    rows = [{"Condition": r.条件, "Candidate pair": r.候选对, "Track": r.轨, "True relative gain, mean (%)": fpct(r.真值相对收益均值, 2, True),
             "Mean ± 1.96 SE (%)": fci(r.下, r.上), "vs m": D0[r.相对m方向], "P5 to P95 (%)": fci(r.P5, r.P95)} for r in c2.itertuples()]
    t2 = T("ST12.2", "ST12", "Supplementary Table ST12.2. True relative gain of the specialists over pooled references in T2", pd.DataFrame(rows),
           ["True relative gain, mean (%)", "Mean ± 1.96 SE (%)", "P5 to P95 (%)"],
           ["D|P*: HTFS_D vs P*_D; D|S0*: HTFS_D vs S0*_D (full development data); F|P*, F|S0*: the same on the 2/3 fitting split. 200 replicates per condition; mean ± 1.96 between-replicate SE "
            "quantifies Monte Carlo uncertainty of the mean, and P5–P95 the spread across replicates; neither is a clinical confidence interval."],
           [("R47", "C2_T2候选对真值收益", len(c2))], [c2[["真值相对收益均值", "下", "上", "P5", "P95"]]])
    c3 = oc2(sheet("R47", "C3_T2两轨等权部署"), sel="选择器")
    rows = [{"Condition": r.条件, "Selector": r.选择器, "Balanced improvement vs P*_D, mean (%)": fpct(r.两轨等权部署改善均值, 2, True), "Mean ± 1.96 SE (%)": fci(r.下, r.上),
             "vs 0": D0[r["均值±1.96SE方向"]], "P5 to P95 (%)": fci(r.P5, r.P95), "Replicates < 0": fnum(r.点估计为负的重复)} for _, r in c3.iterrows()]
    t3 = T("ST12.3", "ST12", "Supplementary Table ST12.3. Balanced true-risk improvement of the deployed model relative to full-development P* in T2", pd.DataFrame(rows),
           ["Balanced improvement vs P*_D, mean (%)", "Mean ± 1.96 SE (%)", "P5 to P95 (%)", "Replicates < 0"],
           ["Selectors are defined in Supplementary Methods 2. Track-level deployment results and actual-deterioration counts are in data table ST12_track_deployment."],
           [("R47", "C3_T2两轨等权部署", len(c3))], [c3[["两轨等权部署改善均值", "下", "上", "P5", "P95", "点估计为负的重复"]]],
           long=[("ST12_track_deployment", english(oc2(sheet("R47", "C4_T2逐轨部署与变差"), sel="选择器", trk="轨")), "R47 C4"),
                 ("ST12_MIX_three_views", english(sheet("R47", "C5_MIX等三面并列")), "R47 C5"), ("ST12_MIX_vs_POOLMIX", english(sheet("R47", "C5b_MIX与POOLMIX配对差")), "R47 C5b"),
                 ("ST12_track_enabling", english(sheet("R47", "C6a_T2逐轨启用与检出"), drop=["证据性质"]), "R47 C6a"),
                 ("ST12_pairwise_reversal", english(sheet("R47", "C8_两两排序反转_描述"), drop=["性质"]), "R47 C8"),
                 ("ST12_pairwise_all", english(sheet("R48", "S77_X5_T2两两排序全表")), "R48 S77"),
                 ("ST12_extremes_first100k", english(sheet("R47", "C9_T2极端值_初始10万"), drop=["评价范围"]), "R47 C9")])
    c6 = oc2(sheet("R47", "C6b_T2重复级启用错误"), sel="选择器")
    rows = [{"Condition": r["条件"], "Selector": r["选择器"],
             "≥1 track enabled below m": f"{fnd(r['≥1轨不达标启用_分子'], r['≥1轨不达标启用_分母'])} ({fpct(r['≥1轨不达标启用_CP下'], 1)} to {fpct(r['≥1轨不达标启用_CP上'], 1)})",
             "≥1 track enabled with actual deterioration": f"{fnum(r['≥1轨实际变差启用_分子'])} (upper {fpct(r['≥1轨实际变差启用_CP上'], 1)})"} for _, r in c6.iterrows()]
    t4 = T("ST12.4", "ST12", "Supplementary Table ST12.4. Replicate-level wrong enabling by certification-based and adaptive selectors in T2", pd.DataFrame(rows),
           ["≥1 track enabled below m", "≥1 track enabled with actual deterioration"],
           ["Counts over 200 independent replicates per condition with Clopper–Pearson 95% intervals (%). Monte Carlo estimates under the specified generating mechanisms; 0/200 is reported with its upper limit "
            "and is not a validated error rate."],
           [("R47", "C6b_T2重复级启用错误", len(c6))], [c6[["≥1轨不达标启用_分子", "≥1轨不达标启用_分母", "≥1轨不达标启用_CP下", "≥1轨不达标启用_CP上", "≥1轨实际变差启用_分子", "≥1轨实际变差启用_CP上"]]])
    c6c = oc2(sheet("R47", "C6c_T2认证样本成本"), trk="轨")
    rows = [{"Condition": r["条件"], "Track": r["轨"], "P*_F vs P*_D, mean (%)": fpct(r["P*_F相对P*_D改善均值"], 2, True), "Mean ± 1.96 SE (%)": fci(r["下"], r["上"]),
             "vs 0": D0[r["均值±1.96SE方向"]]} for _, r in c6c.iterrows()]
    t5 = T("ST12.5", "ST12", "Supplementary Table ST12.5. Sample cost of reserving one third of development data for certification in T2", pd.DataFrame(rows),
           ["P*_F vs P*_D, mean (%)", "Mean ± 1.96 SE (%)"],
           ["True relative improvement of the pooled reference fitted on the 2/3 fitting split over the one fitted on all development data; negative values are the sample cost."],
           [("R47", "C6c_T2认证样本成本", len(c6c))], [c6c[["P*_F相对P*_D改善均值", "下", "上"]]])
    return [t1, t2, t3, t4, t5]


# ── ST13 ─────────────────────────────────────────────────────────
C10TBL = {"候选对真值收益（HTFS_D vs P*_D）": "Candidate-pair true gain (HTFS_D vs P*_D)", "选择器两轨等权部署改善": "Balanced deployment improvement"}


def st13():
    c10 = sheet("R47", "C10_plasmode探索")
    rows = [{"Quantity": C10TBL[r.表], "δ": fnum(r.δ, 1), "Candidate pair / selector": r.候选对 if pd.notna(r.候选对) else r.选择器, "Track": r.轨 if pd.notna(r.轨) else "Balanced",
             "Replicates": fnum(r.重复), "Mean (%)": fpct(r.均值, 2, True), "SE (%)": fpct(r.SE, 2)} for r in c10.itertuples()]
    t1 = T("ST13.1", "ST13", "Supplementary Table ST13.1. Exploratory plasmode simulation on real MIMIC-IV covariate structure", pd.DataFrame(rows), ["δ", "Replicates", "Mean (%)", "SE (%)"],
           ["Exploratory: real MIMIC-IV covariates with a specified outcome mechanism whose track-specific signal is scaled by δ; 20 replicates per δ. Not part of the confirmatory evidence."],
           [("R47", "C10_plasmode探索", len(c10))], [c10[["δ", "重复", "均值", "SE"]]], long=[("ST13_T1_vs_T2", english(sheet("R47", "C7_T1与T2数值并列"), drop=["地位", "T1与T2"]), "R47 C7")])
    o1 = sheet("R48", "O1_有限D候选硬选择oracle")
    rows = [{"Candidate set": tr(r.候选集), "Condition": r.条件, "Balanced improvement, mean (%)": fpct(r.两轨等权改善均值, 2, True), "Mean ± 1.96 SE (%)": fci(r.下, r.上),
             "Track A, mean (%)": fpct(r.A轨改善均值, 2, True), "Track B, mean (%)": fpct(r.B轨改善均值, 2, True),
             "HTFS_D chosen (A/B replicates)": f"{fnum(r.A轨取HTFS_D的重复)}/{fnum(r.B轨取HTFS_D的重复)}"} for r in oc2(o1).itertuples()]
    t2 = T("ST13.2", "ST13", "Supplementary Table ST13.2. Post hoc finite-candidate hard-selection oracle in T2", pd.DataFrame(rows),
           ["Balanced improvement, mean (%)", "Mean ± 1.96 SE (%)", "Track A, mean (%)", "Track B, mean (%)", "HTFS_D chosen (A/B replicates)"],
           ["Post hoc description: in each replicate and track, the D-arm candidate with the lowest true risk was chosen with hindsight; improvement is relative to P*_D. "
            "It is not an upper bound for all methods (MIX can blend predictions; FCT deploys F-trained models), not prespecified and not a rerun of T2."],
           [("R48", "O1_有限D候选硬选择oracle", len(o1))], [o1[["两轨等权改善均值", "下", "上", "A轨改善均值", "B轨改善均值", "A轨取HTFS_D的重复", "B轨取HTFS_D的重复"]]])
    return [t1, t2]


# ── ST14 ─────────────────────────────────────────────────────────
S3EN = {"同一住院首段 ICU 的 intime 并列（按 stay_id 最小打破）": "Tied first ICU entry within an admission (broken by smallest stay identifier)", "年龄不明": "Unknown age",
        "性别为 F/M 之外或缺失（不排除；表型不可判定）": "Sex other than F/M or missing (retained; phenotype not ascertainable)",
        "死亡状态且 deathtime 缺失（用 dischtime 作死亡时点）": "Died with missing death time (discharge time used)",
        "结局不确定①：死亡状态但死亡时点晚于出院 + 1 天": "Outcome uncertain (1): died but death time later than discharge + 1 day",
        "结局不确定②：活着出院的状态但 deathtime ≤ dischtime": "Outcome uncertain (2): discharged alive but death time ≤ discharge time",
        "结局不确定③：discharge_location 缺失": "Outcome uncertain (3): missing discharge location",
        "活着出院且 deathtime 晚于出院（出院后死亡，非事件）": "Discharged alive with death after discharge (not an event)",
        "hospital_expire_flag = 1（不使用）": "hospital_expire_flag = 1 (not used)", "hospital_expire_flag = 1 且死亡状态且无 deathtime": "hospital_expire_flag = 1, died, no death time"}
S7EN = {"UNKNOWN 人次（合格队列）": "Stays with UNKNOWN discharge location (eligible cohort)", "类别 K1": "Category K1", "类别 K2": "Category K2 (evidence of survival)",
        "类别 K3": "Category K3 (no evidence)", "类别 K3a": "Category K3a (death close to discharge)", "类别 K4": "Category K4",
        "K1 中按既定规则记事件（E_dt > landmark）": "K1 coded as events by the prespecified rule", "K2 依据：后续入院": "K2 evidence: subsequent admission",
        "K2 依据：deathtime 晚于出院 24 h": "K2 evidence: death time >24 h after discharge", "K2 依据：dod 晚于出院日 ≥ 2 天": "K2 evidence: date of death ≥2 days after discharge",
        "K2 只有一种依据": "K2 with a single type of evidence", "UNKNOWN 中 hospital_expire_flag = 1": "UNKNOWN with hospital_expire_flag = 1",
        "UNKNOWN 中 deathtime 非空": "UNKNOWN with nonmissing death time", "UNKNOWN 中 dod 非空": "UNKNOWN with nonmissing date of death",
        "住院死亡不可评价（排除）合计": "In-hospital death not evaluable (excluded), total", "住院死亡标签改变（可评价者中）": "In-hospital death labels changed (among evaluable)",
        "排除人次涉及患者数": "Patients involved in excluded stays", "其中该患者在合格队列中的全部人次都被排除": "Of whom all eligible stays were excluded"}
S34INF = {"R44 E1（HTFS vs P*）": "Original E1 (HTFS vs P*)", "R45 C1（HTFS_pop vs P*_pop）": "C1 (HTFS_pop vs P*_pop)"}


def st14():
    s7 = sheet("R48", "S7_B1_NWICU_UNKNOWN核实")
    t1 = T("ST14.1", "ST14", "Supplementary Table ST14.1. Verification of NWICU stays with UNKNOWN discharge location (in-hospital death outcome)",
           pd.DataFrame([{"Item": S7EN[r.项], "N": fnum(r.N)} for r in s7.itertuples()]), ["N"],
           ["The correction was identified after the first NWICU results were seen. The 200 stays without evidence were excluded from in-hospital death only; no evaluable label changed. "
            "The total ICU stay >72 h outcome was unaffected."],
           [("R48", "S7_B1_NWICU_UNKNOWN核实", len(s7))], [s7[["N"]]],
           long=[("ST14_NWICU_outcome_registry", english(sheet("R48", "S3_E2_NWICU_结局登记").assign(项=lambda d: d.项.map(S3EN))), "R48 S3"),
                 ("ST14_NWICU_task_changes", english(order(sheet("R48", "S8_B3_NWICU_住院死亡任务变化"))), "R48 S8"),
                 ("ST14_NWICU_vital_sign_defs", english(sheet("R48", "S83_E3_NWICU体征两口径")), "R48 S83")])
    s11 = sheet("R48", "S11_C1_NHANES_未知死因")
    t2 = T("ST14.2", "ST14", "Supplementary Table ST14.2. NHANES death with unknown underlying cause and the affected cardiovascular-death tasks",
           pd.DataFrame([{"Outcome": OC[r.结局], "Phenotype": PH[r.表型], "Included originally": tr(r.R44已纳入), "Unknown-cause death in task": tr(r.未知者在任务人群中),
                          "Track": tr(r.未知者所在轨) if pd.notna(r.未知者所在轨) else "—", "Rerun required": tr(r.需要重跑)} for _, r in order(s11).iterrows()]), [],
           ["The participant was removed from the affected 10-year cardiovascular-death tasks after outer folds were formed on the original population, so fold membership of all others was preserved."],
           [("R48", "S11_C1_NHANES_未知死因", len(s11))], [])
    s28 = sheet("R48", "S28_B4_NWICU_住院死亡E1修正前后"); s34 = sheet("R48", "S34_C2_NHANES_O_CVD10修正前后")
    rows = []
    for _, r in order(s28, sc="尺度").iterrows():
        rows.append({"Source": "NWICU", "Contrast": "E1", "Outcome": OC[r.结局], "Phenotype": PH[r.表型], "Scale": SC[r.尺度],
                     "Before (%)": f"{fpct(r.观察相对改善_R45, 2, True)} ({fci(r.相对_P2_5_R45, r.相对_P97_5_R45)})", "Judgment before": TRI(r["三态_m0.010_R45"]),
                     "After (%)": f"{fpct(r.观察相对改善_R46, 2, True)} ({fci(r.相对_P2_5_R46, r.相对_P97_5_R46)})", "Judgment after": TRI(r["三态_m0.010_R46"])})
    for _, r in order(s34, sc="尺度").iterrows():
        rows.append({"Source": "NHANES", "Contrast": S34INF[r.推断], "Outcome": OC[r.结局], "Phenotype": PH[r.表型], "Scale": SC[r.尺度],
                     "Before (%)": f"{fpct(r.观察相对改善_原, 2, True)} ({fci(r.相对_P2_5_原, r.相对_P97_5_原)})", "Judgment before": TRI(r["三态_m0.010_原"]),
                     "After (%)": f"{fpct(r.观察相对改善_v2, 2, True)} ({fci(r.相对_P2_5_v2, r.相对_P97_5_v2)})", "Judgment after": TRI(r["三态_m0.010_v2"])})
    t3 = T("ST14.3", "ST14", "Supplementary Table ST14.3. Contrasts before and after the outcome-status corrections", pd.DataFrame(rows), ["Before (%)", "After (%)"],
           ["NWICU: in-hospital death after excluding 200 unresolvable stays; NHANES: 10-year cardiovascular death after removing one unknown-cause death. "
            "Both corrections required refitting, so the differences mix the correction with refitting perturbation (Supplementary Table ST14.4)."],
           [("R48", "S28_B4_NWICU_住院死亡E1修正前后", len(s28)), ("R48", "S34_C2_NHANES_O_CVD10修正前后", len(s34))],
           [s28.select_dtypes("number"), s34.select_dtypes("number")])
    s82 = sheet("R48", "S82_C6_重训扰动汇总")
    t4 = T("ST14.4", "ST14", "Supplementary Table ST14.4. Magnitude of point-estimate changes after refitting for the outcome corrections",
           pd.DataFrame([{"Source": tr(r.来源), "Rows": fnum(r.行), "Median |change| (percentage points)": fnum(100 * r.绝对变化中位, 2),
                          "Maximum |change| (percentage points)": fnum(100 * r.绝对变化最大, 2), "Median |change| / half-width": fnum(r.比值中位, 2),
                          "Maximum |change| / half-width": fnum(r.比值最大, 2), "Sign flips": fnum(r.符号翻转), "Three-state changes": fnum(r.三态改变)} for r in s82.itertuples()]),
           ["Rows", "Median |change| (percentage points)", "Maximum |change| (percentage points)", "Median |change| / half-width", "Maximum |change| / half-width", "Sign flips", "Three-state changes"],
           ["Fixed out-of-fold intervals do not include redevelopment variability; changes of this size can arise from refitting alone. Row-level values are in data table ST14_perturbation_rows."],
           [("R48", "S82_C6_重训扰动汇总", len(s82))], [s82[["行", "符号翻转", "三态改变", "比值中位", "比值最大"]], (s82[["绝对变化中位", "绝对变化最大"]] * 100)],
           long=[("ST14_perturbation_rows", english(sheet("R48", "S81_C5_重训扰动幅度")), "R48 S81")])
    return [t1, t2, t3, t4]


# ── ST15 ─────────────────────────────────────────────────────────
def st15():
    e = sheet("R52", "8_E1主对照"); bal = order(e[e.尺度 == "bal"])
    rows = [{"Outcome": OC[r.结局], "Phenotype": PH[r.表型], "Frozen primary (%)": f"{fpct(r.观察相对改善_旧, 2, True)} ({fci(r.相对_P2_5_旧, r.相对_P97_5_旧)})",
             "Standardized creatinine (%)": f"{fpct(r.观察相对改善_新, 2, True)} ({fci(r.相对_P2_5_新, r.相对_P97_5_新)})",
             "Judgment, primary → sensitivity": f"{TRI(r['三态_m0.010_旧'])} → {TRI(r['三态_m0.010_新'])}"} for _, r in bal.iterrows()]
    t1 = T("ST15.1", "ST15", "Supplementary Table ST15.1. Post hoc NHANES serum-creatinine standardization sensitivity analysis: primary balanced E1 contrast", pd.DataFrame(rows),
           ["Frozen primary (%)", "Standardized creatinine (%)"],
           ["Post hoc: the NCHS-recommended standardization (−0.016 + 0.978 × original, mg/dL) was applied to 4,154 nonmissing 2005–2006 values, replacing the creatinine predictor and recalculating eGFR "
            "and the reduced-eGFR phenotype; the complete NHANES program was redeveloped. It does not replace the frozen primary analysis. Pointwise 95% intervals in parentheses. "
            "The complete evidence workbook (all scales, contrasts, max-T, auxiliary metrics, run checks) is supplied as a separate file."],
           [("R52", "8_E1主对照", len(e))], [bal[["观察相对改善_旧", "相对_P2_5_旧", "相对_P97_5_旧", "观察相对改善_新", "相对_P2_5_新", "相对_P97_5_新"]]],
           fidelity="shortened for print (balanced scale; full R52 evidence workbook attached)")
    jd = order(sheet("R52", "16_L3预定判读")); jd = jd.assign(_k=jd.对照.map({"C1": 0, "D1": 1})).sort_values("_k", kind="stable").drop(columns="_k").reset_index(drop=True)
    rows = [{"Contrast": r.对照, "Outcome": OC[r.结局], "Phenotype": PH[r.表型], "Scale": SC[r.判读尺度], "Frozen primary (%)": f"{fpct(r.观察相对改善_旧, 2, True)} ({fci(r.相对_P2_5_旧, r.相对_P97_5_旧)})",
             "Standardized creatinine (%)": f"{fpct(r.观察相对改善_新, 2, True)} ({fpct(r.相对_P2_5_新, 3 if abs(r.相对_P2_5_新) < 0.00005 else 2, True)} to {fpct(r.相对_P97_5_新, 2, True)})",
             "Judgment, primary → sensitivity": f"{TRI(r.三态_m1_旧)} → {TRI(r.三态_m1_新)}"} for _, r in jd.iterrows()]
    t2 = T("ST15.2", "ST15", "Supplementary Table ST15.2. Post hoc creatinine-standardization sensitivity: objective-alignment judgments", pd.DataFrame(rows),
           ["Frozen primary (%)", "Standardized creatinine (%)"], ["C1 and D1 as defined in Supplementary Table ST8.1; both remain explanatory analyses."],
           [("R52", "16_L3预定判读", len(jd))], [jd[["观察相对改善_旧", "相对_P2_5_旧", "相对_P97_5_旧", "观察相对改善_新", "相对_P2_5_新", "相对_P97_5_新"]]])
    return [t1, t2]


# ── SM2 ──────────────────────────────────────────────────────────
SM2_TEXT = [
    "Supplementary Methods 2. Learning programs, inference, simulation and selectors",
    "Frozen engine. XGBoost binary logistic models (histogram tree method; learning rate 0.05; subsample 0.8; column subsample 0.8; L2 penalty 1.0; 256 bins; one thread) with up to 2,000 rounds "
    "and early stopping after 50 rounds on a group-based hold-out (one of seven folds). Recalibration was training-only, three-fold cross-fitted Platt scaling fitted within each phenotype track. "
    "Inner model selection used five group-based folds. Grids: specialists, maximum depth {2, 3, 4, 6} × minimum child weight {1, 10}; pooled candidates additionally depth 7. "
    "Feature-panel sizes k ∈ {5, 8, 12, 16, 23}, ranked by mean absolute TreeSHAP from a depth-3 ranking model.",
    "Arms. Pooled families: S0 (23-predictor pool without Z), SZ (with Z; identical to S0 in the female-sex task), SZ-GFS (training-fold global top-k plus Z) and SZ-union "
    "(union of the two tracks' training-fold selections plus Z), each with unweighted or track-balanced training weights. Specialist families, trained separately in each track: Hfull (full pool), "
    "HGFS (global panel) and HTFS (each track's own training-fold selection, grid including k = 23). P* is the pooled candidate (family × calibration × weighting) with the smallest inner "
    "cross-validated balanced Brier risk, ties broken in the listed order.",
    "Scales and judgments. R_A and R_B are mean Brier scores within tracks; the balanced scale is (R_A + R_B)/2 and the sample-weighted scale weights records equally within the task. "
    "Relative improvement = [R(reference) − R(candidate)]/R(reference); it is not a reduction in patients' event risk. With the technical margin m = 1% (not a minimal clinically important difference): "
    "Supported if the lower limit exceeds m, Excluded if the upper limit is below m, otherwise Inconclusive. Real-data intervals come from joint cluster resampling (2,000 draws) of fixed out-of-fold "
    "predictions after averaging per-unit losses over three seeds; study-level track claims use max-T simultaneous bands. Neither includes algorithm-level redevelopment variability.",
    "Objective alignment (NHANES). C1 and D1 changed the pooled program's development objective — early-stopping weights, inner scoring, P* selection and calibration choice — to sample-weighted (C1) "
    "or track-B (D1) risk; specialist hyperparameters and panels were frozen and only their calibration choice followed the objective.",
    "Known-truth simulation (T2). Development n = 6,000, π_A = 0.15, 200 replicates per condition (conditions N1, N2, P1, P2, N1-T, P2-T, N1-H, P2-H; root seed 20461). "
    "Covariates X0–X22 are indexed as in the analysis code. With common(X) = 0.8 X0 − 0.5 X1 + 0.4 tanh(X2 X3) + 0.3 X5, the logit of the true risk was: N1, −1.6 + common(X); "
    "N2, −1.9 + common(X) + 0.8 Z; P1, −1.6 + common(X) + 1.0 Z X6; P2, −1.4 + g_A(X) in track A and −1.7 + g_B(X) in track B, with g_A = 1.2 tanh(1.5 X0) + 0.9 X1 X2 − 0.6 (X3² − 1) and "
    "g_B = 1.0 X4 − 0.8 |X5| + 0.7 X6 X7. Gaussian covariates had compound-symmetric correlation 0.2 and unit variance. The -T conditions defined Z = 1[X8 > Φ⁻¹(0.85)]; "
    "the -H conditions used multivariate t5 covariates scaled to unit variance. "
    "Conditional Brier loss p(1 − p) + (p − p̂)² was averaged over independent evaluation covariates (100,000 to 1,000,000 per replicate, extended adaptively near decision boundaries).",
    "T2 arms and selectors. D arms (P*_D, HTFS_D, S0*_D) were developed on all development data; F arms on the 2/3 fitting split after reserving 1/3 for certification. "
    "ALWAYS_POOLED / ALWAYS_SPLIT deploy P*_D / HTFS_D. CV_ANY, CV_POINT_M and CV_LB_M (reference P* or S0*) select on inner out-of-fold estimates — any point gain, point gain ≥ m, "
    "or an approximate one-sided lower bound > m — and deploy D arms without a formal guarantee. FCT (fit–certify–test) tests H0: gain ≤ m on the certification split for each track "
    "(one-sided α = 0.025) and deploys HTFS_F only on rejection, otherwise the F-arm reference, without refitting. USTUN_ADAPT is an adaptation of the rationality condition of Ustun et al. "
    "(m = 0 test against P*_F), not a reproduction. MIX blends (1 − t) P*_D + t HTFS_D per track with t ∈ {0, 0.25, 0.5, 0.75, 1} chosen by inner track-specific Brier risk; "
    "POOLMIX blends P*_D with the alternative pooled family instead of HTFS_D.",
    "T2 outcomes. Deployment improvement is the true relative improvement of the selected model over P*_D (per track; balanced after averaging track risks). Actual deterioration: improvement < 0 "
    "by at least two integration standard errors. Enabling below m: enabling a specialist whose candidate-pair true gain is below m (and not near m). Means are reported with ±1.96 between-replicate SE; "
    "proportions with Clopper–Pearson 95% intervals.",
    "Data and code availability. MIMIC-IV, eICU-CRD and NWICU are credentialed-access PhysioNet databases and are not redistributed; NHANES public-use files and linked mortality files are available from NCHS. "
    "Individual-level predictions and restricted records are not released. The analysis code, frozen summary workbooks and synthetic-simulation records can be provided as described in the main text; "
    "the public release status is to be confirmed by the authors.",
]


def sm2():
    e1 = sheet("R48", "E1_环境依赖")
    ROLE = {"python": "Interpreter (Windows 11)", "pillow": "300-dpi TIFF export", "matplotlib": "Figures", "numpy": "Numerical computation", "pandas": "Tables and input/output",
            "psutil": "Resource logging", "psycopg2-binary": "MIMIC-IV/eICU-CRD data build (PostgreSQL)", "scipy": "Inference and optimisation",
            "scikit-learn": "Cross-validation utilities, Platt calibration", "threadpoolctl": "Thread pinning to one", "xgboost": "Frozen engine",
            "pyarrow": "Parquet input/output (pandas engine)", "openpyxl": "Excel input/output (pandas engine)"}
    t = T("SM2.1", "SM2", "Supplementary Table SM2.1. Software environment",
          pd.DataFrame([{"Package": r.包, "Version": str(r.版本), "Role": ROLE.get(r.包, "")} for r in e1.itertuples()]), [],
          ["Versions recorded from the frozen analysis environment."], [("R48", "E1_环境依赖", len(e1))], [])
    return [t]


def all_tables():
    out = [sm1()]
    for f in (st1, st2, st3, st4, st5, st6, st7, st8, st9, st10, st11, st12, st13, st14, st15, sm2):
        r = f()
        out += r if isinstance(r, list) else [r]
    return out
