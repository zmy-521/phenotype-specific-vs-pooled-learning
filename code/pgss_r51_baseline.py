# -*- coding: utf-8 -*-
r"""R51（GPT 任务书 §6）：四库 Table 1 基线描述统计、结局分母对账、六表型可判定覆盖率、补充图 S1 数字核对。只读。

纪律：只读冻结文件；不插补、不换算单位（单位核实不了就标 not comparable）、不重新清洗或定义终点、不拟合；
不同来源之间不做 P 值比较；只输出汇总统计，不输出任何个人记录。
数据版本：MIMIC-IV / eICU-CRD = data v5（19_R38/data_v5，SHA 对 DATA_VERSION.json）；NWICU = R45 冻结队列（内容指纹对 COHORT.json）
+ R46 v2 住院死亡状态（_local_only/MORT_STATUS.parquet）；NHANES = R44 冻结队列（内容指纹对 COHORT.json）+ R46 v2 心血管死亡状态（_local_only/CVD_STATUS.parquet）。
NHANES 的白蛋白、BMI、eGFR 连续值不在冻结文件里：按 R44 队列脚本同一路径在内存重建（不写文件），闸门 = 重建四指纹与登记逐位相同，
过闸才用于 Table 1 描述；不过闸则三项保持 NA。另报 NHANES 2005–2006 血肌酐未按官方建议校准的影响计数（T6），冻结结果不变。
STOP 规则：某库指纹不符 → 只把该库标为暂停，其余继续。
输出：34_R51_基线只读提数/R51_Table1_Baseline_Source.xlsx 与 R51_BASELINE.json；QC 报告由同一结果写成 R51_Baseline_QC.md。
"""
import sys, json, hashlib, pathlib, re, html
HERE = pathlib.Path(__file__).resolve().parent
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd

ROOT = HERE.parent
OUT = ROOT / "34_R51_基线只读提数"
V5 = ROOT / "19_R38" / "data_v5"
NW = ROOT / "27_R45_外部复现与机制核验" / "nwicu" / "data"
NH = ROOT / "25_R44_NHANES跨领域复现" / "data"
R46 = ROOT / "28_R46_最终核验与T2"
NWRAW = pathlib.Path(r"E:/ICU_外部数据/nwicu-northwestern-icu/0.1.0/data/nw_hosp")
X47 = ROOT / "29_R47_写作交接" / "R47_冻结数字包.xlsx"
X49 = ROOT / "31_R49_只读补齐" / "R49_只读补齐_证据表.xlsx"
X48 = ROOT / "30_R48_图源与补充材料" / "R48_图源与补充材料.xlsx" if False else ROOT / "30_R48_图稿与补充材料" / "R48_图源与补充材料.xlsx"
DBS = ["MIMIC-IV", "eICU-CRD", "NWICU", "NHANES"]
PH = ["女性", "年龄≥65", "贫血", "eGFR降低", "低白蛋白", "肥胖"]
PH_EN = {"女性": "Female", "年龄≥65": "Age ≥65 years", "贫血": "Anemia", "eGFR降低": "Reduced eGFR", "低白蛋白": "Low albumin", "肥胖": "Obesity"}
# 连续变量：(英文名, 单位, 小数位, {库: 列名或 None})
CONT = [("Age", "years", 0, {"MIMIC-IV": "年龄", "eICU-CRD": "年龄", "NWICU": "年龄", "NHANES": "年龄"}),
        ("Hemoglobin", "g/dL", 1, {"MIMIC-IV": "血红蛋白", "eICU-CRD": "血红蛋白", "NWICU": "血红蛋白", "NHANES": "血红蛋白"}),
        ("Serum creatinine", "mg/dL", 2, {"MIMIC-IV": "血肌酐", "eICU-CRD": "血肌酐", "NWICU": "血肌酐", "NHANES": "血肌酐"}),
        ("eGFR (CKD-EPI 2021)", "mL/min/1.73 m²", 0, {"MIMIC-IV": "eGFR_landmark", "eICU-CRD": "eGFR_landmark", "NWICU": "eGFR_landmark", "NHANES": None}),
        ("Albumin", "g/dL", 1, {"MIMIC-IV": "白蛋白", "eICU-CRD": "白蛋白", "NWICU": "白蛋白", "NHANES": None}),
        ("Body mass index", "kg/m²", 1, {"MIMIC-IV": "BMI", "eICU-CRD": "BMI", "NWICU": "BMI", "NHANES": None}),
        ("RDW", "%", 1, {"MIMIC-IV": "RDW", "eICU-CRD": "RDW", "NWICU": "RDW", "NHANES": "RDW"}),
        ("Platelet count", "×10³/µL", 0, {"MIMIC-IV": "血小板", "eICU-CRD": "血小板", "NWICU": "血小板", "NHANES": "血小板"}),
        ("Chloride", "mmol/L", 0, {"MIMIC-IV": "血氯", "eICU-CRD": "血氯", "NWICU": "血氯", "NHANES": "血氯"})]
NWICU_ITEMS = {"血肌酐": 100002, "血红蛋白": 100007, "RDW": 100008, "血氯": 100012, "血小板": 100014, "白蛋白": 100021}
NH_EXTRA = {"eGFR (CKD-EPI 2021)": "eGFR_landmark", "Albumin": "白蛋白", "Body mass index": "BMI"}
NHIDX = pathlib.Path(r"D:/NAH公共数据库/NHANES/1_数据/索引/变量索引.csv")
NHDOC = pathlib.Path(r"D:/NAH公共数据库/NHANES/3_存档/官方codebook/2005-2006/BIOPRO_D.htm")
# 图 S1 中印出的数字（GPT generate_flow.py 与 PNG 逐个读出）
FIG = {"MIMIC-IV": [85242, 67222, 67222, 67000], "eICU-CRD": [158442, 110431, 110161, 109104],
       "NWICU": [28612, 26764, 20057, 20057, 20011, 19965], "NHANES": [51623, 26246, 24693, 24666, 23513]}
FIG_EXCL = {"MIMIC-IV": [18020, 0, 222], "eICU-CRD": [48011, 270, 1057], "NWICU": [1848, 6707, 0, 46, 46], "NHANES": [25377, 1553, 27, 1153]}
FIG_OUT = (("MIMIC-IV", "患者", 53539), ("MIMIC-IV", "死亡事件", 7193), ("MIMIC-IV", "LOS72事件", 27068), ("eICU-CRD", "患者", 94920), ("eICU-CRD", "医院", 207),
           ("eICU-CRD", "死亡事件", 9569), ("eICU-CRD", "LOS72事件", 39609), ("NWICU", "单位", 19965), ("NWICU", "患者", 17812), ("NWICU", "LOS72事件", 8146),
           ("NWICU", "死亡分母", 19765), ("NWICU", "死亡患者", 17626), ("NWICU", "死亡事件", 2100), ("NHANES", "全因死亡分母", 23513), ("NHANES", "全因死亡事件", 3813),
           ("NHANES", "CVD分母", 23512))


def sha16f(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]


def sha16df(df):
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[:16]


# ── 读取与版本核对（STOP 规则）────────────────────────────────────
def load():
    D, ver = {}, []
    man = json.loads((V5 / "DATA_VERSION.json").read_text(encoding="utf-8"))["v5_SHA"]
    for db, tag in (("MIMIC-IV", "mimic"), ("eICU-CRD", "eicu")):
        ok = True
        for t in ("X", "META", "PHENO", "PHENOOK"):
            s = sha16f(V5 / f"R38_{tag}_{t}.parquet"); same = s == man[f"{tag}_{t}"]; ok &= same
            ver.append(dict(数据库=db, 文件=f"R38_{tag}_{t}.parquet", 实测=s, 登记=man[f"{tag}_{t}"], 方式="文件 SHA-256 前16位 vs DATA_VERSION.json", 一致=same))
        M = pd.read_parquet(V5 / f"R38_{tag}_META.parquet")
        D[db] = dict(ok=ok, X=pd.read_parquet(V5 / f"R38_{tag}_X.parquet").loc[M.index], M=M,
                     P=pd.read_parquet(V5 / f"R38_{tag}_PHENO.parquet").loc[M.index], OK=pd.read_parquet(V5 / f"R38_{tag}_PHENOOK.parquet").loc[M.index])
    for db, d, coh in (("NWICU", NW, NW / "COHORT.json"), ("NHANES", NH, NH / "COHORT.json")):
        fp = json.loads(coh.read_text(encoding="utf-8"))["指纹"]; ok = True; F = {}
        for t in ("X", "P", "OK", "meta"):
            F[t] = pd.read_parquet(d / f"{t}.parquet"); s = sha16df(F[t]); same = s == fp[t]; ok &= same
            ver.append(dict(数据库=db, 文件=f"{t}.parquet", 实测=s, 登记=fp[t], 方式="内容指纹 sha256(hash_pandas_object(index=True)) vs COHORT.json「指纹」", 一致=same))
        M = F["meta"]
        X = pd.read_parquet(NW / "Xfull.parquet").loc[M.index] if db == "NWICU" else F["X"].loc[M.index]
        D[db] = dict(ok=ok, X=X, Xmodel=F["X"].loc[M.index], M=M, P=F["P"].loc[M.index], OK=F["OK"].loc[M.index])
    reg = json.loads((R46 / "outcome_fix" / "R46_OUTCOME_VERIFY.json").read_text(encoding="utf-8"))
    for db, f, k in (("NWICU", "nwicu_v2/_local_only/MORT_STATUS.parquet", "NWICU"), ("NHANES", "nhanes_v2/_local_only/CVD_STATUS.parquet", "NHANES")):
        S = pd.read_parquet(R46 / f); s = sha16df(S); same = s == reg[k]["状态SHA"]
        D[db]["S"] = S.loc[D[db]["M"].index]; D[db]["ok"] &= same
        ver.append(dict(数据库=db, 文件=f"{f}（R46 v2 结局状态）", 实测=s, 登记=reg[k]["状态SHA"],
                        方式="内容指纹 vs R46_OUTCOME_VERIFY.json「状态SHA」（R46 重跑读取的同一登记）", 一致=same))
    return D, pd.DataFrame(ver)


def nhanes_rebuild(D):
    """NHANES 冻结文件只存了表型标志，没有白蛋白、BMI、eGFR 的连续值。按 R44 队列脚本（pgss_r44_nhanes_cohort.build）
    同一路径在内存中重建（不写任何文件），闸门 = 重建的 X/P/OK/meta 四个内容指纹与 COHORT.json 登记逐位相同；
    过闸才把这三列加进基线描述，否则该三项保持 NA（STOP 只作用于这三项）。只用于描述，不进入任何分析。"""
    sys.path.append(r"D:/NAH公共数据库/NHANES/2_代码")
    import nhanes as nh
    from pgss_r44_nhanes_cohort import RAW, CYCLES, HORIZON_M, CVD_CODES
    d = nh.add_mortality(nh.add_derived(nh.load(RAW, cycles=CYCLES)))
    age = d["D_年龄"]
    keep = (age.ge(20) & d["D_WTMEC"].gt(0) & d["ELIGSTAT"].eq(1) & ~d["RIDEXPRG"].eq(1)).to_numpy()
    d = d[keep].reset_index(drop=True)
    dbp = d["D_舒张压"].where(d["D_舒张压"] > 0); sbp = d["D_收缩压"]
    X = pd.DataFrame({
        "MCV": d.LBXMCVSI, "RDW": d.LBXRDW, "尿素氮": d.LBXSBU, "平均动脉压": (sbp + 2 * dbp) / 3, "年龄": age[keep].to_numpy(),
        "心率": d.BPXPLS, "收缩压": sbp, "白细胞计数": d.LBXWBCSI, "碳酸氢盐": d.LBXSC3SI, "舒张压": dbp, "血小板": d.LBXPLTSI,
        "血氯": d.LBXSCLSI, "血糖": d.LBXSGL, "血红蛋白": d.LBXHGB, "血细胞比容": d.LBXHCT, "血肌酐": d["D_血肌酐_校准"],
        "血钙": d.LBXSCA, "血钠": d.LBXSNASI, "血钾": d.LBXSKSI, "阴离子间隙": d.LBXSNASI - d.LBXSCLSI - d.LBXSC3SI,
        "性别F": d.RIAGENDR.eq(2).astype(float).where(d.RIAGENDR.isin([1, 2]))}).astype(float)
    uid = d.SEQN.astype(np.int64).to_numpy(); X.index = uid
    fem = d.RIAGENDR.eq(2).to_numpy(); hgb = d.LBXHGB.to_numpy(); egfr = d["D_eGFR_CKDEPI2021"].to_numpy()
    alb = d.LBXSAL.to_numpy(); bmi = d["D_BMI"].to_numpy()
    with np.errstate(invalid="ignore"):
        P = pd.DataFrame({"女性": fem, "年龄≥65": X["年龄"].ge(65).to_numpy(), "贫血": np.where(fem, hgb < 12.0, hgb < 13.0) & ~np.isnan(hgb),
                          "eGFR降低": egfr < 60, "低白蛋白": alb < 3.5, "肥胖": bmi >= 30}, index=uid)
    hs, es, as_, bs = (pd.Series(v) for v in (hgb, egfr, alb, bmi))
    OK = pd.DataFrame({"女性": d.RIAGENDR.isin([1, 2]).to_numpy(), "年龄≥65": X["年龄"].notna().to_numpy(),
                       "贫血": (hs.notna() & d.RIAGENDR.isin([1, 2])).to_numpy(), "eGFR降低": es.notna().to_numpy(),
                       "低白蛋白": as_.notna().to_numpy(), "肥胖": bs.notna().to_numpy()}, index=uid)
    died10 = d.MORTSTAT.eq(1) & d.PERMTH_EXM.le(HORIZON_M)
    meta = pd.DataFrame({"Cycle": d.Cycle.to_numpy(), "eligible": True, "group_id": d.SEQN.astype(np.int64).astype(str).to_numpy(),
                         "hospital_id": "NA", "sex": d.RIAGENDR.to_numpy(), "O_MORT10": died10.to_numpy(),
                         "O_CVD10": (died10 & d.UCOD_LEADING.isin(CVD_CODES)).to_numpy(),
                         "PERMTH_EXM": d.PERMTH_EXM.to_numpy(), "MORTSTAT": d.MORTSTAT.to_numpy()}, index=uid)
    reg = json.loads((NH / "COHORT.json").read_text(encoding="utf-8"))["指纹"]
    got = {k: sha16df(v) for k, v in (("X", X), ("P", P), ("OK", OK), ("meta", meta))}
    extra = pd.DataFrame({"eGFR_landmark": egfr, "白蛋白": alb, "BMI": bmi}, index=uid)
    # 变异对照：重建 X 的一个非缺失单元 +0.1，指纹必须变
    Xm = X.copy(); c0 = "血肌酐"; r0 = Xm.index[Xm[c0].notna()][0]; Xm.loc[r0, c0] += 0.1
    rows = [dict(数据库="NHANES", 文件=f"内存重建 {k}（R51 只读，用于取白蛋白/BMI/eGFR 连续值）", 实测=got[k], 登记=reg[k],
                 方式="按 pgss_r44_nhanes_cohort.build 同一路径内存重建，内容指纹 vs COHORT.json「指纹」", 一致=got[k] == reg[k]) for k in got]
    return extra, pd.DataFrame(rows), bool(sha16df(Xm) != reg["X"])


def cohort(D, db):
    """基线分母 = 合格队列；ICU 的单位是住院（每次住院一段），NHANES 是受访者。"""
    d = D[db]; e = d["M"]["eligible"].astype(bool) if "eligible" in d["M"] else pd.Series(True, index=d["M"].index)
    return e


def female(D, db, e):
    d = D[db]
    if db == "NHANES":
        f = d["Xmodel"]["性别F"][e]; known = f.notna(); fem = (f == 1)
    else:
        s = d["M"]["sex"][e]; known = s.isin(["F", "M"]); fem = s == "F"
    return int(fem[known].sum()), int(known.sum()), int((~known).sum())


# ── T1：基线 ─────────────────────────────────────────────────────
def t1_long(D):
    rows = []
    for db in DBS:
        d = D[db]; e = cohort(D, db); N = int(e.sum())
        nf, nk, nm = female(D, db, e)
        rows.append(dict(数据库=db, 变量="Female", 单位="n (%)", 分母=N, 观测n=nk, 缺失n=nm, 缺失比例=nm / N, 中位数=np.nan, Q1=np.nan, Q3=np.nan, 阳性n=nf, 阳性比例=nf / nk, 小数位=1, 备注="性别缺失或非 F/M 记为缺失"))
        for name, unit, dp, cols in CONT:
            c = cols[db]
            rebuilt = False
            if c is None and db == "NHANES" and D["NHANES"].get("extra_ok"):
                c = NH_EXTRA[name]; rebuilt = True
            if c is None:
                rows.append(dict(数据库=db, 变量=name, 单位=unit, 分母=N, 观测n=np.nan, 缺失n=np.nan, 缺失比例=np.nan, 中位数=np.nan, Q1=np.nan, Q3=np.nan, 阳性n=np.nan, 阳性比例=np.nan, 小数位=dp,
                                 备注="冻结分析文件中没有该连续值（NHANES 只保存了由它判定的表型标志）；未从原始文件重新提取"))
                continue
            v = pd.to_numeric(d["X"][c], errors="coerce")[e]; obs = v.notna()
            q = v[obs].quantile([0.01, 0.25, 0.5, 0.75, 0.99]).to_numpy()
            note = "1999–2000 已按 Selvin 2007 校准；2005–2006 未做官方推荐的校准（见 T6）" if (db == "NHANES" and c == "血肌酐") else ""
            if rebuilt:
                note = ("冻结分析文件只存表型标志；本值按 R44 队列脚本同一路径内存重建取得（X/P/OK/meta 四指纹与登记逐位相同，见 A1b），只用于描述"
                        + ("；NHANES BMI 未设 ICU 的 10–80 范围限制" if c == "BMI" else "")
                        + ("；由校准后血肌酐计算（2005–2006 未校准，见 T6）" if c == "eGFR_landmark" else ""))
            rows.append(dict(数据库=db, 变量=name, 单位=unit, 分母=N, 观测n=int(obs.sum()), 缺失n=int((~obs).sum()), 缺失比例=float((~obs).mean()),
                             中位数=float(q[2]), Q1=float(q[1]), Q3=float(q[3]), P1=float(q[0]), P99=float(q[4]), 阳性n=np.nan, 阳性比例=np.nan, 小数位=dp, 备注=note))
        for ph in PH:
            ok = d["OK"][ph][e].astype(bool); pos = (d["P"][ph][e].astype(bool) & ok)
            rows.append(dict(数据库=db, 变量=f"Phenotype: {PH_EN[ph]}", 单位="n/N ascertainable (%)", 分母=N, 观测n=int(ok.sum()), 缺失n=int((~ok).sum()), 缺失比例=float((~ok).mean()),
                             中位数=np.nan, Q1=np.nan, Q3=np.nan, 阳性n=int(pos.sum()), 阳性比例=float(pos.sum() / ok.sum()), 小数位=1, 备注="不可判定者不计为阴性"))
    return pd.DataFrame(rows)


# ── T2：单位、患者与结局（对 GPT 任务书 §3 表逐值对账）──────────────
GPT_T3 = {("MIMIC-IV", "单位"): 67000, ("MIMIC-IV", "患者"): 53539, ("MIMIC-IV", "死亡事件"): 7193, ("MIMIC-IV", "死亡分母"): 67000, ("MIMIC-IV", "LOS72事件"): 27068, ("MIMIC-IV", "LOS72分母"): 67000,
          ("eICU-CRD", "单位"): 109104, ("eICU-CRD", "患者"): 94920, ("eICU-CRD", "死亡事件"): 9569, ("eICU-CRD", "死亡分母"): 109104, ("eICU-CRD", "LOS72事件"): 39609, ("eICU-CRD", "LOS72分母"): 109104, ("eICU-CRD", "医院"): 207,
          ("NWICU", "单位"): 19965, ("NWICU", "患者"): 17812, ("NWICU", "死亡事件"): 2100, ("NWICU", "死亡分母"): 19765, ("NWICU", "死亡患者"): 17626, ("NWICU", "LOS72事件"): 8146, ("NWICU", "LOS72分母"): 19965,
          ("NHANES", "单位"): 23513, ("NHANES", "患者"): 23513, ("NHANES", "全因死亡事件"): 3813, ("NHANES", "全因死亡分母"): 23513, ("NHANES", "CVD分母"): 23512}


def t2_outcomes(D):
    R = {}
    for db in ("MIMIC-IV", "eICU-CRD"):
        M = D[db]["M"]; e = M.eligible.astype(bool)
        R.update({(db, "单位"): int(e.sum()), (db, "患者"): int(M.group_id[e].nunique()), (db, "死亡事件"): int((e & M.OUT_HOSP_MORT).sum()), (db, "死亡分母"): int(e.sum()),
                  (db, "LOS72事件"): int((e & M.OUT_LOS72).sum()), (db, "LOS72分母"): int(e.sum())})
        if db == "eICU-CRD":
            R[(db, "医院")] = int(M.hospital_id[e].nunique())
    M, S = D["NWICU"]["M"], D["NWICU"]["S"]; e = M.eligible.astype(bool); ev = e & S.住院死亡可评价.astype(bool)
    R.update({("NWICU", "单位"): int(e.sum()), ("NWICU", "患者"): int(M.group_id[e].nunique()), ("NWICU", "死亡事件"): int(S.OUT_HOSP_MORT_v2[ev].sum()),
              ("NWICU", "死亡分母"): int(ev.sum()), ("NWICU", "死亡患者"): int(M.group_id[ev].nunique()), ("NWICU", "LOS72事件"): int((e & M.OUT_LOS72).sum()), ("NWICU", "LOS72分母"): int(e.sum())})
    M, C = D["NHANES"]["M"], D["NHANES"]["S"]; e = M.eligible.astype(bool); cv = e & C.O_CVD10可评价.astype(bool)
    R.update({("NHANES", "单位"): int(e.sum()), ("NHANES", "患者"): int(M.group_id[e].nunique()), ("NHANES", "全因死亡事件"): int((e & M.O_MORT10).sum()),
              ("NHANES", "全因死亡分母"): int(e.sum()), ("NHANES", "CVD分母"): int(cv.sum()), ("NHANES", "CVD事件"): int(C.O_CVD10_v2[cv].sum())})
    rows = [dict(数据库=k[0], 项=k[1], 冻结数据计数=v, GPT任务书表值=GPT_T3.get(k, np.nan), 一致=(GPT_T3[k] == v) if k in GPT_T3 else np.nan) for k, v in R.items()]
    return pd.DataFrame(rows), R


# ── T3：六表型可判定覆盖率（按结局分母），对 R47 B1 / R49 C2 ─────────
def t3_coverage(D):
    b1 = pd.read_excel(X47, sheet_name="B1_数据与任务概况")
    key = {"MIMIC-IV": "mimic", "eICU-CRD": "eicu", "NWICU": "nwicu", "NHANES": "nhanes"}
    rows = []
    for db in DBS:
        d = D[db]; M = d["M"]; e = cohort(D, db)
        if db in ("MIMIC-IV", "eICU-CRD"):
            outs = {"OUT_HOSP_MORT": (e, M.OUT_HOSP_MORT), "OUT_LOS72": (e, M.OUT_LOS72)}
        elif db == "NWICU":
            ev = e & d["S"].住院死亡可评价.astype(bool)
            outs = {"OUT_HOSP_MORT": (ev, d["S"].OUT_HOSP_MORT_v2.fillna(0).astype(bool)), "OUT_LOS72": (e, M.OUT_LOS72)}
        else:
            cv = e & d["S"].O_CVD10可评价.astype(bool)
            outs = {"O_MORT10": (e, M.O_MORT10.astype(bool)), "O_CVD10": (cv, d["S"].O_CVD10_v2.fillna(0).astype(bool))}
        for oc, (den, y) in outs.items():
            for ph in PH:
                ok = d["OK"][ph].astype(bool) & den; A = ok & d["P"][ph].astype(bool); B = ok & ~d["P"][ph].astype(bool)
                r = dict(数据库=db, 结局=oc, 表型=ph, Phenotype=PH_EN[ph], 结局可评价分母=int(den.sum()), 可判定n=int(ok.sum()), 不可判定n=int((den & ~d["OK"][ph].astype(bool)).sum()),
                         不可判定比例=float((den & ~d["OK"][ph].astype(bool)).sum() / den.sum()), A轨n=int(A.sum()), A轨事件=int((A & y).sum()), B轨n=int(B.sum()), B轨事件=int((B & y).sum()))
                h = b1[(b1.库 == key[db]) & (b1.结局 == oc) & (b1.表型 == ph)]
                assert len(h) == 1, (db, oc, ph)
                x = h.iloc[0]
                r["冻结程序纳入"] = bool(x.纳入)
                r.update({f"R47_{c}": int(x[c]) for c in T3_KEYS.values()})
                r["说明"] = "" if r["冻结程序纳入"] else f"列出但不进入分析；R47 B1 来源注：{x.来源}"
                rows.append(r)
    T = pd.DataFrame(rows)
    T["与R47_B1一致"] = t3_cmp(T)
    return T


T3_KEYS = {"可判定n": "n", "A轨n": "A轨n", "A轨事件": "A轨事件", "B轨n": "B轨n", "B轨事件": "B轨事件"}


def t3_cmp(T):
    return np.logical_and.reduce([T[a].to_numpy() == T[f"R47_{b}"].to_numpy() for a, b in T3_KEYS.items()])


# ── T4：单位核实 ─────────────────────────────────────────────────
def t4_units():
    nw = {}
    if NWRAW.exists():
        cnt = {}
        for ch in pd.read_csv(NWRAW / "labevents.csv.gz", usecols=["itemid", "valueuom"], chunksize=2_000_000):
            ch = ch[ch.itemid.isin(NWICU_ITEMS.values())]
            for (i, u), n in ch.groupby(["itemid", ch.valueuom.fillna("（空）")]).size().items():
                cnt[(i, u)] = cnt.get((i, u), 0) + int(n)
        for v, i in NWICU_ITEMS.items():
            nw[v] = "；".join(f"{u}: {n:,}" for (ii, u), n in sorted(cnt.items()) if ii == i)
    vm = pd.read_excel(ROOT / "13_MIMIC_EICU" / "R34_VARMAP_VERIFIED.xlsx", sheet_name="1_映射与核验").set_index("变量")
    # NHANES：读本地 NHANES 变量索引（来自官方 XPT 的 SAS 标签与取值范围），1999–2008 五个周期逐个列出
    idx = pd.read_csv(NHIDX, encoding="utf-8-sig")
    idx = idx[idx.years.isin(["1999-2000", "2001-2002", "2003-2004", "2005-2006", "2007-2008"])]
    def nhlab(vars_):
        z = idx[idx.variable.isin(vars_)].sort_values("years")
        return "；".join(f"{r.years} {r.variable}「{r.label}」[{r['min']}, {r['max']}]" for _, r in z.iterrows())
    nh_src = {"血红蛋白": nhlab(["LBXHGB"]), "血肌酐": nhlab(["LBXSCR", "LBDSCR"]) + "。1999–2000 按 Selvin 2007 校准（1.013×SCr+0.147）；2005–2006 未校准（T6）",
              "RDW": nhlab(["LBXRDW"]), "血小板": nhlab(["LBXPLTSI"]) + "。1999–2002 标签写作「(%) SI」，取值范围与其余周期的 1000 cells/µL 相同，按标签笔误处理",
              "血氯": nhlab(["LBXSCLSI"])}
    rb = "冻结分析文件只存表型标志；Table 1 用内存重建值（A1b 闸门：四指纹逐位相同）。"
    nh_src_extra = {"白蛋白": rb + nhlab(["LBXSAL"]),
                    "BMI": rb + nhlab(["BMXBMI"]) + "；BMXBMI 缺失时由 BMXWT/BMXHT 计算（nhanes.add_derived）；未设 ICU 的 10–80 范围限制",
                    "eGFR_landmark": rb + "nhanes.add_derived 的 CKD-EPI 2021（免种族），输入为校准后血肌酐（2005–2006 未校准，见 T6）"}
    rows = []
    for name, unit, dp, cols in CONT:
        c = cols["MIMIC-IV"]
        if name == "Age":
            ev_icu, ev_nw, comp = "年（MIMIC：anchor_age + 入院年份 − anchor_year；eICU：'> 89' 记 90）", "年", "comparable（NHANES 为体检时年龄；各源均有高龄截顶）"
            ev_nh = "RIDAGEYR：" + "；".join(f"{r.years} 最大 {r['max']:g}" for _, r in idx[idx.variable == "RIDAGEYR"].sort_values("years").iterrows())
        elif c in ("eGFR_landmark", "BMI"):
            ev_icu = "由冻结 phenotypes() 计算：" + ("CKD-EPI 2021（SCr mg/dL、年龄、性别）" if c == "eGFR_landmark" else "体重kg/(身高m)²，限 10–80")
            ev_nw = ev_icu + "；身高、体重在构建时由英寸、盎司换算（pgss_r45_nwicu_build.BODY）" if c == "BMI" else ev_icu
            ev_nh = nh_src_extra[c]
            comp = "comparable（NHANES 未设 10–80 范围限制）" if c == "BMI" else "comparable（NHANES 2005–2006 血肌酐未校准，见 T6）"
        else:
            r = vm.loc[c]
            ev_icu = (f"R34 变量映射核验：{r.单位}，{r.换算说明}；R35 抽取限定 itemid 与生理范围 [{r.生理范围下限:g}, {r.生理范围上限:g}]（SI 量级的值落在范围外）；"
                      "本轮未连库，未重扫单位字段")
            ev_nw = f"NWICU labevents.valueuom 全表实扫（该 itemid 的全部记录）：{nw.get(c, '未扫描（原始数据不在）')}"
            ev_nh = nh_src.get(c) or nh_src_extra[c]
            comp = "comparable"
        rows.append(dict(变量=name, 表中单位=unit, MIMIC与eICU依据=ev_icu, NWICU依据=ev_nw, NHANES依据=ev_nh, 可比性=comp))
    return pd.DataFrame(rows)


# ── T5：补充图 S1 数字核对 ───────────────────────────────────────
def t5_flow(R, fig=FIG, excl=FIG_EXCL, out=FIG_OUT):
    c1 = pd.read_excel(X49, sheet_name="C1_纳排流程")
    nw = pd.read_excel(X48, sheet_name="S2_E1_NWICU_队列流程"); nh = json.loads((NH / "COHORT.json").read_text(encoding="utf-8"))["纳排"]
    a2 = pd.read_excel(X48, sheet_name="S9_A2_纳排流程")
    src = {db: c1[(c1.数据库 == db) & c1.步骤.notna()].sort_values("步骤").住院单位数.astype(int).tolist() for db in ("MIMIC-IV", "eICU-CRD")}
    src["NWICU"] = nw.人次.astype(int).tolist(); src["NHANES"] = [int(x["剩余"]) for x in nh]
    lab = {"MIMIC-IV": "R49 C1_纳排流程", "eICU-CRD": "R49 C1_纳排流程", "NWICU": "R48 S2_E1_NWICU_队列流程（= R45 E1）", "NHANES": "R44 data/COHORT.json 纳排（= R48 S9）"}
    rows = []
    for db in fig:
        rows.append(dict(面板=db, 步骤="步数", 图中数字=len(fig[db]), 来源值=len(src[db]), 来源=lab[db], 一致=len(fig[db]) == len(src[db])))   # 防 zip 截断
        for i, (f, s) in enumerate(zip(fig[db], src[db])):
            ok = f == s and (db != "NHANES" or int(a2.剩余.iloc[i]) == s)
            rows.append(dict(面板=db, 步骤=i + 1, 图中数字=f, 来源值=s, 来源=lab[db], 一致=ok))
        for i, x in enumerate(excl[db]):
            rows.append(dict(面板=db, 步骤=f"排除 {i + 1}→{i + 2}", 图中数字=x, 来源值=src[db][i] - src[db][i + 1], 来源="来源相邻两步之差", 一致=x == src[db][i] - src[db][i + 1]))
        rows.append(dict(面板=db, 步骤="最终（冻结数据直接计数）", 图中数字=fig[db][-1], 来源值=R[(db, "单位")], 来源="本脚本从冻结文件计数", 一致=fig[db][-1] == R[(db, "单位")]))
    for db, k, figv in out:
        rows.append(dict(面板=db, 步骤=f"结局框：{k}", 图中数字=figv, 来源值=R[(db, k)], 来源="本脚本从冻结文件计数", 一致=figv == R[(db, k)]))
    return pd.DataFrame(rows)


# ── T6：NHANES 2005–2006 血肌酐校准（官方建议未执行）的影响，只读计数，不改冻结分析 ──
def ckd_epi_2021(scr, age, female):
    k = np.where(female, 0.7, 0.9); a = np.where(female, -0.241, -0.302); r = scr / k
    return 142.0 * np.minimum(r, 1) ** a * np.maximum(r, 1) ** (-1.200) * 0.9938 ** age * np.where(female, 1.012, 1.0)


def t6_nhanes_scr(D, cut=60.0):
    d = D["NHANES"]; M = d["M"]; e = M.eligible.astype(bool)
    X = d["Xmodel"]; scr = X["血肌酐"].to_numpy(float); age = X["年龄"].to_numpy(float); fem = X["性别F"].to_numpy(float) == 1
    eg = ckd_epi_2021(scr, age, fem); ok = np.isfinite(eg) & e.to_numpy()
    parity = bool(np.array_equal((eg < cut)[ok], d["P"]["eGFR降低"].to_numpy(bool)[ok]) and np.array_equal(ok, d["OK"]["eGFR降低"].to_numpy(bool) & e.to_numpy()))
    c56 = (M.Cycle == "2005-2006").to_numpy()
    scr2 = np.where(c56, -0.016 + 0.978 * scr, scr); eg2 = ckd_epi_2021(scr2, age, fem)
    was, now = (eg < 60) & ok, (eg2 < 60) & ok
    rows = [dict(项="eGFR 降低表型：由冻结血肌酐、年龄、性别重算，与冻结 P/OK 逐个一致", 值=parity),
            dict(项="合格队列中 2005–2006 周期人数", 值=int((c56 & e.to_numpy()).sum())),
            dict(项="其中血肌酐可得", 值=int((c56 & ok).sum())),
            dict(项="2005–2006 血肌酐中位数（冻结，未校准）mg/dL", 值=float(np.median(scr[c56 & ok]))),
            dict(项="2005–2006 血肌酐中位数（按 BIOPRO_D 公式校准后）mg/dL", 值=float(np.median(scr2[c56 & ok]))),
            dict(项="全队列 eGFR 降低（冻结）n", 值=int(was.sum())),
            dict(项="若按公式校准：eGFR 降低 n", 值=int(now.sum())),
            dict(项="由「降低」改判为「不降低」n", 值=int((was & ~now).sum())),
            dict(项="由「不降低」改判为「降低」n", 值=int((~was & now).sum())),
            dict(项="改判占冻结 eGFR 降低（A 轨）的比例", 值=float((was & ~now).sum() / was.sum())),
            dict(项="全队列血肌酐中位数 [Q1, Q3]：冻结", 值=" / ".join(f"{x:.2f}" for x in np.quantile(scr[ok], [0.5, 0.25, 0.75]))),
            dict(项="全队列血肌酐中位数 [Q1, Q3]：若校准", 值=" / ".join(f"{x:.2f}" for x in np.quantile(scr2[ok], [0.5, 0.25, 0.75])))]
    doc = NHDOC.read_text(encoding="utf-8", errors="replace") if NHDOC.exists() else ""
    doc = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", doc)))
    rows.append(dict(项="官方原文（本地存档 BIOPRO_D.htm）含「Correction of serum creatinine values in NHANES 2005-2006 is highly recommended」与「-0.016 + 0.978」",
                     值=bool("Correction of serum creatinine values in NHANES 2005-2006 is highly recommended" in doc and "-0.016 + 0.978" in doc)))
    return pd.DataFrame(rows), parity


# ── T0：直接排版用的 Table 1（宽表）──────────────────────────────
def fmt_row(r):
    if r["变量"] == "Female" or r["变量"].startswith("Phenotype"):
        if pd.isna(r["阳性n"]):
            return "NA"
        return f"{int(r['阳性n']):,}/{int(r['观测n']):,} ({100 * r['阳性比例']:.1f}%)"
    if pd.isna(r["中位数"]):
        return "NA"
    dp = int(r["小数位"])
    return f"{r['中位数']:.{dp}f} [{r['Q1']:.{dp}f}, {r['Q3']:.{dp}f}]"


def t0_table1(L, R):
    roles = {"MIMIC-IV": "ICU development benchmark", "eICU-CRD": "ICU development benchmark (multicentre)", "NWICU": "ICU independent-program replication",
             "NHANES": "Cross-domain program replication"}
    units = {"MIMIC-IV": "ICU stay", "eICU-CRD": "ICU stay", "NWICU": "ICU stay", "NHANES": "Survey participant"}
    T = []
    add = lambda sec, lab, vals: T.append(dict(区块=sec, 行=lab, **vals))
    add("Source", "Role in study", roles)
    add("Source", "Unit of observation", units)
    add("Source", "Eligible units, n", {db: f"{R[(db, '单位')]:,}" for db in DBS})
    add("Source", "Distinct patients, n", {db: f"{R[(db, '患者')]:,}" for db in DBS})
    add("Source", "Hospitals, n", {"MIMIC-IV": "NA (single centre per source documentation; no hospital identifier in frozen files)",
                                   "eICU-CRD": f"{R[('eICU-CRD', '医院')]:,}",
                                   "NWICU": "NA (no hospital identifier in frozen files; confirm from source documentation)", "NHANES": "NA (population survey)"})
    add("Source", "Prediction time", {"MIMIC-IV": "24 h after ICU admission", "eICU-CRD": "24 h after ICU admission", "NWICU": "24 h after ICU admission", "NHANES": "Examination visit"})
    for name, unit, dp, cols in [("Female", "n (%)", 1, None)] + [(c[0], c[1], c[2], c[3]) for c in CONT]:
        vals = {}
        for db in DBS:
            r = L[(L.数据库 == db) & (L.变量 == name)].iloc[0]
            v = fmt_row(r)
            if v != "NA" and name != "Female":
                v += f"; observed {int(r['观测n']):,} ({100 * (1 - r['缺失比例']):.1f}%)"
            elif name == "Female" and v != "NA":
                v += f"; sex unknown {int(r['缺失n']):,}"
            vals[db] = v
        add("Baseline (median [Q1, Q3] unless stated)", f"{name}" + ("" if name == "Female" else f", {unit}"), vals)
    for ph in PH:
        vals = {db: fmt_row(L[(L.数据库 == db) & (L.变量 == f"Phenotype: {PH_EN[ph]}")].iloc[0]) for db in DBS}
        add("Phenotype prevalence among ascertainable units", PH_EN[ph], vals)
    pct = lambda e, n: f"{e:,}/{n:,} ({100 * e / n:.1f}%)"
    add("Outcomes (source-specific; not pooled)", "In-hospital death", {"MIMIC-IV": pct(R[("MIMIC-IV", "死亡事件")], R[("MIMIC-IV", "死亡分母")]),
        "eICU-CRD": pct(R[("eICU-CRD", "死亡事件")], R[("eICU-CRD", "死亡分母")]), "NWICU": pct(R[("NWICU", "死亡事件")], R[("NWICU", "死亡分母")]) + f"; {R[('NWICU', '死亡患者')]:,} patients", "NHANES": "—"})
    add("Outcomes (source-specific; not pooled)", "Total ICU stay >72 h from ICU admission", {"MIMIC-IV": pct(R[("MIMIC-IV", "LOS72事件")], R[("MIMIC-IV", "LOS72分母")]),
        "eICU-CRD": pct(R[("eICU-CRD", "LOS72事件")], R[("eICU-CRD", "LOS72分母")]), "NWICU": pct(R[("NWICU", "LOS72事件")], R[("NWICU", "LOS72分母")]), "NHANES": "—"})
    add("Outcomes (source-specific; not pooled)", "10-year all-cause death", {"MIMIC-IV": "—", "eICU-CRD": "—", "NWICU": "—", "NHANES": pct(R[("NHANES", "全因死亡事件")], R[("NHANES", "全因死亡分母")])})
    add("Outcomes (source-specific; not pooled)", "10-year cardiovascular death", {"MIMIC-IV": "—", "eICU-CRD": "—", "NWICU": "—", "NHANES": pct(R[("NHANES", "CVD事件")], R[("NHANES", "CVD分母")])})
    return pd.DataFrame(T)


# ── 质控与变异对照（每条检查都配一个必须失败的变异；变异只改非缺失单元）──────
FORBID = {"SEQN", "subject_id", "hadm_id", "stay_id", "patientunitstayid", "uniquepid", "group_id", "uid", "sid"}


def no_record_level(sheets):
    return all(len(df) <= 300 and not (set(map(str, df.columns)) & FORBID) for df in sheets.values())


def qc(D, L, T2, R, T3, T5, T6ok, V, sheets, V2, rb_mut):
    rows = []
    def add(chk, ok, det, mut):
        rows.append(dict(检查=chk, 通过=bool(ok), 说明=det, 变异对照检出=bool(mut)))
    # 1 版本
    same = lambda v: bool((v.实测 == v.登记).all())
    Vm = V.copy(); Vm.loc[0, "登记"] = ("0" if Vm.loc[0, "登记"][0] != "0" else "1") + Vm.loc[0, "登记"][1:]
    add("版本：全部冻结文件指纹（实测 = 登记）", same(V), f"{int((V.实测 == V.登记).sum())}/{len(V)}", not same(Vm))
    # 2 T1 恒等式
    ident = lambda l: bool(((l.观测n + l.缺失n) == l.分母)[l.观测n.notna()].all())
    Lm = L.copy(); k = Lm.index[Lm.观测n.notna()][0]; Lm.loc[k, "缺失n"] += 1
    add("T1：每项 观测n + 缺失n = 合格分母", ident(L), f"{int(L.观测n.notna().sum())} 项", not ident(Lm))
    # 3 T2 对 GPT 任务书 §3
    t2ok = lambda r: all(r[k] == v for k, v in GPT_T3.items())
    Rm = dict(R); Rm[("NWICU", "死亡患者")] += 1
    add("T2：与 GPT 任务书 §3 表逐值一致", t2ok(R), f"{len(GPT_T3)}/{len(GPT_T3)} 项", not t2ok(Rm))
    # 4 T3 对 R47 B1
    T3m = T3.copy(); T3m.loc[T3m.index[0], "A轨事件"] += 1
    add("T3：48 个任务的可判定 n、A/B 轨 n 与事件 = R47 B1", bool(t3_cmp(T3).all()) and len(T3) == 48, f"{int(t3_cmp(T3).sum())}/{len(T3)}", not t3_cmp(T3m).all())
    # 5 T3 对 R49 C2
    c2 = pd.read_excel(X49, sheet_name="C2_表型任务")
    m = T3.merge(c2[["数据库", "结局", "表型", "任务n", "表型不可判定"]], on=["数据库", "结局", "表型"])
    c2ok = lambda z: bool(((z.可判定n == z.任务n) & (z.不可判定n == z.表型不可判定)).all())
    mm = m.copy(); mm.loc[mm.index[0], "不可判定n"] += 1
    add("T3：MIMIC/eICU 可判定 n 与不可判定 n = R49 C2", c2ok(m) and len(m) == 24, f"{len(m)} 行", not c2ok(mm))
    # 6 T5 图 S1
    fm = {k: list(v) for k, v in FIG.items()}; fm["NWICU"][4] += 1
    om = tuple((db, kk, v + 1) if kk == "死亡患者" else (db, kk, v) for db, kk, v in FIG_OUT)
    add("T5：补充图 S1 的全部数字与来源一致", bool(T5.一致.all()), f"{int(T5.一致.sum())}/{len(T5)}",
        (not t5_flow(R, fig=fm).一致.all()) and (not t5_flow(R, out=om).一致.all()))
    # 7 NWICU 贫血切点
    nwP = D["NWICU"]["P"]; X = D["NWICU"]["X"]; M = D["NWICU"]["M"]; ok = D["NWICU"]["OK"]["贫血"].astype(bool).to_numpy()
    an = lambda f: bool((nwP["贫血"].to_numpy()[ok] == np.where(M.sex == "F", X["血红蛋白"] < f, X["血红蛋白"] < 13.0)[ok]).all())
    add("NWICU 贫血切点：冻结 P = 「女 <12、男 <13」（构建脚本里的 12.5 只出现在 parity 变异对照中）", an(12.0), f"可判定 {int(ok.sum())}", not an(12.5))
    # 8 NWICU Xfull 与建模矩阵
    Xm0 = D["NWICU"]["Xmodel"]; shared = [c for c in Xm0.columns if c in X.columns]
    eq = lambda A: all(np.array_equal(A[c].to_numpy(float), Xm0[c].to_numpy(float), equal_nan=True) for c in shared)
    Xmut = X.copy(); c0 = shared[0]; r0 = Xmut.index[Xmut[c0].notna()][0]; Xmut.loc[r0, c0] = float(Xmut.loc[r0, c0]) + 0.1
    add("NWICU：基线所读 Xfull 与冻结建模矩阵在共有列逐值相同", eq(X), f"{len(shared)} 列", not eq(Xmut))
    # 9 NHANES eGFR 重算 parity（T6 的前提）
    add("NHANES：由冻结血肌酐/年龄/性别重算 eGFR<60 = 冻结表型 P/OK", T6ok, "CKD-EPI 2021", not t6_nhanes_scr(D, cut=59.0)[1])
    # 10 官方原文定位
    doc = NHDOC.read_text(encoding="utf-8", errors="replace")
    doc = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", doc)))
    has = lambda t: t in doc
    add("NHANES 2005–2006 血肌酐校准：本地官方 codebook 原文含推荐语与公式", has("Correction of serum creatinine values in NHANES 2005-2006 is highly recommended") and has("-0.016 + 0.978"),
        "BIOPRO_D.htm", not has("-0.017 + 0.978"))
    # 11 NHANES 内存重建闸门 + 由重建值重算的表型与冻结 P/OK 一致
    Xn, Pn, On = D["NHANES"]["X"], D["NHANES"]["P"], D["NHANES"]["OK"]
    def flags(cut_alb):
        a = Xn["白蛋白"].to_numpy(float); b = Xn["BMI"].to_numpy(float); g = Xn["eGFR_landmark"].to_numpy(float)
        with np.errstate(invalid="ignore"):
            res = []
            for ph, v, f in (("低白蛋白", a, a < cut_alb), ("肥胖", b, b >= 30), ("eGFR降低", g, g < 60)):
                okk = ~np.isnan(v)
                res.append(np.array_equal(okk, On[ph].to_numpy(bool)) and np.array_equal(f[okk], Pn[ph].to_numpy(bool)[okk]))
        return all(res)
    add("NHANES：内存重建四指纹 = 登记，且由重建的白蛋白/BMI/eGFR 重算三个表型 = 冻结 P/OK", bool(V2.一致.all()) and flags(3.5),
        f"{int(V2.一致.sum())}/{len(V2)} 指纹", rb_mut and not flags(3.4))
    # 12 交付表只含汇总
    sm = dict(sheets); sm["_mut"] = pd.DataFrame({"SEQN": [1]})
    add("交付工作簿只含汇总（每表 ≤300 行、无个人标识列）", no_record_level(sheets), f"{len(sheets)} 张表", not no_record_level(sm))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    D, V = load()
    for db in DBS:
        if not D[db]["ok"]:
            print(f"STOP：{db} 指纹不符，该库暂停")
    extra, V2, rb_mut = nhanes_rebuild(D)
    D["NHANES"]["extra_ok"] = bool(V2.一致.all())
    if D["NHANES"]["extra_ok"]:
        D["NHANES"]["X"] = D["NHANES"]["Xmodel"].join(extra.loc[D["NHANES"]["M"].index])
    else:
        print("STOP（仅 NHANES 白蛋白/BMI/eGFR 三项）：内存重建指纹不符")
    L = t1_long(D)
    T2, R = t2_outcomes(D)
    T3 = t3_coverage(D)
    T4 = t4_units()
    T5 = t5_flow(R)
    T6, T6ok = t6_nhanes_scr(D)
    T0 = t0_table1(L, R)
    note = pd.DataFrame({"说明": [
        "R51 只读提数：四库 Table 1 基线描述统计、结局分母、六表型可判定覆盖率、补充图 S1 核对。只含汇总统计，不含个人记录。",
        "T0 可直接排成三线表（每列一个数据源）；T1 为长表（含观测 n、缺失 n/%、P1/四分位数/P99）；连续变量不插补、不换算单位；来源之间不做 P 值比较。",
        "ICU 基线 = 每次合格 ICU 住院在 24 h landmark 前可得的最后一个值（data v5 窗口规则）；一人多次住院按住院计，独立患者数另报。NHANES = 体检时测量。",
        "NHANES 血肌酐 = 冻结分析值：1999–2000 已按 Selvin 2007 校准；2005–2006 未做官方推荐的校准（T6 给出影响计数，冻结结果不变）。",
        "医院数：只有 eICU-CRD 的冻结文件带医院标识；MIMIC-IV 与 NWICU 写 NA，正文如需写单中心须引数据源文献。",
        "NHANES 的白蛋白、BMI、eGFR：冻结分析文件只存表型标志，本表用 R44 队列脚本同一路径内存重建的值（A1b：X/P/OK/meta 四指纹与登记逐位相同；由重建值重算的三个表型与冻结 P/OK 逐个一致），只用于描述。",
        "T0 的表型区与基线区有两行重复信息（Female、Age ≥65 years），排版时可删其一。"]})
    sheets = {"0_说明": note, "T0_Table1_排版": T0, "T1_基线长表": L, "T2_单位患者与结局对账": T2, "T3_表型可判定覆盖率": T3, "T4_单位核实": T4,
              "T5_补充图S1核对": T5, "T6_NHANES血肌酐校准影响": T6, "A1_版本与指纹": V}
    sheets["A1b_NHANES重建闸门"] = V2
    Q = qc(D, L, T2, R, T3, T5, T6ok, V, sheets, V2, rb_mut)
    sheets["A2_质控"] = Q
    with pd.ExcelWriter(OUT / "R51_Table1_Baseline_Source.xlsx", engine="openpyxl") as xw:
        for k, df in sheets.items():
            df.to_excel(xw, sheet_name=k, index=False)
    rec = dict(版本全部一致=bool((V.实测 == V.登记).all()), 质控=f"{int(Q.通过.sum())}/{len(Q)}", 变异对照=f"{int(Q.变异对照检出.sum())}/{len(Q)}",
               T2一致=f"{int(T2.一致.dropna().astype(bool).sum())}/{int(T2.一致.notna().sum())}", T3与R47一致=f"{int(T3.与R47_B1一致.sum())}/{len(T3)}",
               图S1=f"{int(T5.一致.sum())}/{len(T5)}", NHANES_CVD=f"{R[('NHANES', 'CVD事件')]}/{R[('NHANES', 'CVD分母')]}",
               T6=dict(zip(T6.项, [v if not isinstance(v, (np.floating, float)) else round(float(v), 4) for v in T6.值])), 脚本SHA16=sha16f(__file__))
    (OUT / "R51_BASELINE.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 90)
    print(json.dumps(rec, ensure_ascii=False, indent=1, default=str)); print(Q.to_string()); print(T5[~T5.一致].to_string())
    return rec


if __name__ == "__main__":
    main()
