# -*- coding: utf-8 -*-
r"""R45 NWICU 首次构建·第 2 步：按 data v5 的目标规范落实到 NWICU（GPT 第 49 轮 §6）。只读暂存（pgss_r45_nwicu_stage.py）与 E 盘原始小表。

定位：**冻结学习程序在独立 ICU 数据中的复现**（不是现有 ICU 模型的直接迁移）。本步骤只构建数据与无性能资格表，不拟合任何模型。

队列（与 MIMIC pgss_r35_cohort 同口径）
  每次住院的首段 ICU（按 intime，并列再按 stay_id 最小，计数登记）；landmark = intime + 24 h；landmark 时仍在 ICU（outtime > lm）；
  年龄 = anchor_age + (入院年 − anchor_year)，≥ 18 且已知；landmark 前未死亡；结局来源可靠（见下）。性别 F/M 之外记 NaN（不排除，只影响表型可判定）。
住院死亡（GPT 第 49 轮 §6.2，b 口径）
  死亡出院状态 = discharge_location ∈ {EXPIRED, EXPIRED - HOSPICE}。EXPIRED - HOSPICE 的语义以本地交叉表核实（R45_NWICU_DEATHCHECK）：
  有 deathtime 者绝大多数在出院前后 24 h 内死亡 ⇒ 住院结束时已死亡。只有 HOSPICE / 转临终关怀的活着出院**不算**死亡。
  死亡时点 = deathtime；死亡状态但无 deathtime ⇒ 用 dischtime（与 MIMIC 的 hef=1 & deathtime 缺失同一处理，单独计数）。
  NWICU 的 deathtime 也记录出院后的死亡（实测），**不能只因 deathtime 有值就判住院死亡**。
  结局不确定（排除出队列，分别计数，不静默填零）：
    ① 死亡状态但死亡时点晚于出院 + 1 天（与 MIMIC 同一冲突规则）；
    ② 活着出院的状态但 deathtime ≤ dischtime（状态与死亡时间冲突）；
    ③ discharge_location 缺失。
  hospital_expire_flag 不使用（只在死亡状态且无 deathtime 的记录上为 1，与状态不一致，登记说明）。
  c 口径（deathtime ≤ 出院 + 1 天，不看状态）只作时间一致性的描述交叉表，不作终点。
LOS72：icustays.los > 3（天）。
特征（ICU 23 项池；v5 窗口）：charttime ∈ [intime, lm) 且 storetime < lm；valuenum、charttime、storetime 非空；SQL 同口径的范围 QC（沿用 MIMIC 常量）
  取值三步：最大 charttime → 同 itemid 取最晚 storetime（同键多值取均值）→ 跨 itemid 取算术平均；不跨不同 charttime 平均。
  **平均动脉压为来源适配**：NWICU 无直接测量 MAP 的 itemid ⇒ 在同一设备（动脉线 320050/320051；无创 320179/320180）、同一 charttime 上
  以 (SBP + 2·DBP)/3 计算逐条 MAP（各分量先过范围 QC、各取该时点最晚 storetime 版本），再过 MAP 范围 QC，再走同一三步取值。
  血红蛋白无血气来源（MIMIC 含 50811），碳酸氢盐 = 化学组 Total CO2（100013），血糖 = 化学组 100001 + 血气 100045（同 MIMIC 50931 + 50809）。
  单位：逐 itemid 的单位分布见 R45_NWICU_UNITS.xlsx；各拼写变体为同一单位（mEq/L = mmol/L；K/µL = 10³/µL 等），不需换算；
  非法值（如 9999999 哨兵）由范围 QC 去除并计数。
基线体型：storetime < lm 的合法记录（身高 英寸×2.54 ∈ [100, 230] cm；体重 盎司×0.028349523125 ∈ [25, 300] kg）中 charttime 距 intime 最近者；
  并列依次按 storetime 最早、itemid 最小；仍并列且取值不同 ⇒ 取均值（计数登记）。BMI = kg/m²，∈ [10, 80]。R BMI（300001）不使用，只报告覆盖。
表型与可判定：与 MIMIC pgss_r35_qc.phenotypes **逐行相同**的函数（本文件内副本；在 MIMIC data v5 上与已存表型逐位比对，见 parity）。
分组：subject_id（患者）。
输出（本地，含 stay_id）：27_R45_外部复现与机制核验/nwicu/data/{X,P,OK,meta}.parquet；汇总（无标识）：COHORT.json、R45_NWICU_BUILD.xlsx
"""
import sys, json, pathlib, hashlib, time
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd

SRC = pathlib.Path(r"E:/ICU_外部数据/nwicu-northwestern-icu/0.1.0/data")
R45 = HERE.parent / "27_R45_外部复现与机制核验"
OUT = R45 / "nwicu"; LOC = OUT / "_local_only"; DATA = OUT / "data"
DEAD = ["EXPIRED", "EXPIRED - HOSPICE"]
LAB = {"血肌酐": ([100002], (0.1, 20)), "尿素氮": ([100004], (1, 200)), "血钠": ([100010], (100, 180)),
       "血钾": ([100011], (1.5, 9)), "血氯": ([100012], (70, 140)), "碳酸氢盐": ([100013], (5, 50)),
       "阴离子间隙": ([100040, 100041], (-5, 45)), "血糖": ([100001, 100045], (20, 1000)), "血钙": ([100003], (3, 20)),
       "白蛋白": ([100021], (1, 6)), "白细胞计数": ([100016], (0.1, 100)), "血红蛋白": ([100007], (3, 22)),
       "血细胞比容": ([100006], (10, 70)), "血小板": ([100014], (5, 1500)), "MCV": ([100017], (50, 130)),
       "RDW": ([100008], (8, 40))}
CHART = {"心率": ([320045], (20, 220)), "收缩压": ([320050, 320179], (40, 250)), "舒张压": ([320051, 320180], (10, 160)),
         "呼吸频率": ([320210], (4, 60)), "SpO2": ([320277], (50, 100))}
MAP_SRC = {-1: (320050, 320051), -2: (320179, 320180)}      # 伪 itemid：-1 动脉线、-2 无创
MAP_RANGE = (20, 200)
BODY = {"身高cm": (326707, 2.54, (100.0, 230.0)), "体重kg": (326531, 0.028349523125, (25.0, 300.0))}
FEATS = ['MCV', 'RDW', 'SpO2', '呼吸频率', '尿素氮', '平均动脉压', '年龄', '心率', '收缩压', '白细胞计数', '碳酸氢盐', '舒张压',
         '血小板', '血氯', '血糖', '血红蛋白', '血细胞比容', '血肌酐', '血钙', '血钠', '血钾', '阴离子间隙', '性别F']
OUTCOMES = ["OUT_HOSP_MORT", "OUT_LOS72"]
PHENOS = ["女性", "年龄≥65", "贫血", "eGFR降低", "低白蛋白", "肥胖"]


# ── 与 pgss_r35_qc 逐行相同的表型函数（副本；parity() 在 MIMIC v5 上逐位核对）──────────
def ckd_epi_2021(scr, age, sex):
    scr = np.asarray(scr, float); age = np.asarray(age, float)
    isF = (sex == "F").to_numpy() if hasattr(sex, "to_numpy") else np.asarray(sex) == "F"
    isM = (sex == "M").to_numpy() if hasattr(sex, "to_numpy") else np.asarray(sex) == "M"
    known = isF | isM
    k = np.where(isF, 0.7, 0.9); a = np.where(isF, -0.241, -0.302)
    f = np.where(isF, 1.012, 1.0)
    with np.errstate(all="ignore"):
        r = scr / k
        e = 142.0 * np.minimum(r, 1) ** a * np.maximum(r, 1) ** (-1.200) * 0.9938 ** age * f
    return np.where(known, e, np.nan)


def phenotypes(X, meta):
    sex = meta.sex
    isF = sex == "F"; known = sex.isin(["F", "M"])
    hgb = pd.to_numeric(X.get("血红蛋白"), errors="coerce")
    age = pd.to_numeric(X.get("年龄"), errors="coerce")
    alb = pd.to_numeric(X.get("白蛋白"), errors="coerce")
    scr = pd.to_numeric(X.get("血肌酐"), errors="coerce")
    bmi = pd.to_numeric(X["体重kg"], errors="coerce") / (
        pd.to_numeric(X["身高cm"], errors="coerce") / 100.0) ** 2
    bmi = bmi.where((bmi >= 10) & (bmi <= 80))
    eg = pd.Series(ckd_epi_2021(scr, age, sex), index=X.index)
    P = pd.DataFrame(index=X.index)
    OK = pd.DataFrame(index=X.index)
    P["女性"] = isF;                     OK["女性"] = known
    P["年龄≥65"] = age >= 65;            OK["年龄≥65"] = age.notna()
    P["贫血"] = np.where(isF, hgb < 12.0, hgb < 13.0)
    OK["贫血"] = hgb.notna() & known
    P["eGFR降低"] = eg < 60;              OK["eGFR降低"] = eg.notna()
    P["低白蛋白"] = alb < 3.5;             OK["低白蛋白"] = alb.notna()
    P["肥胖"] = bmi >= 30;                OK["肥胖"] = bmi.notna()
    X = X.copy(); X["BMI"] = bmi; X["eGFR_landmark"] = eg
    return X, P.fillna(False).astype(bool), OK.fillna(False).astype(bool)


def parity():
    """表型函数副本在 MIMIC data v5 上与已存 PHENO/PHENOOK 逐位比对；另做一个变异对照（贫血切点改 12.5 必须检出不同）。"""
    V5 = HERE.parent / "19_R38" / "data_v5"
    X = pd.read_parquet(V5 / "R38_mimic_X.parquet"); meta = pd.read_parquet(V5 / "R38_mimic_META.parquet")
    P0 = pd.read_parquet(V5 / "R38_mimic_PHENO.parquet"); O0 = pd.read_parquet(V5 / "R38_mimic_PHENOOK.parquet")
    _, P, OK = phenotypes(X, meta)
    same = bool(P[P0.columns].equals(P0) and OK[O0.columns].equals(O0))
    Xm = X.copy(); _, Pm, _ = phenotypes(Xm, meta)
    Pm["贫血"] = np.where(meta.sex == "F", X["血红蛋白"] < 12.5, X["血红蛋白"] < 13.0)
    mut = not Pm[P0.columns].fillna(False).astype(bool).equals(P0)
    return dict(MIMIC_v5逐位相同=same, 变异对照检出=bool(mut), 行数=len(P0))


# ── 取值 ─────────────────────────────────────────────────────
def window(df, E, key):
    d = df.merge(E[[key, "stay_id", "intime", "lm"]] if key != "stay_id" else E[["stay_id", "intime", "lm"]], on=key)
    n0 = len(d)
    d = d[d.valuenum.notna() & d.charttime.notna() & d.storetime.notna()]
    d = d[(d.charttime >= d.intime) & (d.charttime < d.lm) & (d.storetime < d.lm)]
    return d.rename(columns={"valuenum": "v", "storetime": "stt"}), n0


def pick3(d, name, stats):
    """三步取值：最大 charttime → 同 itemid 最晚 storetime（同键多值取均值）→ 跨 itemid 均值。"""
    if d.empty:
        stats.append(dict(变量=name, 有值就诊=0)); return pd.Series(dtype=float, name=name)
    mct = d.groupby("stay_id").charttime.transform("max")
    a = d[d.charttime == mct]
    mstt = a.groupby(["stay_id", "itemid"]).stt.transform("max")
    rev = a.groupby(["stay_id", "itemid"]).stt.nunique()
    b = a[a.stt == mstt]
    nv = b.groupby(["stay_id", "itemid"]).v.nunique()
    l = b.groupby(["stay_id", "itemid"]).v.mean()
    nid = l.groupby("stay_id").size()
    stats.append(dict(变量=name, 有值就诊=int(len(nid)), 同charttime多来源聚合=int((nid > 1).sum()),
                      同charttime同itemid多版本=int((rev > 1).groupby("stay_id").any().sum()),
                      键相同但取值不同=int((nv > 1).groupby("stay_id").any().sum())))
    return l.groupby("stay_id").mean().rename(name)


def qc_range(d, rng):
    m = d.v.between(*rng)
    return d[m], int((~m).sum()), int((d.v >= 9_999_000).sum())


def build():
    t0 = time.time(); DATA.mkdir(parents=True, exist_ok=True)
    reg, flow = [], []
    p = pd.read_csv(SRC / "nw_hosp/patients.csv.gz")
    a = pd.read_csv(SRC / "nw_hosp/admissions.csv.gz", parse_dates=["admittime", "dischtime", "deathtime"])
    s = pd.read_csv(SRC / "nw_icu/icustays.csv.gz", parse_dates=["intime", "outtime"])
    flow.append(dict(步骤="ICU 入住记录", 人次=len(s)))
    s = s.sort_values(["hadm_id", "intime", "stay_id"])
    mn = s.groupby("hadm_id").intime.transform("min")
    tie = (s.intime == mn).groupby(s.hadm_id).transform("sum") > 1
    reg.append(dict(项="同一住院首段 ICU 的 intime 并列（按 stay_id 最小打破）", N=int(s[tie].hadm_id.nunique())))
    st = s.drop_duplicates("hadm_id").copy()
    flow.append(dict(步骤="每次住院首段 ICU", 人次=len(st)))
    st["lm"] = st.intime + pd.Timedelta("24h")
    st = st.merge(a, on=["subject_id", "hadm_id"], how="left", validate="one_to_one").merge(p, on="subject_id", how="left", validate="many_to_one")
    st["age"] = st.anchor_age + (st.admittime.dt.year - st.anchor_year)
    st["sex"] = st.gender.astype(str).str.strip().str.upper().map({"F": "F", "M": "M"})
    loc = st.discharge_location
    dead = loc.isin(DEAD); loc_missing = loc.isna(); alive = loc.notna() & ~dead
    st["death_ts"] = st.deathtime.where(dead).fillna(st.dischtime.where(dead))
    c_dead_after = dead & (st.death_ts > st.dischtime + pd.Timedelta("1D"))
    c_alive_died = alive & st.deathtime.notna() & (st.deathtime <= st.dischtime)
    dead_before_lm = dead & (st.death_ts <= st.lm)
    in_icu = st.outtime > st.lm; adult = st.age >= 18
    reg += [dict(项="年龄不明", N=int(st.age.isna().sum())),
            dict(项="性别为 F/M 之外或缺失（不排除；表型不可判定）", N=int(st.sex.isna().sum())),
            dict(项="死亡状态且 deathtime 缺失（用 dischtime 作死亡时点）", N=int((dead & st.deathtime.isna()).sum())),
            dict(项="结局不确定①：死亡状态但死亡时点晚于出院 + 1 天", N=int(c_dead_after.sum())),
            dict(项="结局不确定②：活着出院的状态但 deathtime ≤ dischtime", N=int(c_alive_died.sum())),
            dict(项="结局不确定③：discharge_location 缺失", N=int(loc_missing.sum())),
            dict(项="活着出院且 deathtime 晚于出院（出院后死亡，非事件）", N=int((alive & st.deathtime.notna() & (st.deathtime > st.dischtime)).sum())),
            dict(项="hospital_expire_flag = 1（不使用）", N=int(st.hospital_expire_flag.eq(1).sum())),
            dict(项="hospital_expire_flag = 1 且死亡状态且无 deathtime", N=int((st.hospital_expire_flag.eq(1) & dead & st.deathtime.isna()).sum()))]
    # 死亡口径核对（住院层面，全部 admissions）：deathtime − dischtime 按出院去向
    aa = a.copy(); aa["Δh"] = (aa.deathtime - aa.dischtime).dt.total_seconds() / 3600
    grp = aa.discharge_location.where(aa.discharge_location.isin(DEAD + ["INPATIENT HOSPICE", "HOME WITH HOSPICE", "UNKNOWN"]), "其他活着出院")
    dc = []
    for gname, h in aa.groupby(grp.fillna("缺失")):
        d = h["Δh"].dropna()
        dc.append(dict(出院去向=gname, 住院数=len(h), 有deathtime=int(h.deathtime.notna().sum()),
                       Δ中位_小时=float(d.median()) if len(d) else np.nan, Δ_P5=float(d.quantile(.05)) if len(d) else np.nan,
                       Δ_P95=float(d.quantile(.95)) if len(d) else np.nan, 绝对差不超过24h=int((d.abs() <= 24).sum()),
                       死亡早于出院超24h=int((d < -24).sum()), 死亡晚于出院超24h=int((d > 24).sum()),
                       hospital_expire_flag为1=int(h.hospital_expire_flag.eq(1).sum())))
    DC = pd.DataFrame(dc)
    st["outcome_certain"] = ~(c_dead_after | c_alive_died | loc_missing)
    st["eligible"] = in_icu & adult & st.age.notna() & ~dead_before_lm & st.outcome_certain
    flow += [dict(步骤="landmark 时仍在 ICU", 人次=int(in_icu.sum())),
             dict(步骤="且年龄 ≥ 18 且已知", 人次=int((in_icu & adult & st.age.notna()).sum())),
             dict(步骤="且 landmark 前未死亡", 人次=int((in_icu & adult & st.age.notna() & ~dead_before_lm).sum())),
             dict(步骤="且结局来源可靠（最终合格队列）", 人次=int(st.eligible.sum()))]
    E = st[st.eligible].copy()
    E["OUT_HOSP_MORT"] = E.discharge_location.isin(DEAD) & (E.death_ts > E.lm)
    E["OUT_LOS72"] = E.los > 3.0
    E["HOSP_MORT_c_描述"] = E.deathtime.notna() & (E.deathtime <= E.dischtime + pd.Timedelta("1D")) & (E.deathtime > E.lm)
    xt = pd.crosstab(E.OUT_HOSP_MORT, E["HOSP_MORT_c_描述"]).reset_index()
    dloc = (E.assign(去向=E.discharge_location.where(E.discharge_location.isin(DEAD + ["INPATIENT HOSPICE", "HOME WITH HOSPICE", "UNKNOWN"]), "其他活着出院"))
            .groupby("去向").agg(人次=("stay_id", "size"), 住院死亡事件=("OUT_HOSP_MORT", "sum"), c口径=("HOSP_MORT_c_描述", "sum")).reset_index())
    los_chk = float(np.nanmax(np.abs(E.los - (E.outtime - E.intime).dt.total_seconds() / 86400)))
    # ── 特征 ──
    stats, qc = [], []
    lab = pd.read_parquet(LOC / "stage_lab.parquet", columns=["hadm_id", "itemid", "charttime", "storetime", "valuenum"])
    ch = pd.read_parquet(LOC / "stage_chart.parquet", columns=["stay_id", "itemid", "charttime", "storetime", "valuenum"])
    labw, n0 = window(lab, E, "hadm_id")
    chw, n1 = window(ch, E, "stay_id")
    F = pd.DataFrame(index=pd.Index(E.stay_id.to_numpy(), name="stay_id"))
    for nm, (ids, rng) in LAB.items():
        d0 = labw[labw.itemid.isin(ids)]
        d, nbad, nsent = qc_range(d0, rng)
        qc.append(dict(变量=nm, 来源="labevents", itemid=",".join(map(str, ids)), 窗内记录=len(d0), 范围外剔除=nbad, 其中哨兵9999999=nsent))
        F[nm] = pick3(d, nm, stats).reindex(F.index)
    for nm, (ids, rng) in CHART.items():
        d0 = chw[chw.itemid.isin(ids)]
        d, nbad, nsent = qc_range(d0, rng)
        qc.append(dict(变量=nm, 来源="chartevents", itemid=",".join(map(str, ids)), 窗内记录=len(d0), 范围外剔除=nbad, 其中哨兵9999999=nsent))
        F[nm] = pick3(d, nm, stats).reindex(F.index)
    # MAP：同设备、同 charttime 的逐条计算值
    recs, unpaired = [], 0
    for pid, (sid, did) in MAP_SRC.items():
        comp = {}
        for c, it, rng in (("S", sid, CHART["收缩压"][1]), ("D", did, CHART["舒张压"][1])):
            d = chw[chw.itemid == it]
            d = d[d.v.between(*rng)]
            mx = d.groupby(["stay_id", "charttime"]).stt.transform("max")
            d = d[d.stt == mx]
            comp[c] = d.groupby(["stay_id", "charttime"]).agg(v=("v", "mean"), stt=("stt", "max"))
        j = comp["S"].join(comp["D"], lsuffix="_S", rsuffix="_D", how="outer")
        unpaired += int(j.v_S.isna().sum() + j.v_D.isna().sum())
        j = j.dropna(subset=["v_S", "v_D"])
        r = pd.DataFrame({"v": (j.v_S + 2 * j.v_D) / 3, "stt": np.maximum(j.stt_S.to_numpy(), j.stt_D.to_numpy())}, index=j.index).reset_index()
        r["itemid"] = pid
        recs.append(r)
    MR = pd.concat(recs, ignore_index=True)
    MRq, nbad, _ = qc_range(MR, MAP_RANGE)
    qc.append(dict(变量="平均动脉压（计算值）", 来源="chartevents 同设备 S/D 配对", itemid="320050/320051；320179/320180",
                   窗内记录=len(MR), 范围外剔除=nbad, 其中哨兵9999999=0, 未配对分量=unpaired))
    F["平均动脉压"] = pick3(MRq, "平均动脉压", stats).reindex(F.index)
    # 基线体型
    body = {}
    for col, (it, fac, rng) in BODY.items():
        b = ch[(ch.itemid == it)].merge(E[["stay_id", "intime", "lm"]], on="stay_id")
        b = b[b.valuenum.notna() & b.charttime.notna() & b.storetime.notna() & (b.storetime < b.lm)].copy()
        b["v"] = b.valuenum * fac
        nbad = int((~b.v.between(*rng)).sum()); b = b[b.v.between(*rng)]
        b["dist"] = (b.charttime - b.intime).abs()
        b = b.sort_values(["stay_id", "dist", "storetime", "itemid"])
        first = b.groupby("stay_id").head(1)[["stay_id", "dist", "storetime", "itemid"]]
        tied = b.merge(first, on=["stay_id", "dist", "storetime", "itemid"])
        nt = tied.groupby("stay_id").v.nunique()
        body[col] = tied.groupby("stay_id").v.mean()
        qc.append(dict(变量=col, 来源="chartevents 基线体型", itemid=str(it), 窗内记录=len(b) + nbad, 范围外剔除=nbad, 其中哨兵9999999=0,
                       仍并列且取值不同=int((nt > 1).sum())))
        F[col] = body[col].reindex(F.index)
    # 缺失模式：体征是「整段入住无任何体征记录」还是「窗内无记录」；按首个 ICU 类型与锚定年份组
    vit_ids = [320045, 320050, 320051, 320179, 320180, 320210, 320277]
    any_vit = set(ch[ch.itemid.isin(vit_ids)].stay_id.unique())
    Em = E[["stay_id", "first_careunit", "anchor_year_group", "OUT_HOSP_MORT", "OUT_LOS72"]].copy()
    Em["窗内有体征"] = Em.stay_id.map(F["心率"].notna())
    Em["整段入住有任何体征记录"] = Em.stay_id.isin(any_vit)
    MP = []
    for by in ("first_careunit", "anchor_year_group"):
        for k, h in Em.groupby(by):
            MP.append(dict(分组方式=by, 组=k, 人次=len(h), 窗内有体征=float(h.窗内有体征.mean()),
                           整段无任何体征记录=float((~h.整段入住有任何体征记录).mean()),
                           住院死亡率_窗内有体征=float(h[h.窗内有体征].OUT_HOSP_MORT.mean()) if h.窗内有体征.any() else np.nan,
                           住院死亡率_窗内无体征=float(h[~h.窗内有体征].OUT_HOSP_MORT.mean()) if (~h.窗内有体征).any() else np.nan))
    MP.append(dict(分组方式="全部", 组="全部", 人次=len(Em), 窗内有体征=float(Em.窗内有体征.mean()),
                   整段无任何体征记录=float((~Em.整段入住有任何体征记录).mean()),
                   住院死亡率_窗内有体征=float(Em[Em.窗内有体征].OUT_HOSP_MORT.mean()),
                   住院死亡率_窗内无体征=float(Em[~Em.窗内有体征].OUT_HOSP_MORT.mean())))
    MPT = pd.DataFrame(MP)
    rb = ch[(ch.itemid == 300001)].merge(E[["stay_id", "lm"]], on="stay_id")
    rbmi_cov = float(rb[rb.valuenum.notna() & (rb.storetime < rb.lm)].stay_id.nunique() / len(E))
    Ei = E.set_index("stay_id")
    F["年龄"] = Ei.age.reindex(F.index).astype(float)
    meta = pd.DataFrame({"sex": Ei.sex, "eligible": True, "group_id": Ei.subject_id.astype(str), "hospital_id": "NWICU",
                         "OUT_HOSP_MORT": Ei.OUT_HOSP_MORT, "OUT_LOS72": Ei.OUT_LOS72,
                         "discharge_location": Ei.discharge_location, "los": Ei.los}).reindex(F.index)
    Xfull, P, OK = phenotypes(F, meta)
    X = Xfull.copy()
    X["性别F"] = (meta.sex == "F").where(meta.sex.isin(["F", "M"]))
    X = X[FEATS].astype(float)
    for nm, df in (("X", X), ("P", P), ("OK", OK), ("meta", meta), ("Xfull", Xfull)):
        df.to_parquet(DATA / f"{nm}.parquet")
    fp = {nm: hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[:16]
          for nm, df in (("X", X), ("P", P), ("OK", OK), ("meta", meta))}
    # 无性能资格表（不拟合）
    tk = []
    for oc in OUTCOMES:
        y = meta[oc].to_numpy(bool)
        for ph in PHENOS:
            m = OK[ph].to_numpy(bool); z = P[ph].to_numpy(bool) & m
            tk.append(dict(结局=oc, 表型=ph, 可判定人次=int(m.sum()), 唯一患者=int(meta.group_id[m].nunique()),
                           A轨人次=int(z.sum()), A轨比例=float(z[m].mean()), 事件=int(y[m].sum()), A轨事件=int(y[z].sum()),
                           B轨事件=int(y[m & ~z].sum()), A轨非事件=int((~y[z]).sum()), B轨非事件=int((~y[m & ~z]).sum())))
    TK = pd.DataFrame(tk)
    cov_rows = []
    for c in [f for f in FEATS if f != "性别F"] + ["白蛋白", "身高cm", "体重kg", "BMI"]:
        v = Xfull[c]
        cov_rows.append(dict(变量=c, 有值比例=float(v.notna().mean()), P1=float(v.quantile(.01)), 中位数=float(v.median()),
                             P99=float(v.quantile(.99))))
    cov_rows.append(dict(变量="性别F", 有值比例=float(X["性别F"].notna().mean())))
    cov = pd.DataFrame(cov_rows)
    par = parity()
    summ = dict(数据="NWICU 0.1.0（PhysioNet 认证访问；源文件 SHA256 与官方清单一致，见 R45_NWICU_STAGE.json）",
                定位="冻结学习程序在独立 ICU 数据中的复现（不是 ICU 模型直接迁移）",
                最终人次=len(X), 唯一患者=int(meta.group_id.nunique()),
                结局事件={oc: int(meta[oc].sum()) for oc in OUTCOMES}, 结局事件率={oc: float(meta[oc].mean()) for oc in OUTCOMES},
                住院死亡_b与c口径交叉=xt.to_dict("records"), los字段与出入时间差最大差_天=los_chk,
                R_BMI覆盖_仅描述=rbmi_cov, 表型函数parity=par, 指纹=fp,
                代码SHA=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()[:16], 秒=round(time.time() - t0))
    with pd.ExcelWriter(OUT / "R45_NWICU_BUILD.xlsx", engine="openpyxl") as xw:
        pd.DataFrame([{k: (json.dumps(v, ensure_ascii=False, default=str) if isinstance(v, (dict, list)) else v)
                       for k, v in summ.items()}]).T.reset_index().to_excel(xw, sheet_name="0_汇总", index=False)
        pd.DataFrame(flow).to_excel(xw, sheet_name="1_队列流程", index=False)
        pd.DataFrame(reg).to_excel(xw, sheet_name="2_结局不确定与登记", index=False)
        dloc.to_excel(xw, sheet_name="3_出院去向与事件", index=False)
        pd.DataFrame(qc).to_excel(xw, sheet_name="4_范围QC与非法值", index=False)
        pd.DataFrame(stats).to_excel(xw, sheet_name="5_三步取值统计", index=False)
        cov.to_excel(xw, sheet_name="6_特征覆盖", index=False)
        TK.to_excel(xw, sheet_name="7_候选任务无性能资格", index=False)
        DC.to_excel(xw, sheet_name="8_死亡口径核对_住院层面", index=False)
        MPT.to_excel(xw, sheet_name="9_体征缺失模式", index=False)
    (DATA / "COHORT.json").write_text(json.dumps(dict(summ, 队列流程=flow, 登记=reg, 候选任务=TK.to_dict("records")),
                                                 ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(json.dumps(summ, ensure_ascii=False, indent=1, default=str))
    for t in (pd.DataFrame(flow), pd.DataFrame(reg), dloc, pd.DataFrame(qc), pd.DataFrame(stats), cov, TK, DC, MPT):
        print(t.to_string())


if __name__ == "__main__":
    build()
