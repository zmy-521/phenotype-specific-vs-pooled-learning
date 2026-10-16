# -*- coding: utf-8 -*-
"""R35 §C：修正版队列与**两个合格主结局**（IMPACT_LIST #3/#4/#5/#10）。

结局恢复为第 33 轮 registry 已授权的口径
    OUT_HOSP_MORT  **住院死亡**：landmark(入室24h) 之后、同次住院内死亡
                   MIMIC `hospital_expire_flag=1` 且死亡时点 > landmark
                   eICU `hospitaldischargestatus='Expired'` 且 `hospitaldischargeoffset > 1440`
    OUT_LOS72      **ICU 停留 > 72 小时**
                   MIMIC `icustays.los > 3`（天）  eICU `unitdischargeoffset > 4320`（分钟）
    （R34 用的 ICU 内死亡与 >7 天留在旧版本，不混入新主结果）

队列资格显式执行（#5）
    成人（≥18）；该次住院的首段 ICU；landmark 时仍在该 ICU 且无已知的此前死亡；结局有可靠来源。
    **年龄不明 / 出院状态缺失 / 死亡时间冲突分别登记**，不自动当作成年或存活。

通气与 AKI 本轮**不在此脚本产出标签**（降为待核验分支，见 pgss_r35_labeldoc.py）。

用法： $env:PGPASSWORD="..."; python pgss_r35_cohort.py
"""
import os, pathlib, sys, time, warnings
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pandas as pd, psycopg2
OUT = pathlib.Path(__file__).parent.parent / "16_R35_FIX"; OUT.mkdir(exist_ok=True)
CONN = dict(host=os.environ.get("PGHOST", "127.0.0.1"), port=int(os.environ.get("PGPORT", "5431")),
            user=os.environ.get("PGUSER", "postgres"), password=os.environ.get("PGPASSWORD"))
if not CONN["password"]: raise SystemExit("未设置 PGPASSWORD")

MIMIC = """
with first_icu as (
  select distinct on (hadm_id) stay_id, hadm_id, subject_id, intime, outtime, los
  from mimiciv_icu.icustays order by hadm_id, intime),
 st as (select f.*, f.intime + interval '24 hour' lm from first_icu f),
 a as (select hadm_id, admittime, dischtime, deathtime, hospital_expire_flag from mimiciv_hosp.admissions),
 p as (select subject_id, gender, anchor_age, anchor_year, dod from mimiciv_hosp.patients)
select st.stay_id, st.subject_id, st.hadm_id, st.intime, st.outtime, st.lm, st.los los_d,
       p.gender gender_raw,
       p.anchor_age + (extract(year from a.admittime) - p.anchor_year) age,
       a.hospital_expire_flag hef, a.deathtime, a.dischtime,
       -- 死亡时点：优先 deathtime，缺失时用 dischtime（仅当 hef=1）
       coalesce(a.deathtime, case when a.hospital_expire_flag = 1 then a.dischtime end) death_ts,
       (st.outtime > st.lm) in_icu_at_lm
from st join a on a.hadm_id = st.hadm_id join p on p.subject_id = st.subject_id;
"""

EICU = """
select p.patientunitstayid sid, p.patienthealthsystemstayid phs, p.uniquepid, p.hospitalid,
       p.gender gender_raw, p.age age_raw,
       p.unitdischargeoffset udo, p.unitdischargestatus uds,
       p.hospitaldischargeoffset hdo, p.hospitaldischargestatus hds,
       p.admissionheight height_raw, p.admissionweight weight_raw,
       (p.unitdischargeoffset > 1440) in_icu_at_lm
from eicu_crd.patient p where p.unitvisitnumber = 1;
"""


def q(db, sql):
    t = time.time()
    with psycopg2.connect(dbname=db, **CONN) as c:
        df = pd.read_sql(sql, c)
    print(f"  {len(df):,} 行  {time.time()-t:.0f}s", flush=True)
    return df


def build_mimic(d):
    reg = []
    d["age"] = pd.to_numeric(d.age, errors="coerce")
    g = d.gender_raw.astype(str).str.strip().str.upper()
    d["sex"] = g.map({"F": "F", "M": "M"})                    # 其他一律 NaN（#20）
    d["death_ts"] = pd.to_datetime(d.death_ts, errors="coerce")
    d["lm"] = pd.to_datetime(d.lm); d["outtime"] = pd.to_datetime(d.outtime)
    d["dischtime"] = pd.to_datetime(d.dischtime)
    # 分别登记（#5）
    reg.append(dict(项="年龄不明", N=int(d.age.isna().sum())))
    reg.append(dict(项="性别为 F/M 之外或缺失", N=int(d.sex.isna().sum())))
    # deathtime 本身缺失、被 dischtime 替代的例数必须单独登记（不能因 coalesce 而显示为 0）
    dt_raw = pd.to_datetime(d.deathtime, errors="coerce")
    reg.append(dict(项="hef=1 且 deathtime 缺失（已用 dischtime 替代，死亡时点为近似）",
                    N=int(((d.hef == 1) & dt_raw.isna()).sum())))
    reg.append(dict(项="hef=1 且替代后仍无死亡时点（结局来源不可靠，排除）",
                    N=int(((d.hef == 1) & d.death_ts.isna()).sum())))
    reg.append(dict(项="死亡时点早于 landmark（landmark 前已死亡）",
                    N=int(((d.hef == 1) & d.death_ts.notna() & (d.death_ts <= d.lm)).sum())))
    conflict = ((d.hef == 1) & d.death_ts.notna()
                & (d.death_ts > d.dischtime + pd.Timedelta("1D")))
    reg.append(dict(项="死亡时点晚于出院时间超1天（时间冲突）", N=int(conflict.sum())))
    adult = d.age >= 18
    alive_lm = ~((d.hef == 1) & d.death_ts.notna() & (d.death_ts <= d.lm))
    d["eligible"] = (d.in_icu_at_lm.fillna(False) & adult & alive_lm
                     & d.age.notna() & ~conflict)
    d["OUT_HOSP_MORT"] = ((d.hef == 1) & d.death_ts.notna() & (d.death_ts > d.lm))
    d["OUT_LOS72"] = pd.to_numeric(d.los_d, errors="coerce") > 3.0
    d["mort_source_ok"] = ~((d.hef == 1) & d.death_ts.isna())   # 结局来源可靠
    return d, pd.DataFrame(reg)


def build_eicu(d):
    reg = []
    a = d.age_raw.astype(str).str.strip()
    d["age"] = pd.to_numeric(a.where(~a.str.startswith(">"), "90"), errors="coerce")
    g = d.gender_raw.astype(str).str.strip().str.lower()
    d["sex"] = g.map({"female": "F", "male": "M"})              # Unknown/Other/NULL → NaN（#20）
    reg.append(dict(项="年龄不明", N=int(d.age.isna().sum())))
    reg.append(dict(项="性别为 Male/Female 之外或缺失", N=int(d.sex.isna().sum())))
    reg.append(dict(项="住院出院状态缺失/为空",
                    N=int(d.hds.isna().sum() + (d.hds.astype(str).str.strip() == "").sum())))
    reg.append(dict(项="住院出院偏移缺失", N=int(d.hdo.isna().sum())))
    reg.append(dict(项="住院出院偏移 ≤ landmark 但状态为 Expired（landmark 前已死亡）",
                    N=int(((d.hds == "Expired") & (pd.to_numeric(d.hdo, errors="coerce")
                                                   <= 1440)).sum())))
    hdo = pd.to_numeric(d.hdo, errors="coerce")
    adult = d.age >= 18
    alive_lm = ~((d.hds == "Expired") & (hdo <= 1440))
    hds_ok = d.hds.notna() & (d.hds.astype(str).str.strip() != "")
    d["eligible"] = d.in_icu_at_lm.fillna(False) & adult & alive_lm & d.age.notna() & hds_ok
    d["OUT_HOSP_MORT"] = (d.hds == "Expired") & (hdo > 1440)
    d["OUT_LOS72"] = pd.to_numeric(d.udo, errors="coerce") > 4320
    d["mort_source_ok"] = hds_ok
    return d, pd.DataFrame(reg)


def summarize(d, name, grp):
    e = d.eligible & d.mort_source_ok
    rows = [dict(数据库=name, 项="分析单位（每次住院首段ICU / unitvisit=1）", N=int(len(d))),
            dict(数据库=name, 项="landmark(24h) 仍在 ICU", N=int(d.in_icu_at_lm.fillna(False).sum())),
            dict(数据库=name, 项="＋成人且年龄已知", N=int((d.in_icu_at_lm.fillna(False)
                                                      & (d.age >= 18)).sum())),
            dict(数据库=name, 项="＋landmark 前无已知死亡 ＋结局来源可靠（**最终合格队列**）",
                 N=int(e.sum())),
            dict(数据库=name, 项="最终合格队列的唯一患者数", N=int(d.loc[e, grp].nunique()))]
    for oc, lab in (("OUT_HOSP_MORT", "OUT_HOSP_MORT 住院死亡(landmark后)"),
                    ("OUT_LOS72", "OUT_LOS72 ICU停留>72h")):
        n = int(e.sum()); ev = int((e & d[oc]).sum())
        rows.append(dict(数据库=name, 项=lab, N=n, 事件=ev,
                         事件率=round(ev / n, 5) if n else None))
    return pd.DataFrame(rows)


def main():
    print("取 MIMIC 队列 …", flush=True); dm = q("mimic", MIMIC)
    print("取 eICU 队列 …", flush=True); de = q("eicu", EICU)
    dm, rm = build_mimic(dm); de, re_ = build_eicu(de)
    sm = summarize(dm, "MIMIC-IV", "subject_id"); se = summarize(de, "eICU-CRD", "uniquepid")
    S = pd.concat([sm, se], ignore_index=True)
    R = pd.concat([rm.assign(数据库="MIMIC-IV"), re_.assign(数据库="eICU-CRD")], ignore_index=True)
    note = pd.DataFrame([
        dict(项="已删除的判据", 说明="R34 的「事件率比值 ≤2 即两库语义可比」已删除（#10）。"
                             "事件率比只作描述；语义可比由定义/来源字段/时间窗/分母/缺失处理支撑"),
        dict(项="结局口径来源", 说明="第33轮 registry：hospital mortality 判定合格；"
                               "prolonged ICU stay 定义为 >3 天（eICU >4320 min）"),
        dict(项="R34 偏差", 说明="R34 实际用了 ICU 内死亡与 los>7 天，属静默改变研究任务；"
                            "旧结果留档，不混入新主结果"),
        dict(项="未知状态处理", 说明="年龄不明、性别非 F/M、出院状态缺失、死亡时间冲突分别登记并排除出合格队列，"
                               "不自动当作成年或存活"),
        dict(项="本轮不产出", 说明="新发有创通气、新发 AKI 降为待核验标签分支，不在此产出标签")])
    with pd.ExcelWriter(OUT / "R35_COHORT_OUTCOMES.xlsx", engine="openpyxl") as xw:
        S.to_excel(xw, sheet_name="1_队列与结局", index=False)
        R.to_excel(xw, sheet_name="2_未知与冲突登记", index=False)
        note.to_excel(xw, sheet_name="3_口径说明", index=False)
    keep_m = ["stay_id", "subject_id", "hadm_id", "age", "sex", "eligible", "mort_source_ok",
              "OUT_HOSP_MORT", "OUT_LOS72", "los_d"]
    keep_e = ["sid", "uniquepid", "phs", "hospitalid", "age", "sex", "eligible", "mort_source_ok",
              "OUT_HOSP_MORT", "OUT_LOS72", "height_raw", "weight_raw"]
    dm[keep_m].to_parquet(OUT / "R35_MIMIC_COHORT.parquet", index=False)
    de[keep_e].to_parquet(OUT / "R35_EICU_COHORT.parquet", index=False)
    print("\n" + S.to_string(index=False))
    print("\n=== 未知与冲突登记 ===\n" + R.to_string(index=False))
    print(f"\n已写 {OUT/'R35_COHORT_OUTCOMES.xlsx'}")


if __name__ == "__main__":
    main()
