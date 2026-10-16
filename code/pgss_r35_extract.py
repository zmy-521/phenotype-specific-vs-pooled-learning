# -*- coding: utf-8 -*-
"""R35 §C：修正版特征抽取（IMPACT_LIST #21/#22/#23）。

三处修正
  #21 **时间资格用 `storetime`**（结果进入系统、临床可获得的时间），不再只用 `charttime`。
      实测 `labevents.storetime` 完整度 100.00%（200万行抽样缺3行），中位滞后 1.38 小时。
      `chartevents` 同样有 `storetime`。eICU 用 `labresultoffset`（该库无独立的采样/可得二分字段），
      并在 QC 表里标注该差异。
  #22 **先判有效再取末值**：生理范围过滤放进 SQL，再按时间取最后一条
      （R34 是先取末值、后把超范围置缺失 ⇒ 末值异常就变成缺失）。
      多 itemid 映射到同一指标时，**比较真实时间**取最晚的一条，而非 pivot 任取。
  #23 **不再使用未校验的 derived 表**：身高体重改从 raw `chartevents` 自建并显式换算
      （226730 cm / 226707 Inch×2.54 / 226512 kg / 226531 lbs×0.4536 / 224639 Daily kg）。

用法： $env:PGPASSWORD="..."; python pgss_r35_extract.py --db mimic|eicu
"""
import os, pathlib, sys, time, warnings
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pandas as pd, psycopg2
OUT = pathlib.Path(__file__).parent.parent / "16_R35_FIX" / "raw"
OUT.mkdir(parents=True, exist_ok=True)
CONN = dict(host=os.environ.get("PGHOST", "127.0.0.1"), port=int(os.environ.get("PGPORT", "5431")),
            user=os.environ.get("PGUSER", "postgres"), password=os.environ.get("PGPASSWORD"))
if not CONN["password"]: raise SystemExit("未设置 PGPASSWORD")
DB = sys.argv[sys.argv.index("--db") + 1] if "--db" in sys.argv else "mimic"

# 变量 → (MIMIC itemid 列表, eICU labname 列表, 生理范围)
#   itemid 已按 labevents 实测量核验；范围用于 SQL 内过滤（#22）
LAB = {
    "血肌酐": ([50912], ["creatinine"], (0.1, 20)),
    "尿素氮": ([51006], ["BUN"], (1, 200)),
    "血钠": ([50983], ["sodium"], (100, 180)),
    "血钾": ([50971], ["potassium"], (1.5, 9)),
    "血氯": ([50902], ["chloride"], (70, 140)),
    "碳酸氢盐": ([50882], ["bicarbonate"], (5, 50)),
    "阴离子间隙": ([50868], ["anion gap"], (-5, 45)),
    "血糖": ([50931, 50809], ["glucose"], (20, 1000)),
    "血钙": ([50893], ["calcium"], (3, 20)),
    "血磷": ([50970], ["phosphate"], (0.5, 15)),
    "血镁": ([50960], ["magnesium"], (0.5, 6)),
    "白蛋白": ([50862], ["albumin"], (1, 6)),
    "总胆红素": ([50885], ["total bilirubin"], (0.1, 50)),
    "ALT": ([50861], ["ALT (SGPT)"], (1, 5000)),
    "AST": ([50878], ["AST (SGOT)"], (1, 5000)),
    "碱性磷酸酶": ([50863], ["alkaline phos."], (10, 2000)),
    "白细胞计数": ([51301], ["WBC x 1000"], (0.1, 100)),
    "血红蛋白": ([51222, 50811], ["Hgb"], (3, 22)),
    "血细胞比容": ([51221], ["Hct"], (10, 70)),
    "血小板": ([51265], ["platelets x 1000"], (5, 1500)),
    "MCV": ([51250], ["MCV"], (50, 130)),
    "RDW": ([51277], ["RDW"], (8, 40)),
    "中性粒细胞比例": ([51256], ["polys"], (0, 100)),
    "淋巴细胞比例": ([51244], ["lymphs"], (0, 100)),
    "INR": ([51237], ["PT - INR"], (0.5, 15)),
    "PT": ([51274], ["PT"], (8, 150)),
    "PTT": ([51275], ["PTT"], (15, 250)),
    "乳酸": ([50813], ["lactate"], (0.2, 30)),
    "动脉pH": ([50820], ["pH"], (6.5, 7.8)),
    "PaO2": ([50821], ["paO2"], (20, 700)),
    "PaCO2": ([50818], ["paCO2"], (10, 150)),
    "碱剩余": ([50802], ["Base Excess"], (-30, 30)),
}
CHART = {                      # MIMIC chartevents：itemid 列表 + 范围
    "心率": ([220045], (20, 220)),
    "收缩压": ([220050, 220179], (40, 250)),
    "舒张压": ([220051, 220180], (10, 160)),
    "平均动脉压": ([220052, 220181], (20, 200)),
    "呼吸频率": ([220210], (4, 60)),
    "SpO2": ([220277], (50, 100)),
}
E_VITAL = {                    # eICU：(表, 列, 范围)
    "心率": [("vitalperiodic", "heartrate")], "呼吸频率": [("vitalperiodic", "respiration")],
    "SpO2": [("vitalperiodic", "sao2")],
    "收缩压": [("vitalperiodic", "systemicsystolic"), ("vitalaperiodic", "noninvasivesystolic")],
    "舒张压": [("vitalperiodic", "systemicdiastolic"), ("vitalaperiodic", "noninvasivediastolic")],
    "平均动脉压": [("vitalperiodic", "systemicmean"), ("vitalaperiodic", "noninvasivemean")],
}
VRANGE = {k: v[1] for k, v in CHART.items()}
# 体温：MIMIC 223762 已是 °C；223761 为 °F 需换算（#22 的单位先统一）
TEMP = {"C": 223762, "F": 223761}
BODY = {"height_cm": [(226730, 1.0), (226707, 2.54)],
        "weight_kg": [(226512, 1.0), (226531, 0.45359237), (224639, 1.0)]}


def q(db, sql):
    t = time.time()
    with psycopg2.connect(dbname=db, **CONN) as c:
        df = pd.read_sql(sql, c)
    print(f"    {len(df):,} 行  {time.time()-t:.0f}s", flush=True)
    return df


def save(n, df):
    df.to_parquet(OUT / f"{n}.parquet", index=False)
    print(f"    已存 {n}.parquet", flush=True)


ST = """with first_icu as (
  select distinct on (hadm_id) stay_id, hadm_id, intime, intime + interval '24 hour' lm
  from mimiciv_icu.icustays order by hadm_id, intime)"""


def mimic():
    # 化验：storetime 做时间资格 + SQL 内生理范围过滤 + 跨 itemid 比真实时间取最晚
    cases = "\n".join(
        f"      when le.itemid in ({','.join(map(str, ids))}) and le.valuenum between {lo} and {hi} "
        f"then '{name}'" for name, (ids, _, (lo, hi)) in LAB.items())
    allids = sorted({i for v in LAB.values() for i in v[0]})
    print("  [1/4] MIMIC 化验（storetime + 范围内过滤）…", flush=True)
    save("mimic_lab", q("mimic", f"""
{ST}
select distinct on (x.stay_id, x.var) x.stay_id, x.var, x.valuenum
from (select st.stay_id, le.valuenum, le.storetime,
             case
{cases}
             end var
      from first_icu st
      join mimiciv_hosp.labevents le on le.hadm_id = st.hadm_id
      where le.itemid in ({','.join(map(str, allids))}) and le.valuenum is not null
        and le.storetime is not null
        and le.storetime >= st.intime and le.storetime < st.lm) x
where x.var is not null
order by x.stay_id, x.var, x.storetime desc;"""))

    ccases = "\n".join(
        f"      when ce.itemid in ({','.join(map(str, ids))}) and ce.valuenum between {lo} and {hi} "
        f"then '{name}'" for name, (ids, (lo, hi)) in CHART.items())
    cids = sorted({i for v in CHART.values() for i in v[0]})
    print("  [2/4] MIMIC 体征（大表）…", flush=True)
    save("mimic_chart", q("mimic", f"""
{ST}
select distinct on (x.stay_id, x.var) x.stay_id, x.var, x.valuenum
from (select st.stay_id, ce.valuenum, ce.storetime,
             case
{ccases}
             end var
      from first_icu st
      join mimiciv_icu.chartevents ce on ce.stay_id = st.stay_id
      where ce.itemid in ({','.join(map(str, cids))}) and ce.valuenum is not null
        and ce.storetime is not null
        and ce.storetime >= st.intime and ce.storetime < st.lm) x
where x.var is not null
order by x.stay_id, x.var, x.storetime desc;"""))

    print("  [3/4] MIMIC 体温（F→C 先换算再过滤）…", flush=True)
    save("mimic_temp", q("mimic", f"""
{ST}
select distinct on (x.stay_id) x.stay_id, x.tc
from (select st.stay_id, ce.storetime,
             case when ce.itemid = {TEMP['C']} then ce.valuenum
                  else (ce.valuenum - 32.0) * 5.0 / 9.0 end tc
      from first_icu st
      join mimiciv_icu.chartevents ce on ce.stay_id = st.stay_id
      where ce.itemid in ({TEMP['C']},{TEMP['F']}) and ce.valuenum is not null
        and ce.storetime is not null
        and ce.storetime >= st.intime and ce.storetime < st.lm) x
where x.tc between 30 and 43
order by x.stay_id, x.storetime desc;"""))

    hcase = " ".join(f"when ce.itemid = {i} then ce.valuenum * {f}" for i, f in BODY["height_cm"])
    wcase = " ".join(f"when ce.itemid = {i} then ce.valuenum * {f}" for i, f in BODY["weight_kg"])
    hids = ",".join(str(i) for i, _ in BODY["height_cm"])
    wids = ",".join(str(i) for i, _ in BODY["weight_kg"])
    print("  [4/4] MIMIC 身高体重（raw chartevents，不用 derived）…", flush=True)
    save("mimic_body", q("mimic", f"""
{ST}
, h as (select distinct on (x.stay_id) x.stay_id, x.v height_cm from
   (select st.stay_id, ce.storetime, case {hcase} end v
    from first_icu st join mimiciv_icu.chartevents ce on ce.stay_id = st.stay_id
    where ce.itemid in ({hids}) and ce.valuenum is not null and ce.storetime < st.lm) x
   where x.v between 100 and 230 order by x.stay_id, x.storetime desc)
, w as (select distinct on (x.stay_id) x.stay_id, x.v weight_kg from
   (select st.stay_id, ce.storetime, case {wcase} end v
    from first_icu st join mimiciv_icu.chartevents ce on ce.stay_id = st.stay_id
    where ce.itemid in ({wids}) and ce.valuenum is not null and ce.storetime < st.lm) x
   where x.v between 25 and 300 order by x.stay_id, x.storetime desc)
select coalesce(h.stay_id, w.stay_id) stay_id, h.height_cm, w.weight_kg
from h full join w on h.stay_id = w.stay_id;"""))


def eicu():
    cases = "\n".join(
        f"      when l.labname in ({','.join(chr(39) + n.replace(chr(39), chr(39)*2) + chr(39) for n in nm)}) "
        f"and l.labresult between {lo} and {hi} then '{name}'"
        for name, (_, nm, (lo, hi)) in LAB.items())
    allnm = sorted({n for v in LAB.values() for n in v[1]})
    nmsql = ",".join(chr(39) + n.replace(chr(39), chr(39) * 2) + chr(39) for n in allnm)
    print("  [1/3] eICU 化验（范围内过滤）…", flush=True)
    save("eicu_lab", q("eicu", f"""
with st as (select patientunitstayid sid from eicu_crd.patient where unitvisitnumber = 1)
select distinct on (x.sid, x.var) x.sid, x.var, x.labresult
from (select l.patientunitstayid sid, l.labresult, l.labresultoffset off,
             case
{cases}
             end var
      from eicu_crd.lab l join st on st.sid = l.patientunitstayid
      where l.labname in ({nmsql}) and l.labresult is not null
        and l.labresultoffset >= 0 and l.labresultoffset < 1440) x
where x.var is not null
order by x.sid, x.var, x.off desc;"""))

    parts = []
    for name, srcs in E_VITAL.items():
        lo, hi = VRANGE[name]
        for tb, col in srcs:
            parts.append(f"""select patientunitstayid sid, '{name}' var, {col} val,
       observationoffset off from eicu_crd.{tb}
where observationoffset >= 0 and observationoffset < 1440
  and {col} is not null and {col} between {lo} and {hi}""")
    print("  [2/3] eICU 体征（单次扫描 + 跨表比真实时间）…", flush=True)
    save("eicu_vital", q("eicu", f"""
with st as (select patientunitstayid sid from eicu_crd.patient where unitvisitnumber = 1),
 u as ({' union all '.join(parts)})
select distinct on (u.sid, u.var) u.sid, u.var, u.val
from u join st on st.sid = u.sid
order by u.sid, u.var, u.off desc;"""))

    print("  [3/3] eICU 体温与 GCS …", flush=True)
    save("eicu_temp_gcs", q("eicu", """
with st as (select patientunitstayid sid from eicu_crd.patient where unitvisitnumber = 1),
 t as (select distinct on (v.patientunitstayid) v.patientunitstayid sid, v.temperature tc
       from eicu_crd.vitalperiodic v join st on st.sid = v.patientunitstayid
       where v.observationoffset >= 0 and v.observationoffset < 1440
         and v.temperature between 30 and 43
       order by v.patientunitstayid, v.observationoffset desc),
 g as (select distinct on (n.patientunitstayid) n.patientunitstayid sid,
              nullif(regexp_replace(n.nursingchartvalue,'[^0-9.]','','g'),'')::numeric gcs
       from eicu_crd.nursecharting n join st on st.sid = n.patientunitstayid
       where n.nursingchartcelltypevallabel = 'Glasgow coma score'
         and n.nursingchartcelltypevalname = 'GCS Total'
         and n.nursingchartoffset >= 0 and n.nursingchartoffset < 1440
         and n.nursingchartvalue ~ '^[0-9.]+$'
       order by n.patientunitstayid, n.nursingchartoffset desc)
select coalesce(t.sid, g.sid) sid, t.tc, g.gcs
from t full join g on t.sid = g.sid;"""))


if __name__ == "__main__":
    t0 = time.time()
    print(f"=== R35 抽取 {DB} ===", flush=True)
    (mimic if DB == "mimic" else eicu)()
    print(f"\n{DB} 完成 {time.time()-t0:.0f}s", flush=True)
