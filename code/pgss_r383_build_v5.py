# -*- coding: utf-8 -*-
r"""R383：数据 v5 —— 把**全部正式时间变化变量**统一到同一个 24h 信息窗口。

GPT 第 42 轮冻结的窗口（MIMIC）
    charttime >= ICU intime        实际测量发生在入科后
    charttime <  landmark          测量发生在预测时点之前
    storetime <  landmark          结果在预测时点前已进入系统

适用：六项常规体征（v4 已完成，本轮原样沿用同一规则）、**体温**、**全部正式化验**。
取值程序（三步，与 v4 同）
    1. 取最大 charttime
    2. 同 charttime 同 itemid 多版本 → 取 landmark 前 storetime 最晚者
    3. 同 charttime 跨多个已批准 itemid → 单位统一 + 范围 QC 后取算术平均
    **不得跨不同 charttime 平均。**

身高/体重/BMI 按「基线体型」单独处理（GPT §4）
    eICU 侧已核实为 `patient.admissionheight` / `admissionweight`，即 admission-level，
    故 MIMIC 冻结为：landmark 前已可用（storetime < lm）的合法记录中，
    取 **charttime 距 ICU intime 最近**者；并列则依次按 storetime 最早、itemid 最小打破。
    BMI 由冻结后的身高体重统一重算。

纪律
    · 不新增/删除 itemid 映射，不改任何单位换算与临床范围 QC（原样沿用 v3 的常量）
    · 不根据模型结果调整规则
    · 删除计数一律**算出来**，不许写字面 0（R382 在此处写了硬编码 0，GPT §5 已指出）

用法
    $env:PGPASSWORD="..."; python pgss_r383_build_v5.py            # 构建
    $env:PGPASSWORD="..."; python pgss_r383_build_v5.py --verify   # 清暂存表重扫并逐内容比对
"""
import hashlib, json, os, pathlib, shutil, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import pgss_threads  # noqa: F401
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, psycopg2

HERE = pathlib.Path(__file__).parent; ROOT = HERE.parent
O = ROOT / "19_R38"; V4 = O / "data_v4"; V5 = O / "data_v5"
CONN = dict(host=os.environ.get("PGHOST", "127.0.0.1"), port=int(os.environ.get("PGPORT", "5431")),
            user=os.environ.get("PGUSER", "postgres"), password=os.environ.get("PGPASSWORD"))
if not CONN["password"]: raise SystemExit("未设置 PGPASSWORD")
from pgss_r35_extract import LAB, CHART, TEMP, BODY, ST
import pgss_r35_qc as QC35
REUSE_STAGING = os.environ.get("REUSE_STAGING", "1") == "1"
STAT = {}
TEMP_RANGE = (30.0, 43.0)          # v3 既定，不改
H_RANGE = (100.0, 230.0)           # v3 既定
W_RANGE = (25.0, 300.0)            # v3 既定


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]


def conn(db):
    c = psycopg2.connect(dbname=db, **CONN); c.autocommit = True
    return c


def q(sql, db="mimic", label=""):
    t = time.time()
    c = conn(db)
    try:
        d = pd.read_sql(sql, c)
    finally:
        c.close()
    print(f"    {label}{len(d):,} 行  {time.time()-t:.0f}s", flush=True)
    return d


# ───────────────────── 暂存：体征 + 体温 ─────────────────────
def stage_chart():
    """一遍扫 chartevents：六项体征 + 体温（体温先 F→C 再过范围）"""
    cids = sorted({i for v in CHART.values() for i in v[0]})
    case = "\n".join(
        f"      when ce.itemid in ({','.join(map(str, c[0]))}) "
        f"and ce.valuenum between {c[1][0]} and {c[1][1]} then '{n}'"
        for n, c in CHART.items())
    lo, hi = TEMP_RANGE
    allids = sorted(set(cids) | {TEMP["C"], TEMP["F"]})
    c = conn("mimic")
    try:
        with c.cursor() as cur:
            cur.execute("SET work_mem = '512MB'")

            def cnt(t):
                cur.execute("select to_regclass(%s) is not null", (t,))
                if not cur.fetchone()[0]: return 0
                cur.execute(f"select count(*) from {t}")
                return cur.fetchone()[0]

            n = cnt("v5_chart") if REUSE_STAGING else 0
            if n == 0:
                t = time.time()
                print("    扫 chartevents 落暂存表（统一窗口，含体温）…", flush=True)
                cur.execute("drop table if exists v5_chart;")
                cur.execute(f"""
create unlogged table v5_chart as
{ST}
select stay_id, itemid, v, charttime, stt, var from (
  select st.stay_id, ce.itemid, ce.charttime, ce.storetime stt,
         case when ce.itemid = {TEMP['F']} then (ce.valuenum - 32.0) * 5.0 / 9.0
              else ce.valuenum end v,
         case
{case}
              when ce.itemid in ({TEMP['C']},{TEMP['F']}) then '体温'
         end var
  from first_icu st join mimiciv_icu.chartevents ce on ce.stay_id = st.stay_id
  where ce.itemid in ({','.join(map(str, allids))}) and ce.valuenum is not null
    and ce.charttime is not null and ce.storetime is not null
    and ce.charttime >= st.intime
    and ce.charttime <  st.lm
    and ce.storetime <  st.lm
) z
where z.var is not null
  and (z.var <> '体温' or z.v between {lo} and {hi});
""")
                cur.execute("create index on v5_chart (var, stay_id, charttime desc, stt desc);")
                cur.execute("analyze v5_chart;")
                n = cnt("v5_chart")
                print(f"      v5_chart {n:,} 行 {time.time()-t:.0f}s", flush=True)
            else:
                print(f"    复用 v5_chart（{n:,} 行）", flush=True)
    finally:
        c.close()
    return n


# ───────────────────── 暂存：化验 ─────────────────────
def stage_lab():
    ids = sorted({i for v in LAB.values() for i in v[0]})
    case = "\n".join(
        f"      when le.itemid in ({','.join(map(str, v[0]))}) "
        f"and le.valuenum between {v[2][0]} and {v[2][1]} then '{n}'"
        for n, v in LAB.items())
    c = conn("mimic")
    try:
        with c.cursor() as cur:
            cur.execute("SET work_mem = '512MB'")

            def cnt(t):
                cur.execute("select to_regclass(%s) is not null", (t,))
                if not cur.fetchone()[0]: return 0
                cur.execute(f"select count(*) from {t}")
                return cur.fetchone()[0]

            n = cnt("v5_lab") if REUSE_STAGING else 0
            if n == 0:
                t = time.time()
                print("    扫 labevents 落暂存表（统一窗口）…", flush=True)
                cur.execute("drop table if exists v5_lab;")
                cur.execute(f"""
create unlogged table v5_lab as
{ST}
select st.stay_id, le.itemid, le.valuenum v, le.charttime, le.storetime stt,
       case
{case}
       end var
from first_icu st join mimiciv_hosp.labevents le on le.hadm_id = st.hadm_id
where le.itemid in ({','.join(map(str, ids))}) and le.valuenum is not null
  and le.charttime is not null and le.storetime is not null
  and le.charttime >= st.intime
  and le.charttime <  st.lm
  and le.storetime <  st.lm;
""")
                cur.execute("delete from v5_lab where var is null;")
                cur.execute("create index on v5_lab (var, stay_id, charttime desc, stt desc);")
                cur.execute("analyze v5_lab;")
                n = cnt("v5_lab")
                print(f"      v5_lab {n:,} 行 {time.time()-t:.0f}s", flush=True)
            else:
                print(f"    复用 v5_lab（{n:,} 行）", flush=True)
    finally:
        c.close()
    return n


# ───────────────────── 三步取值（体征/体温/化验共用） ─────────────────────
def pick(table, names, tag):
    out, rows = [], []
    for name in names:
        print(f"  {name} …", flush=True)
        # 三步取值的**无连接**实现：窗口函数定位最大 charttime、再定位该 itemid
        # 在该时点的最晚 storetime，然后两级 group by 取平均。
        # 旧写法用两次自连接，执行计划一变就退化（体温实测 3s → 1741s）。
        # 等价性已在真实数据上逐值验证（收缩压/心率/白蛋白/PT，最大差 0.0）。
        d = q(f"""
with c as (select stay_id, itemid, v, stt, charttime,
                  max(charttime) over (partition by stay_id) mct
           from {table} where var='{name}'),
     a as (select stay_id, itemid, v, stt,
                  max(stt) over (partition by stay_id, itemid) mstt
           from c where charttime = mct),
     l as (select stay_id, itemid, avg(v) v from a where stt = mstt
           group by stay_id, itemid)
select stay_id, avg(v) v from l group by stay_id;""", "mimic", f"{name} ")
        out.append(d.assign(var=name)[["stay_id", "var", "v"]])
        # 纯 rollup 链：g3(stay,itemid,stt) → g2(stay,itemid) → g1(stay)。
        # 旧写法用三个 atm 自连接，对 30 万行以上的变量退化到 ~20 分钟/个
        # （血红蛋白实测 1238s，取值查询仅 2s）。等价性已在真实数据上验证。
        s = q(f"""
with cand as (select stay_id, itemid, v, charttime, stt from {table} where var='{name}'),
     mx   as (select stay_id, max(charttime) mct from cand group by stay_id),
     atm  as (select c.* from cand c join mx on mx.stay_id=c.stay_id and c.charttime=mx.mct),
     g3   as (select stay_id, itemid, stt, count(distinct v) nv
              from atm group by stay_id, itemid, stt),
     g2   as (select stay_id, itemid, count(*) nstt, max(nv) mxnv
              from g3 group by stay_id, itemid),
     g1   as (select stay_id, count(*) nid, max(nstt) mxstt, max(mxnv) mxnv
              from g2 group by stay_id)
select count(*) n_stay,
       count(case when nid   > 1 then 1 end) n_multisource,
       count(case when mxstt > 1 then 1 end) n_revision,
       count(case when mxnv  > 1 then 1 end) n_exact_dup_diff_value
from g1;""", "mimic", f"{name} 统计 ")
        r = s.iloc[0]
        rows.append(dict(变量=name, 有值就诊=int(r.n_stay or 0),
                         同charttime多来源聚合=int(r.n_multisource or 0),
                         同charttime同itemid多版本=int(r.n_revision or 0),
                         键相同但取值不同=int(r.n_exact_dup_diff_value or 0)))
    STAT[tag] = rows
    W = pd.concat(out, ignore_index=True)
    return W.pivot_table(index="stay_id", columns="var", values="v", aggfunc="first")


# ───────────────────── 基线体型 ─────────────────────
def body_baseline():
    """landmark 前已可用的合法记录中，取 charttime 距 intime 最近者。

    并列打破：先 storetime 最早，再 itemid 最小 —— 完全确定，不依赖行序。
    """
    res = {}
    for col, srcs, (lo, hi) in (("身高cm", BODY["height_cm"], H_RANGE),
                                ("体重kg", BODY["weight_kg"], W_RANGE)):
        case = " ".join(f"when ce.itemid = {i} then ce.valuenum * {f}" for i, f in srcs)
        ids = ",".join(str(i) for i, _ in srcs)
        print(f"  {col} …", flush=True)
        d = q(f"""
{ST}
select distinct on (x.stay_id) x.stay_id, x.v
from (select st.stay_id, ce.itemid, ce.charttime, ce.storetime stt,
             case {case} end v,
             abs(extract(epoch from (ce.charttime - st.intime))) dist
      from first_icu st join mimiciv_icu.chartevents ce on ce.stay_id = st.stay_id
      where ce.itemid in ({ids}) and ce.valuenum is not null
        and ce.charttime is not null and ce.storetime is not null
        and ce.storetime < st.lm) x
where x.v between {lo} and {hi}
order by x.stay_id, x.dist asc, x.stt asc, x.itemid asc;""", "mimic", f"{col} ")
        res[col] = d.set_index("stay_id")["v"].rename(col)
    B = pd.concat(res.values(), axis=1)
    STAT["MIMIC基线体型"] = [dict(变量=c, 有值就诊=int(B[c].notna().sum())) for c in B.columns]
    return B


# ───────────────────── 真正算出来的删除计数 ─────────────────────
def removal_counts():
    """**算出来**每个变量因新窗口被排除的记录数（不许写字面 0）"""
    cids = sorted({i for v in CHART.values() for i in v[0]})
    ccase = "\n".join(
        f"  when ce.itemid in ({','.join(map(str, c[0]))}) then '{n}'" for n, c in CHART.items())
    lids = sorted({i for v in LAB.values() for i in v[0]})
    lcase = "\n".join(
        f"  when le.itemid in ({','.join(map(str, v[0]))}) then '{n}'" for n, v in LAB.items())
    print("  体征/体温 被排除记录…", flush=True)
    a = q(f"""
{ST}
select var,
  sum(case when charttime <  intime then 1 else 0 end) n_pre_icu_removed,
  sum(case when charttime >= lm     then 1 else 0 end) n_post_lm_removed,
  count(*) n_cand_storetime_window
from (select st.intime, st.lm, ce.charttime,
             case
{ccase}
                  when ce.itemid in ({TEMP['C']},{TEMP['F']}) then '体温' end var
      from first_icu st join mimiciv_icu.chartevents ce on ce.stay_id = st.stay_id
      where ce.itemid in ({','.join(map(str, sorted(set(cids)|{TEMP['C'],TEMP['F']})))})
        and ce.valuenum is not null and ce.charttime is not null and ce.storetime is not null
        and ce.storetime < st.lm) z
where var is not null group by var;""", "mimic", "体征/体温 ")
    print("  化验 被排除记录…", flush=True)
    b = q(f"""
{ST}
select var,
  sum(case when charttime <  intime then 1 else 0 end) n_pre_icu_removed,
  sum(case when charttime >= lm     then 1 else 0 end) n_post_lm_removed,
  count(*) n_cand_storetime_window
from (select st.intime, st.lm, le.charttime,
             case
{lcase}
             end var
      from first_icu st join mimiciv_hosp.labevents le on le.hadm_id = st.hadm_id
      where le.itemid in ({','.join(map(str, lids))})
        and le.valuenum is not null and le.charttime is not null and le.storetime is not null
        and le.storetime < st.lm) z
where var is not null group by var;""", "mimic", "化验 ")
    R = pd.concat([a, b], ignore_index=True)
    STAT["被排除记录_算出来的"] = R.to_dict("records")
    return R


# ───────────────────── 组装 ─────────────────────
def build():
    t0 = time.time()
    print("=== MIMIC 体征 + 体温（统一窗口）===", flush=True)
    stage_chart()
    CT = pick("v5_chart", list(CHART.keys()) + ["体温"], "MIMIC体征体温聚合")
    print("\n=== MIMIC 化验（统一窗口）===", flush=True)
    stage_lab()
    LB = pick("v5_lab", list(LAB.keys()), "MIMIC化验聚合")
    print("\n=== MIMIC 基线体型 ===", flush=True)
    BD = body_baseline()
    print("\n=== 被排除记录（算出来的）===", flush=True)
    RM = removal_counts()

    print("\n=== 装进 v4 的骨架（只换 MIMIC 的 X 列）===", flush=True)
    V5.mkdir(parents=True, exist_ok=True)
    X4 = pd.read_parquet(V4 / "R38_mimic_X.parquet")
    meta = pd.read_parquet(V4 / "R38_mimic_META.parquet")
    # stay_id 是**索引**不是列：一律按 X 的索引对齐（v4 同此做法）
    X5 = X4.copy()
    nrep, skipped = 0, []
    for src in (CT, LB, BD):
        for c in src.columns:
            if c not in X5.columns:
                skipped.append(c); continue
            X5[c] = src[c].reindex(X5.index)
            nrep += 1
    print(f"    替换了 {nrep} 列"
          + (f"；不在 X 中故跳过（不新增列）：{skipped}" if skipped else ""))

    # BMI 与表型由 phenotypes() 统一重算（公式/切点沿用 v3，不改）
    X5b, P5, OK5 = QC35.phenotypes(X5, meta)
    for nm, obj in (("R38_mimic_X", X5b), ("R38_mimic_PHENO", P5),
                    ("R38_mimic_PHENOOK", OK5), ("R38_mimic_META", meta)):
        obj.to_parquet(V5 / f"{nm}.parquet")          # 保留 stay_id 索引
    for n in ("eicu_X", "eicu_PHENO", "eicu_PHENOOK", "eicu_META"):
        (V5 / f"R38_{n}.parquet").write_bytes((V4 / f"R38_{n}.parquet").read_bytes())
    print("    eICU 四个文件原样复制（本轮未改 eICU 规则）", flush=True)

    man = dict(
        数据版本="v5", 时间=time.strftime("%Y-%m-%d %H:%M:%S"),
        改动原因="GPT 第 42 轮：把全部正式时间变化变量统一到同一 24h 信息窗口",
        v5窗口规则=dict(
            体征体温化验="charttime >= intime AND charttime < landmark AND storetime < landmark",
            取值="最大charttime → 同itemid取最晚storetime → 同charttime跨itemid取算术平均",
            不得="跨不同 charttime 平均",
            基线体型="storetime < landmark 的合法记录中取 charttime 距 intime 最近；并列按 storetime 最早、itemid 最小",
            BMI="由冻结后的身高体重重算（公式与范围沿用 v3）"),
        未改动项=["itemid 映射", "单位换算系数", "临床范围 QC", "eICU 全部输入", "landmark 定义"],
        已知的跨库不对称=("eICU 的 lab 表只有 labresultoffset（结果时间），无采血时间字段 ⇒ "
                   "eICU 化验无法按 charttime 实现同一窗口，仍为结果时间口径。"
                   "本轮按指令对 MIMIC 施加采血时间窗口，该不对称如实记录，供裁决。"),
        聚合统计=STAT,
        v4_SHA={f"{d}_{k}": sha(V4 / f"R38_{d}_{k}.parquet")
                for d in ("mimic", "eicu") for k in ("X", "PHENO", "PHENOOK", "META")},
        v5_SHA={f"{d}_{k}": sha(V5 / f"R38_{d}_{k}.parquet")
                for d in ("mimic", "eicu") for k in ("X", "PHENO", "PHENOOK", "META")},
        构建脚本SHA=sha(pathlib.Path(__file__)))
    (V5 / "DATA_VERSION.json").write_text(
        json.dumps(man, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    RM.to_parquet(O / "R383_REMOVALS.parquet", index=False)
    print(f"\nv5 已写入 {V5}   {time.time()-t0:.0f}s")
    return True


def verify():
    """清暂存表重扫 → 与已存 v5 逐内容比对"""
    global REUSE_STAGING
    c = conn("mimic")
    try:
        with c.cursor() as cur:
            cur.execute("drop table if exists v5_chart;")
            cur.execute("drop table if exists v5_lab;")
    finally:
        c.close()
    REUSE_STAGING = False
    print("已清暂存表且关闭复用 ⇒ 第二遍重新扫原表（真正独立）", flush=True)
    keep = {}
    for f in sorted(V5.glob("R38_mimic_*.parquet")):
        keep[f.name] = pd.read_parquet(f)
    build()
    det, ok = [], True
    for n, a in keep.items():
        b = pd.read_parquet(V5 / n)
        same_shape = a.shape == b.shape
        num = [x for x in a.columns if pd.api.types.is_numeric_dtype(a[x])]
        worst = 0.0
        if same_shape:
            for x in num:
                u, v = a[x].to_numpy(float), b[x].to_numpy(float)
                m = np.isfinite(u) & np.isfinite(v)
                if (np.isfinite(u) != np.isfinite(v)).any(): worst = float("inf"); break
                if m.any(): worst = max(worst, float(np.abs(u[m] - v[m]).max()))
        nonnum = [x for x in a.columns if x not in num]
        same_non = same_shape and all(a[x].equals(b[x]) for x in nonnum)
        good = same_shape and same_non and worst == 0.0
        ok &= good
        det.append(dict(文件=n, 形状一致=same_shape, 非数值列一致=bool(same_non),
                        数值最大差=worst, 判定="一致" if good else "不一致"))
    out = dict(时间=time.strftime("%Y-%m-%d %H:%M:%S"), 数据版本="v5", 可复现=bool(ok),
               文件数=len(det), 明细=det, 构建脚本SHA=sha(pathlib.Path(__file__)),
               说明="独立重扫 chartevents + labevents 并重算，与已存 v5 逐内容比对")
    (O / "R383_V5_REPRODUCIBILITY.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(pd.DataFrame(det).to_string(index=False))
    print(f"\n可复现 = {ok}")
    return ok


if __name__ == "__main__":
    if "--verify" in sys.argv:
        sys.exit(0 if verify() else 2)
    sys.exit(0 if build() else 2)
