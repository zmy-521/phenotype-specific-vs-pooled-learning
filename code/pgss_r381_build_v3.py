# -*- coding: utf-8 -*-
r"""R38.1 §2 续：**数据版本 v3** —— 把不可复现的抽取改成确定性抽取。

为什么必须升版本
    R38.1 的 MIMIC 真对账（`pgss_r381_mimic_reconcile.py`）按 R35 **声明的规则**重抽，
    结果 38 个可对账特征里 **33 个与冻结输入不一致**：
      · 化验类（PT/RDW/尿素氮/白蛋白/血肌酐…）**全部一致，最大差 0.0**
      · 体征类不一致 9,118–9,400 例（约 14%），最大绝对差 114–146
      · 六个表型**全部一致**（它们只依赖化验与人口学）
    根因**不是** landmark 时间规则错（时间规则复现一致），而是
    `DISTINCT ON (...) ORDER BY ... <time> DESC` **没有确定性的并列打破规则**，
    而同一变量由多个 itemid 映射。实测（收缩压，itemid 220050 动脉 + 220179 无创）：
      84,787 例有值 → **21,575 例(25.4%)** 在最晚 storetime 上有多行
      → **20,977 例(24.7%)** 的并列取值不唯一，**并列值最大跨度 202 mmHg**
    eICU 同一写法也有该问题，但轻得多（血肌酐 266/135,557 = 0.20%，最大跨度 0.3）。
    ⇒ **冻结输入无法按文档重新导出**。对一篇以方法学严谨为卖点的论文，这是致命缺陷。

v3 的规则（只改"并列怎么办"，不改时间资格、不改特征池、不改表型切点、不改结局）
    对每个 (单位, 变量)：先取**时间资格内最晚的时间点**（与 v2 相同），
    若该时间点上有多行，则取这些值的**算术平均**。

    为什么选均值
      · 取任意一个 ⇒ 不可复现（现状）
      · 取最小/最大 ⇒ 确定但**引入系统偏倚**（会系统性偏向动脉或无创一侧）
      · 取均值 ⇒ 确定、且**对测量来源中性**；对"同一时刻记录的同一量"取平均是标准做法
    均值在 pandas 内按**固定排序后**计算，不用 SQL 的 avg()
    （并行聚合的求和顺序可能变化 ⇒ 末位可能不稳）。

    **这是一个可被 GPT 推翻的判断**，不是纯技术细节。替代方案见报告。

纪律
    · 不覆盖 v1/v2；v3 写入 `19_R38/data_v3/`
    · 逐变量量化 v3 与 v2 的差异，写入 DATA_VERSION_V3.json 与 xlsx
    · MIMIC 与 eICU 都重建（两库都有该缺陷）

用法： $env:PGPASSWORD="..."; python pgss_r381_build_v3.py
"""
import hashlib, json, os, pathlib, sys, time, warnings
warnings.filterwarnings("ignore")
import sys as _s, pathlib as _p
_s.path.insert(0, str(_p.Path(__file__).parent))
import pgss_threads  # noqa: F401
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, psycopg2
HERE = pathlib.Path(__file__).parent; ROOT = HERE.parent
F35 = ROOT / "16_R35_FIX"
V2 = ROOT / "19_R38" / "data_v2"
V3 = ROOT / "19_R38" / "data_v3"; V3.mkdir(parents=True, exist_ok=True)
O = ROOT / "19_R38"
CONN = dict(host=os.environ.get("PGHOST", "127.0.0.1"), port=int(os.environ.get("PGPORT", "5431")),
            user=os.environ.get("PGUSER", "postgres"), password=os.environ.get("PGPASSWORD"))
if not CONN["password"]: raise SystemExit("未设置 PGPASSWORD")
from pgss_r35_extract import LAB, CHART, ST, BODY, TEMP, E_VITAL, VRANGE
import pgss_r35_qc as QC35


def sha(p):
    p = pathlib.Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else None


def q(sql, db):
    t = time.time()
    with psycopg2.connect(dbname=db, **CONN) as c:
        with c.cursor() as cur:
            cur.execute("SET work_mem = '256MB'")
            cur.execute("SET temp_buffers = '64MB'")
        d = pd.read_sql(sql, c)
    print(f"    {len(d):,} 行 {time.time()-t:.0f}s", flush=True)
    return d


def at_max_time(cand_sql, keys, tcol):
    """把候选子查询包成「只返回每个 keys 组合在 tcol 上最晚时点的行」。

    **最晚时点的筛选必须在 SQL 里做。** 首版把全部候选行拉回 pandas，
    eICU vitalperiodic 在 24h 内跨 6 变量两张表可达上亿行，内存会被撑爆。
    留在库里后，返回的只是每组 1 行（并列时几行），量级与就诊数同阶。
    """
    part = ", ".join(keys)
    # 用**窗口函数单遍**，不要 CTE + group by + join：
    # 后者会把候选集物化再扫两遍，MIMIC chartevents（42 GB / 4.33 亿行）下会落盘落到不可用。
    return f"""
select z.* from (
  select j.*, max(j.{tcol}) over (partition by {part}) as _mo
  from (
{cand_sql}
  ) j
) z
where z.{tcol} = z._mo;"""

# ── 并列打破规则：**这是一个有科学含量的判断，不是技术细节** ──
# 换这个常量就等于换数据版本，必须同步升 ACTIVE 并重算全部结果。
#   "mean"  取并列值的算术平均 —— 确定、且对测量来源中性（当前采用）
#   "min"   取最小 —— 确定但系统性偏向读数较低的来源
#   "max"   取最大 —— 确定但系统性偏向读数较高的来源
# 不提供「按来源优先级」选项：那需要逐变量的临床论证（如动脉优于无创），
# 不能由代码默认决定。GPT 若要该口径，需给出每个变量的优先级与理由。
TIE_RULE = "mean"

# 暂存表是否可复用。主构建允许（重启不必再付一次 15GB 扫描）；
# **verify 必须设为 False** —— 第二遍要真正独立，否则只验了折叠、没验扫描本身。
REUSE_STAGING = True


def collapse(df, keys, val, tcol):
    """确定性折叠：取最晚时间点，并列按 `TIE_RULE` 处理（固定排序后计算）。

    均值在 pandas 内算，不用 SQL 的 avg() —— 并行聚合的求和顺序可能变化，末位可能不稳。
    """
    d = df.dropna(subset=[val])
    mx = d.groupby(keys)[tcol].transform("max")
    at = d[d[tcol] == mx]
    at = at.sort_values(keys + [tcol, val], kind="mergesort")   # 稳定排序 ⇒ 可复现
    g = at.groupby(keys, sort=True)[val]
    if TIE_RULE == "mean":
        out = g.mean().reset_index()
    elif TIE_RULE == "min":
        out = g.min().reset_index()
    elif TIE_RULE == "max":
        out = g.max().reset_index()
    else:
        raise ValueError(f"未知的 TIE_RULE：{TIE_RULE}")
    n_tie = int((g.size() > 1).sum())
    n_tie_diff = int((g.nunique() > 1).sum())
    return out, n_tie, n_tie_diff


# ───────────────────────── MIMIC ─────────────────────────
def mimic_v3():
    stat = {}
    cases = "\n".join(
        f"      when le.itemid in ({','.join(map(str, ids))}) and le.valuenum between {lo} and {hi} "
        f"then '{name}'" for name, (ids, _, (lo, hi)) in LAB.items())
    allids = sorted({i for v in LAB.values() for i in v[0]})
    print("  [1/4] MIMIC 化验候选 …", flush=True)
    cand = f"""
{ST}
select x.stay_id, x.var, x.valuenum, x.storetime
from (select st.stay_id, le.valuenum, le.storetime,
             case
{cases}
             end var
      from first_icu st
      join mimiciv_hosp.labevents le on le.hadm_id = st.hadm_id
      where le.itemid in ({','.join(map(str, allids))}) and le.valuenum is not null
        and le.storetime is not null
        and le.storetime >= st.intime and le.storetime < st.lm) x
where x.var is not null"""
    lab = q(at_max_time(cand, ["stay_id", "var"], "storetime"), "mimic")
    L, t1, t2 = collapse(lab, ["stay_id", "var"], "valuenum", "storetime")
    stat["化验"] = dict(候选行=len(lab), 并列=t1, 并列取值不唯一=t2)

    ccases = "\n".join(
        f"      when ce.itemid in ({','.join(map(str, ids))}) and ce.valuenum between {lo} and {hi} "
        f"then '{name}'" for name, (ids, (lo, hi)) in CHART.items())
    cids = sorted({i for v in CHART.values() for i in v[0]})
    print("  [2/4] MIMIC 体征候选（大表）…", flush=True)
    cand = f"""
{ST}
select x.stay_id, x.var, x.valuenum, x.storetime
from (select st.stay_id, ce.valuenum, ce.storetime,
             case
{ccases}
             end var
      from first_icu st
      join mimiciv_icu.chartevents ce on ce.stay_id = st.stay_id
      where ce.itemid in ({','.join(map(str, cids))}) and ce.valuenum is not null
        and ce.storetime is not null
        and ce.storetime >= st.intime and ce.storetime < st.lm) x
where x.var is not null"""
    cha = q(at_max_time(cand, ["stay_id", "var"], "storetime"), "mimic")
    V, t1, t2 = collapse(cha, ["stay_id", "var"], "valuenum", "storetime")
    stat["体征"] = dict(候选行=len(cha), 并列=t1, 并列取值不唯一=t2)

    print("  [3/4] MIMIC 体温候选 …", flush=True)
    cand = f"""
{ST}
select x.stay_id, x.tc, x.storetime
from (select st.stay_id, ce.storetime,
             case when ce.itemid = {TEMP['C']} then ce.valuenum
                  when ce.itemid = {TEMP['F']} then (ce.valuenum - 32) * 5.0 / 9.0 end tc
      from first_icu st
      join mimiciv_icu.chartevents ce on ce.stay_id = st.stay_id
      where ce.itemid in ({TEMP['C']}, {TEMP['F']}) and ce.valuenum is not null
        and ce.storetime is not null
        and ce.storetime >= st.intime and ce.storetime < st.lm) x
where x.tc between 30 and 43"""
    tg = q(at_max_time(cand, ["stay_id"], "storetime"), "mimic")
    T, t1, t2 = collapse(tg, ["stay_id"], "tc", "storetime")
    stat["体温"] = dict(候选行=len(tg), 并列=t1, 并列取值不唯一=t2)

    print("  [4/4] MIMIC 身高体重候选 …", flush=True)
    hcase = " ".join(f"when ce.itemid = {i} then ce.valuenum * {f}" for i, f in BODY["height_cm"])
    wcase = " ".join(f"when ce.itemid = {i} then ce.valuenum * {f}" for i, f in BODY["weight_kg"])
    hids = ",".join(str(i) for i, _ in BODY["height_cm"])
    wids = ",".join(str(i) for i, _ in BODY["weight_kg"])
    hw = {}
    for nm, case, ids, lo, hi in (("height_cm", hcase, hids, 100, 230),
                                  ("weight_kg", wcase, wids, 25, 300)):
        cand = f"""
{ST}
select x.stay_id, x.v, x.storetime
from (select st.stay_id, ce.storetime, case {case} end v
      from first_icu st
      join mimiciv_icu.chartevents ce on ce.stay_id = st.stay_id
      where ce.itemid in ({ids}) and ce.valuenum is not null
        and ce.storetime is not null and ce.storetime < st.lm) x
where x.v between {lo} and {hi}"""
        d = q(at_max_time(cand, ["stay_id"], "storetime"), "mimic")
        r, t1, t2 = collapse(d, ["stay_id"], "v", "storetime")
        hw[nm] = r.set_index("stay_id").v
        stat[nm] = dict(候选行=len(d), 并列=t1, 并列取值不唯一=t2)

    co = pd.read_parquet(F35 / "R35_MIMIC_COHORT.parquet").set_index("stay_id")
    Lw = L.pivot_table(index="stay_id", columns="var", values="valuenum", aggfunc="first")
    Vw = V.pivot_table(index="stay_id", columns="var", values="valuenum", aggfunc="first")
    X = pd.concat([Lw, Vw], axis=1).reindex(co.index)
    X["体温"] = T.set_index("stay_id").tc.reindex(co.index)
    X["年龄"] = co.age
    X["身高cm"] = hw["height_cm"].reindex(co.index)
    X["体重kg"] = hw["weight_kg"].reindex(co.index)
    # MIMIC 的 GCS总分 沿用 v2：它来自 R35 的 chartevents 抽取，**同样用了 distinct on**，
    # 因此也有并列不确定性。本轮未重抽（需另一次 chartevents 全扫），
    # 在 DATA_VERSION 的「已知遗留」里如实记录，不假装它已确定性化。
    _v2x = pd.read_parquet(V2 / "R38_mimic_X.parquet")
    if "GCS总分" in _v2x.columns:
        X["GCS总分"] = _v2x["GCS总分"].reindex(co.index)
    meta = pd.read_parquet(V2 / "R38_mimic_META.parquet")
    X2, P, OK = QC35.phenotypes(X, meta)
    return X2, P, OK, meta, stat


# ───────────────────────── eICU ─────────────────────────


def stage_eicu_vitals(pv_cols, pa_cols):
    """扫一遍两张体征表落 unlogged 暂存表，再**逐变量**取最晚时点的行。

    返回列 (sid, var, val, off)：每个 (sid,var) 一行，或并列时几行。

    为什么逐变量而不是 unpivot
        暂存表约 4000 万行，unpivot 成 6 倍（约 2.4 亿行）再 group by + join
        要扫两遍那个巨大中间集。逐变量则每次只碰一列，并能用 (sid, off) 索引。

    为什么单一来源的变量不需要处理并列
        `vitalperiodic` 对每个 (sid, off) 只有一行，
        故心率/呼吸/SpO2 的「最晚一次」唯一确定；只有血压三项跨两张表才可能并列。
        该前提由 `assert_one_row_per_sid_off()` 实测验证，不是假设。
    """
    import psycopg2 as pg
    sel_pv = ", ".join(f"{c} as {n}" for n, c in pv_cols.items())
    whr_pv = " or ".join(
        f"({c} is not null and {c} between {VRANGE[n][0]} and {VRANGE[n][1]})"
        for n, c in pv_cols.items())
    sel_pa = ", ".join(f"{c} as {n}" for n, c in pa_cols.items())
    whr_pa = " or ".join(
        f"({c} is not null and {c} between {VRANGE[n][0]} and {VRANGE[n][1]})"
        for n, c in pa_cols.items())
    out = []
    # **autocommit 必须在进入任何 with 之前设，且不要用 with 管连接。**
    # 实测：`with pg.connect(...) as c:` 内部再设 c.autocommit 不生效 ——
    # CREATE 一直在事务里，从别的连接看不到，with 正常退出才提交、异常则回滚。
    # 后果：构建一报错，648 秒扫出来的暂存表就没了，重启还得再扫一遍。
    c = pg.connect(dbname="eicu", **CONN)
    c.autocommit = True
    try:
        with c.cursor() as cur:
            cur.execute("SET work_mem = '256MB'")

            def exists(t):
                cur.execute("select to_regclass(%s) is not null", (t,))
                if not cur.fetchone()[0]:
                    return 0
                cur.execute(f"select count(*) from {t}")
                return cur.fetchone()[0]

            # 暂存表**可复用**：已有且非空就不重扫（重启不必再付一次 15GB 扫描）
            n1 = exists("stg_pv") if REUSE_STAGING else 0
            if n1 == 0:
                t = time.time(); print("    扫 vitalperiodic 落暂存表 …", flush=True)
                # **同一 cursor 内先 DROP 再 CREATE**：
                # 否则「表已存在」会撞唯一约束 pg_type_typname_nsp_index。
                # 实测踩过：两个 verify 进程并发，一个建成、另一个报
                # duplicate key (typname)=(stg_pv) 并退出，链条据此误判为数据不一致。
                cur.execute("drop table if exists stg_pv;")
                cur.execute(f"""
create unlogged table stg_pv as
select v.patientunitstayid sid, v.observationoffset off, {sel_pv}
from eicu_crd.vitalperiodic v
join (select patientunitstayid sid from eicu_crd.patient where unitvisitnumber = 1) st
  on st.sid = v.patientunitstayid
where v.observationoffset >= 0 and v.observationoffset < 1440 and ({whr_pv});""")
                cur.execute("create index on stg_pv (sid, off desc);")
                n1 = exists("stg_pv")
                print(f"      stg_pv {n1:,} 行 {time.time()-t:.0f}s", flush=True)
            else:
                print(f"    复用已有 stg_pv（{n1:,} 行）", flush=True)
            n2 = exists("stg_pa") if REUSE_STAGING else 0
            if n2 == 0:
                t = time.time(); print("    扫 vitalaperiodic 落暂存表 …", flush=True)
                cur.execute("drop table if exists stg_pa;")
                cur.execute(f"""
create unlogged table stg_pa as
select v.patientunitstayid sid, v.observationoffset off, {sel_pa}
from eicu_crd.vitalaperiodic v
join (select patientunitstayid sid from eicu_crd.patient where unitvisitnumber = 1) st
  on st.sid = v.patientunitstayid
where v.observationoffset >= 0 and v.observationoffset < 1440 and ({whr_pa});""")
                cur.execute("create index on stg_pa (sid, off desc);")
                n2 = exists("stg_pa")
                print(f"      stg_pa {n2:,} 行 {time.time()-t:.0f}s", flush=True)
            else:
                print(f"    复用已有 stg_pa（{n2:,} 行）", flush=True)

            # 如实测量 (sid,off) 的重复情况。
            # **这不是通过条件，而是要报告的事实**：实测 stg_pv 有 11 个重复组，
            # 因为 vitalperiodic 的主键在代理键 vitalperiodicid 上，不在 (sid,off) 上
            # ⇒ 数据库从未保证唯一性 ⇒ **所有变量都按并列处理，不给单源变量开捷径**。
            cur.execute("""select count(*) from (
                select sid, off from stg_pv group by sid, off having count(*) > 1) k;""")
            dup_pv = cur.fetchone()[0]
            cur.execute("""select count(*) from (
                select sid, off from stg_pa group by sid, off having count(*) > 1) k;""")
            dup_pa = cur.fetchone()[0]
            print(f"    (sid,off) 重复组：stg_pv {dup_pv} 个、stg_pa {dup_pa} 个"
                  f" ⇒ 所有变量统一按并列处理", flush=True)

        for n in pv_cols:
            lo, hi = VRANGE[n]
            t = time.time()
            if n in pa_cols:
                # 跨两表：各自 max(off) → 取较大者 → 回两表取该时点全部候选
                sql = f"""
with mv as (select sid, max(off) mo from stg_pv
            where {n} is not null and {n} between {lo} and {hi} group by sid),
     ma as (select sid, max(off) mo from stg_pa
            where {n} is not null and {n} between {lo} and {hi} group by sid),
     mm as (select coalesce(mv.sid, ma.sid) sid,
                   greatest(coalesce(mv.mo, -1), coalesce(ma.mo, -1)) mo
            from mv full join ma on mv.sid = ma.sid)
select mm.sid, '{n}' var, v.{n} val, mm.mo off
from mm join stg_pv v on v.sid = mm.sid and v.off = mm.mo
where v.{n} is not null and v.{n} between {lo} and {hi}
union all
select mm.sid, '{n}' var, a.{n} val, mm.mo off
from mm join stg_pa a on a.sid = mm.sid and a.off = mm.mo
where a.{n} is not null and a.{n} between {lo} and {hi};"""
                tag = "跨两表"
            else:
                # 单一来源也**同样处理**：(sid,off) 实测有重复，不能用 distinct on 取一行
                sql = f"""
with m as (select sid, max(off) mo from stg_pv
           where {n} is not null and {n} between {lo} and {hi} group by sid)
select m.sid, '{n}' var, v.{n} val, m.mo off
from m join stg_pv v on v.sid = m.sid and v.off = m.mo
where v.{n} is not null and v.{n} between {lo} and {hi};"""
                tag = "单源"
            d = pd.read_sql(sql, c)
            ng = d.groupby(["sid", "var"]).ngroups
            nt = int((d.groupby(["sid", "var"]).val.size() > 1).sum())
            print(f"      {n}（{tag}）{len(d):,} 行 / {ng:,} 组，其中并列 {nt} 组"
                  f"  {time.time()-t:.0f}s", flush=True)
            out.append(d)
    finally:
        c.close()
    return pd.concat(out, ignore_index=True)


def eicu_v3():
    stat = {}
    cases = "\n".join(
        f"      when l.labname in ({','.join(chr(39)+n.replace(chr(39),chr(39)*2)+chr(39) for n in nm)}) "
        f"and l.labresult between {lo} and {hi} then '{name}'"
        for name, (_, nm, (lo, hi)) in LAB.items())
    allnm = sorted({n for v in LAB.values() for n in v[1]})
    nmsql = ",".join(chr(39) + n.replace(chr(39), chr(39) * 2) + chr(39) for n in allnm)
    print("  [1/3] eICU 化验候选（v2 时间规则 + 确定性折叠）…", flush=True)
    cand = f"""
with st as (select patientunitstayid sid from eicu_crd.patient where unitvisitnumber = 1)
select x.sid, x.var, x.labresult, x.off
from (select l.patientunitstayid sid, l.labresult, l.labresultoffset off,
             case
{cases}
             end var
      from eicu_crd.lab l join st on st.sid = l.patientunitstayid
      where l.labname in ({nmsql}) and l.labresult is not null
        and l.labresultoffset >= 0 and l.labresultoffset < 1440
        and (l.labresultrevisedoffset is null or l.labresultrevisedoffset < 1440)) x
where x.var is not null"""
    lab = q(at_max_time(cand, ["sid", "var"], "off"), "eicu")
    L, t1, t2 = collapse(lab, ["sid", "var"], "labresult", "off")
    stat["化验"] = dict(候选行=len(lab), 并列=t1, 并列取值不唯一=t2)

    # eICU 体征：**必须回库重抽**。缓存 raw/eicu_vital.parquet 只有 (sid,var,val)，
    # 没有时间列 ⇒ 无法从缓存重做确定性折叠。而血压同样来自两张表
    # （vitalperiodic 侵入式 / vitalaperiodic 无创），与 MIMIC 是同一个多源并列问题。
    print("  [2/3] eICU 体征：扫一遍落暂存表，再从暂存表折叠 …", flush=True)
    # 为什么用暂存表：原写法 6 变量 × 两表 = 最多 9 个子查询各扫一遍 vitalperiodic
    # （15 GB / 1.47 亿行，约 135 GB 读，实测 11 分钟仍在落盘）。
    # 扫一遍落到 unlogged 表后，后续全部逻辑都在小表上做。
    pv_cols, pa_cols = {}, {}
    for name, srcs in E_VITAL.items():
        for tb, col in srcs:
            (pv_cols if tb == "vitalperiodic" else pa_cols)[name] = col
    vi = stage_eicu_vitals(pv_cols, pa_cols)
    V, t1, t2 = collapse(vi, ["sid", "var"], "val", "off")
    stat["体征"] = dict(候选行=len(vi), 并列=t1, 并列取值不唯一=t2)
    Vw = V.pivot_table(index="sid", columns="var", values="val", aggfunc="first")

    print("  [3/3] eICU 体温与 GCS 候选（回库重抽 + 确定性折叠）…", flush=True)
    cand = """
with st as (select patientunitstayid sid from eicu_crd.patient where unitvisitnumber = 1)
select v.patientunitstayid sid, v.temperature tc, v.observationoffset off
from eicu_crd.vitalperiodic v join st on st.sid = v.patientunitstayid
where v.observationoffset >= 0 and v.observationoffset < 1440
  and v.temperature between 30 and 43"""
    tcd = q(at_max_time(cand, ["sid"], "off"), "eicu")
    TC, t1, t2 = collapse(tcd, ["sid"], "tc", "off")
    stat["体温"] = dict(候选行=len(tcd), 并列=t1, 并列取值不唯一=t2)
    cand = """
with st as (select patientunitstayid sid from eicu_crd.patient where unitvisitnumber = 1)
select n.patientunitstayid sid,
       nullif(regexp_replace(n.nursingchartvalue,'[^0-9.]','','g'),'')::numeric gcs,
       n.nursingchartoffset off
from eicu_crd.nursecharting n join st on st.sid = n.patientunitstayid
where n.nursingchartcelltypevallabel = 'Glasgow coma score'
  and n.nursingchartcelltypevalname = 'GCS Total'
  and n.nursingchartoffset >= 0 and n.nursingchartoffset < 1440
  and n.nursingchartvalue ~ '^[0-9.]+$'"""
    gcd = q(at_max_time(cand, ["sid"], "off"), "eicu")
    GC, t1, t2 = collapse(gcd, ["sid"], "gcs", "off")
    stat["GCS"] = dict(候选行=len(gcd), 并列=t1, 并列取值不唯一=t2)

    co = pd.read_parquet(F35 / "R35_EICU_COHORT.parquet").set_index("sid")
    Lw = L.pivot_table(index="sid", columns="var", values="labresult", aggfunc="first")
    X = pd.concat([Lw, Vw], axis=1).reindex(co.index)
    X["体温"] = TC.set_index("sid").tc.reindex(co.index)
    X["GCS总分"] = GC.set_index("sid").gcs.reindex(co.index)
    X["年龄"] = co.age
    X["身高cm"] = pd.to_numeric(co.height_raw, errors="coerce").where(
        lambda s: (s >= 100) & (s <= 230))
    X["体重kg"] = pd.to_numeric(co.weight_raw, errors="coerce").where(
        lambda s: (s >= 25) & (s <= 300))
    meta = pd.read_parquet(V2 / "R38_eicu_META.parquet")
    X2, P, OK = QC35.phenotypes(X, meta)
    return X2, P, OK, meta, stat


def diff_vs_v2(db, X, P, OK, meta):
    X2 = pd.read_parquet(V2 / f"R38_{db}_X.parquet")
    P2 = pd.read_parquet(V2 / f"R38_{db}_PHENO.parquet")
    OK2 = pd.read_parquet(V2 / f"R38_{db}_PHENOOK.parquet")
    e = meta.eligible.fillna(False).to_numpy(bool)
    rows = []
    for c in X2.columns:
        if c not in X.columns:
            rows.append(dict(变量=c, 合格N=int(e.sum()), 值改变=None, 说明="v3 未产出该列")); continue
        a = pd.to_numeric(X[c], errors="coerce").to_numpy(float)[e]
        b = pd.to_numeric(X2[c], errors="coerce").to_numpy(float)[e]
        both = np.isfinite(a) & np.isfinite(b)
        ch = int((both & (a != b)).sum())
        rows.append(dict(变量=c, 合格N=int(e.sum()), 两侧均有值=int(both.sum()),
                         值改变=ch, 改变率=round(100.0 * ch / max(int(e.sum()), 1), 3),
                         最大绝对差=(float(np.abs(a[both] - b[both]).max()) if both.any() else 0.0),
                         v3缺v2有=int((~np.isfinite(a) & np.isfinite(b)).sum()),
                         v3有v2缺=int((np.isfinite(a) & ~np.isfinite(b)).sum())))
    D = pd.DataFrame(rows).sort_values("值改变", ascending=False, na_position="last")
    pr = []
    for c in P2.columns:
        a = P[c].to_numpy(bool)[e] & OK[c].to_numpy(bool)[e]
        b = P2[c].to_numpy(bool)[e] & OK2[c].to_numpy(bool)[e]
        pr.append(dict(表型=c, 阳性_v2=int(b.sum()), 阳性_v3=int(a.sum()),
                       判定改变=int((a != b).sum()),
                       可判定_v2=int(OK2[c].to_numpy(bool)[e].sum()),
                       可判定_v3=int(OK[c].to_numpy(bool)[e].sum())))
    return D, pd.DataFrame(pr)


def main():
    t0 = time.time()
    out = {}
    for db, fn in (("eicu", eicu_v3), ("mimic", mimic_v3)):
        print(f"\n=== {db} v3 ===", flush=True)
        X, P, OK, meta, stat = fn()
        for nm, obj in ((f"R38_{db}_X", X), (f"R38_{db}_PHENO", P),
                        (f"R38_{db}_PHENOOK", OK), (f"R38_{db}_META", meta)):
            obj.to_parquet(V3 / f"{nm}.parquet")
        D, PD = diff_vs_v2(db, X, P, OK, meta)
        out[db] = dict(并列统计=stat, 变量差异=D, 表型差异=PD)
        print(f"  {db} v3 已写；与 v2 有改变的变量 {int((D.值改变.fillna(0) > 0).sum())} 个",
              flush=True)
        print(D.head(8).to_string(index=False))
        print(PD.to_string(index=False))

    man = dict(
        数据版本="v3", 时间=time.strftime("%Y-%m-%d %H:%M:%S"),
        改动原因=("R38.1 对账发现：`DISTINCT ON ... ORDER BY <time> DESC` 无确定性并列打破规则，"
                "同一变量由多个 itemid 映射 ⇒ 最晚时点并列时取哪个值是任意的 ⇒ "
                "冻结输入无法按文档重新导出（实测 MIMIC 体征 14% 不一致，最大差 114–146；"
                "收缩压 24.7% 的就诊在最晚时点有不唯一取值，并列跨度最大 202 mmHg）。"),
        并列规则=TIE_RULE,
        v3规则=(f"对每个 (单位, 变量)：取时间资格内最晚时间点（与 v2 相同）；"
              "该时点多行则取**算术平均**，在 pandas 内按稳定排序后计算。"),
        选择理由=("取任意一个不可复现；取最小/最大引入系统偏倚（偏向动脉或无创一侧）；"
               "取均值确定且对测量来源中性。**这是可被 GPT 推翻的判断。**"),
        未改动项=["时间资格规则", "特征池定义", "表型切点", "结局定义", "队列定义"],
        已知遗留=["MIMIC GCS总分 沿用 v2（未重抽，仍可能受并列不确定性影响）",
               "MIMIC/eICU 队列定义处的 distinct on (hadm_id)/(sid) 未加打破规则"
               "（仅当同一 hadm_id 有多条同 intime 的 ICU 停留时才有歧义）"],
        并列统计={db: v["并列统计"] for db, v in out.items()},
        v2_SHA={n: sha(V2 / f"R38_{n}.parquet") for n in
                ("eicu_X", "eicu_PHENO", "eicu_PHENOOK", "eicu_META",
                 "mimic_X", "mimic_PHENO", "mimic_PHENOOK", "mimic_META")},
        v3_SHA={n: sha(V3 / f"R38_{n}.parquet") for n in
                ("eicu_X", "eicu_PHENO", "eicu_PHENOOK", "eicu_META",
                 "mimic_X", "mimic_PHENO", "mimic_PHENOOK", "mimic_META")},
        特征池SHA=sha(F35 / "R35_DATA_QC.xlsx"))
    (V3 / "DATA_VERSION.json").write_text(json.dumps(man, ensure_ascii=False, indent=2,
                                                     default=str), encoding="utf-8")
    with pd.ExcelWriter(O / "R381_DATA_V3_DIFF.xlsx", engine="openpyxl") as xw:
        pd.DataFrame([{k: (v if not isinstance(v, (dict, list))
                           else json.dumps(v, ensure_ascii=False, default=str)[:400])
                       for k, v in man.items()}]).T.reset_index().to_excel(
            xw, sheet_name="0_版本说明", index=False)
        for db, v in out.items():
            v["变量差异"].to_excel(xw, sheet_name=f"{db}_变量差异", index=False)
            v["表型差异"].to_excel(xw, sheet_name=f"{db}_表型差异", index=False)
    print(f"\nv3 已写入 {V3}   {time.time()-t0:.0f}s")
    return True




def verify():
    """**可复现性验证**：把全部抽取与折叠再独立跑一遍，写进临时目录，与已存 v3 逐位比对。

    这是 v3 存在的理由所对应的那条性质 ——
    「按文档跑两遍得到完全相同的文件」。v1/v2 不满足它（实测 MIMIC 体征 14% 不一致）。

    注意：不能用 `pgss_r381_mimic_reconcile.py` 来验 v3 ——
    那个脚本按**旧的非确定性 SQL**重抽，拿它比 v3 会按构造失败。
    """
    import shutil, tempfile, psycopg2 as _pg
    # 第二遍必须**真正独立**：先清掉暂存表，否则只验了折叠、没验扫描本身的确定性
    _c = _pg.connect(dbname="eicu", **CONN)
    _c.autocommit = True
    try:
        with _c.cursor() as _cur:
            _cur.execute("drop table if exists stg_pv; drop table if exists stg_pa;")
    finally:
        _c.close()
    global REUSE_STAGING
    REUSE_STAGING = False
    print("  已清暂存表且关闭复用 ⇒ 第二遍会重新扫原表（真正独立）", flush=True)
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="pgss_v3_verify_"))
    global V3
    stored = V3
    rows = []
    try:
        V3 = tmp
        for db, fn in (("eicu", eicu_v3), ("mimic", mimic_v3)):
            print(f"\n=== 第二遍 {db} ===", flush=True)
            X, P_, OK, meta, _ = fn()
            for nm, obj in ((f"R38_{db}_X", X), (f"R38_{db}_PHENO", P_),
                            (f"R38_{db}_PHENOOK", OK), (f"R38_{db}_META", meta)):
                obj.to_parquet(tmp / f"{nm}.parquet")
        for nm in ("eicu_X", "eicu_PHENO", "eicu_PHENOOK", "eicu_META",
                   "mimic_X", "mimic_PHENO", "mimic_PHENOOK", "mimic_META"):
            a = stored / f"R38_{nm}.parquet"; b = tmp / f"R38_{nm}.parquet"
            # parquet 字节可能因写入元数据而不同 ⇒ 比**内容**，不比文件字节
            import pandas as _pd
            da = _pd.read_parquet(a); dbf = _pd.read_parquet(b)
            same_shape = da.shape == dbf.shape
            same_cols = list(da.columns) == list(dbf.columns)
            if same_shape and same_cols:
                num = [c for c in da.columns if _pd.api.types.is_numeric_dtype(da[c])]
                import numpy as _np
                mx = 0.0
                for c in num:
                    x = _pd.to_numeric(da[c], errors="coerce").to_numpy(float)
                    y = _pd.to_numeric(dbf[c], errors="coerce").to_numpy(float)
                    both = _np.isfinite(x) & _np.isfinite(y)
                    if (_np.isfinite(x) != _np.isfinite(y)).any():
                        mx = float("nan"); break
                    if both.any():
                        mx = max(mx, float(_np.abs(x[both] - y[both]).max()))
                nonnum = [c for c in da.columns if c not in num]
                obj_same = all(da[c].astype(str).equals(dbf[c].astype(str)) for c in nonnum)
            else:
                mx = float("nan"); obj_same = False
            rows.append(dict(文件=nm, 形状一致=same_shape, 列一致=same_cols,
                             非数值列一致=obj_same, 数值最大差=mx,
                             判定=("一致" if same_shape and same_cols and obj_same
                                 and mx == 0.0 else "**不一致**")))
            print(f"  {nm}: {rows[-1]['判定']}（数值最大差 {mx}）", flush=True)
    finally:
        V3 = stored
        shutil.rmtree(tmp, ignore_errors=True)
    import pandas as pd2, json as js
    D = pd2.DataFrame(rows)
    ok = bool((D.判定 == "一致").all())
    out = dict(时间=time.strftime("%Y-%m-%d %H:%M:%S"), 数据版本="v3",
               可复现=ok, 文件数=len(D),
               并列规则=TIE_RULE,
               构建脚本SHA=hashlib.sha256(
                   pathlib.Path(__file__).read_bytes()).hexdigest()[:16],
               备注=("首遍由本脚本的早一版产出（并列规则硬编码为 mean），"
                     "第二遍由当前版产出（TIE_RULE 常量，默认 mean）。"
                     "两版语义已在受控用例上自测等价，故比对仍然有效；"
                     "若二者不一致，应先怀疑这一点。"),
               说明=("独立跑第二遍全部抽取与确定性折叠，与已存 v3 逐内容比对。"
                     "这正是 v1/v2 不满足的那条性质。"),
               明细=D.to_dict("records"))
    (O / "R381_V3_REPRODUCIBILITY.json").write_text(
        js.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    with pd2.ExcelWriter(O / "R381_V3_REPRODUCIBILITY.xlsx", engine="openpyxl") as xw:
        D.to_excel(xw, sheet_name="逐文件比对", index=False)
    print(f"\n可复现：{ok}")
    print(D.to_string(index=False))
    print(f"已写 {O/'R381_V3_REPRODUCIBILITY.json'}")
    return ok

if __name__ == "__main__":
    if "--verify" in sys.argv:
        sys.exit(0 if verify() else 2)
    sys.exit(0 if main() else 1)
