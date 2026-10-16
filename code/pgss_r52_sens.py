# -*- coding: utf-8 -*-
r"""R52 分支 R52_NHANES_SCr2005_06_SENS：NHANES 2005–2006 血肌酐按 CDC/NCHS BIOPRO_D 官方公式校准后的限定重训敏感性分析。

唯一改动的预处理输入：Cycle = 2005-2006 且血肌酐非缺失者，standardized = -0.016 + 0.978 × 未校准值（mg/dL）；
随之重算 CKD-EPI 2021（与 nhanes.add_derived 同一表达式、同一运算顺序）与 eGFR降低 表型。其余列、其余周期、其余五个表型、
资格、结局（含 R46 v2 O_CVD10 可评价状态）、分组键全部不变。原冻结文件只读，不改任何冻结脚本：
本脚本在独立进程里替换 pgss_r44_nhanes_run 的模块级 DATA / RUNS（与 R46 包装同一做法），学习程序 = 冻结引擎 xgb-1.0 与适配器 r45-target-1.0 原样。

单元
  main ：R44 main 程序，12 个任务 × 3 seed × 5 外层折（含原先不可执行的 O_CVD10×低白蛋白，以核对可执行性不变）
  pop  ：R45 目标一致性 pop 臂，原 11 个纳入任务；HTFS 冻结规格读本分支同一单元的 main specs
  B    ：R45 B 目标臂，O_MORT10×低白蛋白
O_CVD10：与 R46 相同，先在原任务人群与原标签上求外层折，再删除 v2 不可评价者（保留者标签取 v2）。外层划分只依赖分组与结局，因此与原版逐人相同。
零修正模式（zero=True）：用原冻结数据，必须逐位复现现行冻结单元（R44 / R45 / R46 v2）。

用法：python pgss_r52_sens.py build | equiv | audit <jobs> | run <main|target> <jobs> | infer <new|zero> | check | freeze
"""
import sys, os, json, pathlib, time, hashlib, traceback, shutil
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pgss_threads  # noqa: F401
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd

ROOT = HERE.parent
BRANCH = "R52_NHANES_SCr2005_06_SENS"
BR = ROOT / "35_R52_NHANES_SCr2005_06_SENS"
DATA = BR / "data"
DIRS = {"main": BR / "runs44", "pop": BR / "runs_target", "B": BR / "runs_target"}
R44D = ROOT / "25_R44_NHANES跨领域复现"
R45T = ROOT / "27_R45_外部复现与机制核验" / "target"
R46 = ROOT / "28_R46_最终核验与T2"
ORIG_DATA = R44D / "data"
STATUS = R46 / "nhanes_v2" / "_local_only" / "CVD_STATUS.parquet"
VERIFY = R46 / "outcome_fix" / "R46_OUTCOME_VERIFY.json"
R51X = ROOT / "34_R51_基线只读提数" / "R51_Table1_Baseline_Source.xlsx"
NHDOC = pathlib.Path(r"D:/NAH公共数据库/NHANES/3_存档/官方codebook/2005-2006/BIOPRO_D.htm")
CYCLE = "2005-2006"
A0, A1 = -0.016, 0.978                       # BIOPRO_D：Standard creatinine = -0.016 + 0.978 × uncalibrated
CODE_FILES = ["pgss_r52_sens.py", "pgss_r44_nhanes_run.py", "pgss_r44_nhanes_cohort.py", "pgss_r45_target.py",
              "pgss_xgb_engine.py", "pgss_r36_engine.py"]
FROZEN_FILES = ["pgss_r44_nhanes_run.py", "pgss_r44_nhanes_cohort.py", "pgss_r45_target.py", "pgss_xgb_engine.py", "pgss_r36_engine.py",
                "pgss_r44_nhanes_inference.py", "pgss_r45_target_inference.py", "pgss_r42_inference.py", "pgss_r43_readonly.py"]
FROZEN_CODE_SHA = {"R44 main": "35d70452ba8b0f34", "R45 target": "26beaa0f045568d3", "R46 rerun": "c1b377bcb8c69fc9"}


def code_sha():
    h = hashlib.sha256()
    for f in CODE_FILES:
        h.update((HERE / f).read_bytes())
    return h.hexdigest()[:16]


def fsha(p, n=16):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:n]


def sha16df(df):
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[:16]


def ckd_epi_2021(scr, age, female):
    """与 nhanes.add_derived 逐字相同的表达式与运算顺序（逐位一致性由 build 核对）。"""
    k = np.where(female, 0.7, 0.9)
    a = np.where(female, -0.241, -0.302)
    ratio = scr / k
    egfr = (142 * np.minimum(ratio, 1) ** a * np.maximum(ratio, 1) ** -1.200
            * 0.9938 ** age * np.where(female, 1.012, 1.0))
    return np.where(np.isfinite(scr) & np.isfinite(age), egfr, np.nan)


def calibrate(scr, cycle):
    """只对目标周期且非缺失的值施加公式；返回新数组与改动掩码。"""
    scr = np.asarray(scr, float); m = (np.asarray(cycle) == CYCLE) & np.isfinite(scr)
    out = scr.copy(); out[m] = A0 + A1 * scr[m]
    return out, m


# ── 冻结输入 ────────────────────────────────────────────────────
def load_frozen():
    reg = json.loads((ORIG_DATA / "COHORT.json").read_text(encoding="utf-8"))["指纹"]
    F = {t: pd.read_parquet(ORIG_DATA / f"{t}.parquet") for t in ("X", "P", "OK", "meta")}
    fp = {t: sha16df(F[t]) for t in F}
    S = pd.read_parquet(STATUS); ssha = sha16df(S)
    sreg = json.loads(VERIFY.read_text(encoding="utf-8"))["NHANES"]["状态SHA"]
    return F, dict(冻结指纹=fp, 登记指纹=reg, 指纹一致=fp == reg, 状态SHA=ssha, 状态登记=sreg, 状态一致=ssha == sreg), S


def rebuild_raw():
    """按 pgss_r44_nhanes_cohort.build 同一路径内存重建（不写文件）；返回重建四表 + 原始血肌酐与重建 eGFR（按 SEQN）。"""
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
    raw = pd.DataFrame({"raw_scr": d["D_血肌酐mgdl"].to_numpy(float), "egfr_rebuilt": egfr.astype(float),
                        "LBXSCR": d["LBXSCR"].to_numpy(float), "LBDSCR": d["LBDSCR"].to_numpy(float)}, index=uid)
    return dict(X=X, P=P, OK=OK, meta=meta), raw


# ── build：分支数据 + 拟合前核验 + QC 锚点 ───────────────────────────
def build():
    BR.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)
    chk = []
    def add(name, ok, det=""):
        chk.append(dict(核验=name, 通过=bool(ok), 说明=str(det)))
    F, ver, S = load_frozen()
    add("冻结 R44 X/P/OK/meta 内容指纹 = COHORT.json 登记", ver["指纹一致"], ver["冻结指纹"])
    add("R46 v2 O_CVD10 状态内容指纹 = R46_OUTCOME_VERIFY 登记", ver["状态一致"], ver["状态SHA"])
    import pgss_r44_nhanes_run as NR, pgss_r45_target as TG, pgss_r46_rerun as RR
    cur = {"R44 main": NR.code_sha(), "R45 target": TG.code_sha(), "R46 rerun": RR.code_sha()}
    add("冻结学习程序代码指纹 = 原运行登记（R44 / R45 / R46）", cur == FROZEN_CODE_SHA, cur)
    doc = NHDOC.read_text(encoding="utf-8", errors="replace")
    import re, html
    doc = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", doc)))
    add("CDC 公式：本地存档 BIOPRO_D.htm 原文含推荐语与「-0.016 + 0.978 X (NHANES 05-06 uncalibrated serum creatinine, mg/dL)」",
        "Correction of serum creatinine values in NHANES 2005-2006 is highly recommended" in doc
        and "-0.016 + 0.978 X (NHANES 05-06 uncalibrated serum creatinine, mg/dL)" in doc)
    vars_ = sorted(set(re.findall(r"\b(LB[XD]SCR\w*)\b", doc)))
    add("BIOPRO_D 只有 LBXSCR（mg/dL）与 LBDSCRSI（µmol/L）两个血肌酐字段，没有已标准化的变量", vars_ == ["LBDSCRSI", "LBXSCR"], vars_)
    RB, raw = rebuild_raw()
    rb = {t: sha16df(RB[t]) for t in RB}
    add("内存重建四表指纹 = 登记（原始血肌酐与冻结队列逐人对齐）", rb == ver["登记指纹"], rb)
    X, P, OK, meta = (F[t] for t in ("X", "P", "OK", "meta"))
    raw = raw.loc[X.index]; cyc = meta.Cycle.to_numpy()
    s0 = X["血肌酐"].to_numpy(float); r0 = raw.raw_scr.to_numpy(float)
    m56 = cyc == CYCLE; m99 = cyc == "1999-2000"; oth = ~(m56 | m99)
    eq = lambda a, b: bool(np.array_equal(a, b, equal_nan=True))
    add("2005–2006：冻结血肌酐 = 原始 LBXSCR（未校准，mg/dL）", eq(s0[m56], raw.LBXSCR.to_numpy(float)[m56]) and eq(s0[m56], r0[m56]))
    add("1999–2000：冻结血肌酐 = 1.013 × 原始 + 0.147（Selvin，原冻结实现）", eq(s0[m99], np.where(np.isnan(r0[m99]), np.nan, 1.013 * r0[m99] + 0.147)))
    add("其余周期：冻结血肌酐 = 原始值", eq(s0[oth], r0[oth]))
    fem = meta.sex.to_numpy() == 2; age = X["年龄"].to_numpy(float)
    e0 = ckd_epi_2021(s0, age, fem)
    add("CKD-EPI 2021 复写与 nhanes.add_derived 逐位一致（冻结血肌酐上）", eq(e0, raw.egfr_rebuilt.to_numpy(float)))
    with np.errstate(invalid="ignore"):
        add("由复写 eGFR 判定的 eGFR降低 P/OK = 冻结 P/OK", eq((e0 < 60), P["eGFR降低"].to_numpy(bool)) and eq(np.isfinite(e0), OK["eGFR降低"].to_numpy(bool)))
    # —— 分支数据 ——
    s1, mchg = calibrate(s0, cyc)
    X2 = X.copy(); X2["血肌酐"] = s1
    e1 = ckd_epi_2021(s1, age, fem)
    P2 = P.copy()
    with np.errstate(invalid="ignore"):
        P2["eGFR降低"] = e1 < 60
    OK2 = OK.copy(); meta2 = meta.copy()
    add("新 eGFR 可判定 = 原可判定（不插补缺失肌酐）", eq(np.isfinite(e1), OK["eGFR降低"].to_numpy(bool)))
    for t, df in (("X", X2), ("P", P2), ("OK", OK2), ("meta", meta2)):
        df.to_parquet(DATA / f"{t}.parquet")
    new_fp = {t: sha16df(df) for t, df in (("X", X2), ("P", P2), ("OK", OK2), ("meta", meta2))}
    coh = json.loads((ORIG_DATA / "COHORT.json").read_text(encoding="utf-8"))
    coh.update(分支=BRANCH, 指纹=new_fp, 原冻结指纹=ver["登记指纹"], 改动=dict(
        说明=f"仅 {CYCLE} 且非缺失的血肌酐：{A0} + {A1} × 原值（CDC/NCHS BIOPRO_D）；随之重算 CKD-EPI 2021 与 eGFR降低", 改动人数=int(mchg.sum())),
        表型阳性人数={c: int((P2[c] & OK2[c]).sum()) for c in P2.columns}, 构建代码SHA=code_sha(), 构建时间=time.strftime("%Y-%m-%d %H:%M:%S"))
    (DATA / "COHORT.json").write_text(json.dumps(coh, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    D = diff_frames(F, dict(X=X2, P=P2, OK=OK2, meta=meta2), mchg)
    for k, v in D.items():
        add(f"改动范围：{k}", v)
    # —— QC 锚点（与 R51 T6 逐值核对）——
    t6 = pd.read_excel(R51X, sheet_name="T6_NHANES血肌酐校准影响"); t6 = dict(zip(t6.项, t6.值))
    ok56 = m56 & meta.eligible.to_numpy(bool)
    was = (e0 < 60) & np.isfinite(e0); now = (e1 < 60) & np.isfinite(e1)
    anchors = [("合格队列中 2005–2006 周期人数", int(ok56.sum())), ("其中血肌酐可得", int(mchg.sum())),
               ("2005–2006 血肌酐中位数（冻结，未校准）mg/dL", float(np.median(s0[mchg]))),
               ("2005–2006 血肌酐中位数（按 BIOPRO_D 公式校准后）mg/dL", float(np.median(s1[mchg]))),
               ("全队列 eGFR 降低（冻结）n", int(was.sum())), ("若按公式校准：eGFR 降低 n", int(now.sum())),
               ("由「降低」改判为「不降低」n", int((was & ~now).sum())), ("由「不降低」改判为「降低」n", int((~was & now).sum()))]
    A = pd.DataFrame([dict(锚点=k, R52重算=v, R51登记=t6[k], 一致=bool(np.isclose(float(v), float(t6[k]), rtol=0, atol=1e-12))) for k, v in anchors])
    add("QC 锚点与 R51 T6 逐值一致（8 项）", A.一致.all(), f"{int(A.一致.sum())}/{len(A)}")
    # —— 改动描述 ——
    d = s1[mchg] - s0[mchg]
    SC = pd.DataFrame([dict(项="改动人数（2005–2006 且血肌酐非缺失）", 值=int(mchg.sum())),
                       dict(项="2005–2006 合格者中血肌酐缺失（未插补、未改动）", 值=int((ok56 & ~np.isfinite(s0)).sum())),
                       dict(项="其他周期改动人数", 值=int((mchg & ~m56).sum())),
                       dict(项="差值（新−旧）最小 mg/dL", 值=float(d.min())), dict(项="差值 P25", 值=float(np.quantile(d, .25))),
                       dict(项="差值中位", 值=float(np.median(d))), dict(项="差值 P75", 值=float(np.quantile(d, .75))),
                       dict(项="差值最大", 值=float(d.max())), dict(项="|差值| 最大", 值=float(np.abs(d).max())),
                       dict(项="旧值最小", 值=float(s0[mchg].min())), dict(项="旧值最大", 值=float(s0[mchg].max())),
                       dict(项="新值最小", 值=float(s1[mchg].min())), dict(项="新值最大", 值=float(s1[mchg].max()))])
    okE = np.isfinite(e0)
    MIG = pd.DataFrame([dict(旧=a, 新=b, 人数=int((okE & ((e0 < 60) == (a == "A（降低）")) & ((e1 < 60) == (b == "A（降低）"))).sum()))
                        for a in ("A（降低）", "B（不降低）") for b in ("A（降低）", "B（不降低）")])
    TK = task_table(F, dict(X=X2, P=P2, OK=OK2, meta=meta2), S)
    C = pd.DataFrame(chk)
    rec = dict(全部通过=bool(C.通过.all()), 新指纹=new_fp, 改动人数=int(mchg.sum()), 代码SHA=code_sha(), 时间=time.strftime("%Y-%m-%d %H:%M:%S"))
    with pd.ExcelWriter(BR / "R52_BUILD.xlsx", engine="openpyxl") as xw:
        C.to_excel(xw, sheet_name="1_拟合前核验", index=False); A.to_excel(xw, sheet_name="2_QC锚点", index=False)
        SC.to_excel(xw, sheet_name="3_血肌酐改动", index=False); MIG.to_excel(xw, sheet_name="4_eGFR迁移矩阵", index=False)
        TK.to_excel(xw, sheet_name="5_任务旧新对照", index=False)
    (BR / "R52_BUILD.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 80)
    print(C.to_string()); print(A.to_string()); print(MIG.to_string()); print(json.dumps(rec, ensure_ascii=False))
    if not rec["全部通过"]:
        (BR / "_BUILD_STOP").write_text("拟合前核验或 QC 锚点未通过，不拟合", encoding="utf-8")
        raise SystemExit("STOP：拟合前核验未通过")
    return rec


def diff_frames(F, G, mchg):
    """新旧四表逐格比较：只允许 X.血肌酐 在 mchg 行改变、P.eGFR降低 改变；其他一律不变。"""
    out = {}
    X, X2 = F["X"], G["X"]
    same_idx = all(F[t].index.equals(G[t].index) and list(F[t].columns) == list(G[t].columns) for t in F)
    out["四表行（SEQN）与列完全相同"] = same_idx
    ch = {c: ~((X[c].to_numpy(float) == X2[c].to_numpy(float)) | (np.isnan(X[c].to_numpy(float)) & np.isnan(X2[c].to_numpy(float)))) for c in X.columns}
    out["X 只有「血肌酐」一列改变"] = all(not v.any() for c, v in ch.items() if c != "血肌酐")
    out["血肌酐改变的行 = 目标周期且非缺失（不多不少）"] = bool(np.array_equal(ch["血肌酐"], mchg))
    out["P 除 eGFR降低 外五个表型逐人相同"] = all(np.array_equal(F["P"][c].to_numpy(), G["P"][c].to_numpy()) for c in F["P"].columns if c != "eGFR降低")
    out["P eGFR降低 只在改动行变化"] = bool(not ((F["P"]["eGFR降低"].to_numpy() != G["P"]["eGFR降低"].to_numpy()) & ~mchg).any())
    out["OK 六列逐人相同"] = bool(F["OK"].equals(G["OK"]))
    out["meta（资格、分组、性别、结局、随访）逐格相同"] = bool(F["meta"].equals(G["meta"]))
    return out


def task_table(F, G, S):
    rows = []
    ev = S.O_CVD10可评价.reindex(F["meta"].index).to_numpy(bool); y2 = S.O_CVD10_v2.reindex(F["meta"].index).to_numpy(float)
    for oc in ("O_MORT10", "O_CVD10"):
        for ph in ("女性", "年龄≥65", "贫血", "eGFR降低", "低白蛋白", "肥胖"):
            r = dict(结局=oc, 表型=ph)
            for tag, H in (("旧", F), ("新", G)):
                m = H["meta"].eligible.to_numpy(bool) & H["OK"][ph].to_numpy(bool)
                y = H["meta"][oc].to_numpy(bool)
                if oc == "O_CVD10":
                    m = m & ev; y = np.where(ev, y2 == 1, False)
                z = H["P"][ph].to_numpy(bool)
                r.update({f"{tag}_n": int(m.sum()), f"{tag}_事件": int((m & y).sum()), f"{tag}_A轨n": int((m & z).sum()),
                          f"{tag}_A轨事件": int((m & z & y).sum()), f"{tag}_B轨n": int((m & ~z).sum()), f"{tag}_B轨事件": int((m & ~z & y).sum())})
            r["改变"] = any(r[f"旧_{k}"] != r[f"新_{k}"] for k in ("n", "事件", "A轨n", "A轨事件", "B轨n", "B轨事件"))
            rows.append(r)
    return pd.DataFrame(rows)


# ── 运行单元（与 pgss_r46_rerun.run_unit 逐行对应；只改数据目录、HTFS 规格来源与输出目录）────────
def use_data(zero):
    import pgss_r44_nhanes_run as NR
    NR.DATA = ORIG_DATA if zero else DATA
    return NR


def cvd_status():
    S = pd.read_parquet(STATUS)
    return S.O_CVD10可评价, S.O_CVD10_v2


def arrays(oc, ph, zero=False):
    NR = use_data(zero)
    A1, names, gs = NR.task_arrays(oc, ph)
    keep = np.ones(len(A1["uid"]), bool); y2 = A1["y"].copy()
    if oc == "O_CVD10":
        ev, yv2 = cvd_status()
        keep = ev.reindex(A1["uid"]).to_numpy(bool)
        lab = yv2.reindex(A1["uid"]).to_numpy(float)
        assert not np.isnan(lab[keep]).any(), "保留者的 v2 标签缺失"
        y2[keep] = lab[keep].astype(int)
    fp = dict(NR.data_fp(), r46_status=json.loads(VERIFY.read_text(encoding="utf-8"))["NHANES"]["状态SHA"], branch=("zero" if zero else BRANCH))
    return A1, keep, dict(A1, y=y2), names, gs, fp


def tag_of(kind, oc, ph, seed, fold):
    import pgss_r44_nhanes_run as NR, pgss_r45_target as TG
    return NR.tag_of(oc, ph, seed, fold) if kind == "main" else TG.tag_of(kind, oc, ph, seed, fold)


def frozen_dir(kind, oc, ph, seed, fold):
    """现行冻结单元：O_CVD10 中 v2 删人的任务读 R46 目录，其余读 R44 / R45 原目录。"""
    import pgss_r46_rerun as RR
    aff = set(RR.tasks("nh44"))
    t = tag_of(kind, oc, ph, seed, fold)
    if kind == "main":
        return (RR.DIRS["nh44"] if (oc, ph) in aff else R44D / "runs") / t
    return (RR.DIRS["nhpop"] if (kind == "pop" and (oc, ph) in aff) else R45T / "runs") / t


def run_unit(kind, oc, ph, seed, fold, runs=None, zero=False):
    import psutil
    import pgss_r36_engine as E, pgss_xgb_engine as EN, pgss_r44_nhanes_run as NR
    runs = DIRS[kind] if runs is None else runs
    tag = tag_of(kind, oc, ph, seed, fold)
    od = runs / tag; od.mkdir(parents=True, exist_ok=True)
    csha = code_sha()
    if (od / "_DONE").exists():
        return "SKIP_DONE"
    (od / "_FAILED").unlink(missing_ok=True)
    t0 = time.time(); proc = psutil.Process()
    A1, keep, Ac, names, gs, fp = arrays(oc, ph, zero)
    folds = E.group_split(A1["X"], A1["y"], A1["z"], gs, NR.K_OUTER, seed)
    dev0, te0 = folds[fold]
    dev, te = dev0[keep[dev0]], te0[keep[te0]]
    D, Te = EN.sub(Ac, dev), EN.sub(Ac, te)
    lay = EN.task_layout(D["X"], D["z"], names)
    meta = dict(tag=tag, kind=kind, branch=("zero" if zero else BRANCH), db="nhanes", oc=oc, ph=ph, seed=seed, fold=fold,
                engine_version=EN.ENGINE_VERSION, code_sha=csha, data_fp=fp, zero=zero, n_dev_orig=len(dev0), n_test_orig=len(te0),
                n_dev=len(dev), n_test=len(te), 删除_dev=int(len(dev0) - len(dev)), 删除_test=int(len(te0) - len(te)),
                events_dev=int(D["y"].sum()), nA_dev=int(D["z"].sum()), eventsA_dev=int(D["y"][D["z"]].sum()),
                xgb_nthread=EN.PARAM_BASE["nthread"],
                threads={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    out = pd.DataFrame({"unit_id": Te["uid"], "y": Te["y"], "z": Te["z"]})
    if kind in ("pop", "B"):
        import pgss_r45_target as TG
        src = (frozen_dir("main", oc, ph, seed, fold) if zero else DIRS["main"] / tag_of("main", oc, ph, seed, fold))
        sp44 = json.loads((src / "specs.json").read_text(encoding="utf-8"))
        h = {k: v for k, v in sp44["specs"]["HTFS"].items() if k != "cal"}
        TG.activate(kind)
        EN.set_namespace(db="nhanes", oc=oc, ph=ph, seed=seed, fold=fold, branch=f"target-{kind}", adapter=TG.ADAPTER_VERSION,
                         data_fp=fp, code=csha)
        meta.update(target=kind, adapter_version=TG.ADAPTER_VERSION, main_specs_dir=str(src.relative_to(ROOT)),
                    main_specs_sha=fsha(src / "specs.json"), htfs_frozen=h)
        try:
            r = TG.develop_pooled(D, lay, seed * 100 + fold, {"A": h["kA"], "B": h["kB"]})
            hs, hchosen = TG.htfs_switch(D, lay, r["folds"], h)
            fpred = EN.final_predict(r["specs"], D, Te, lay)
            for k, v in fpred.items():
                out[f"{k}|raw"] = v["raw"]; out[f"{k}|cal"] = v["cal"]
            meta.update(pstar=r["pstar"], chosen={k: v["chosen"] for k, v in fpred.items()}, htfs_chosen_target=hchosen, calls=dict(TG.CALLS))
        except EN.FoldFailure as ex:
            last = EN.FITLOG[-1]["role"] if EN.FITLOG else None
            (od / "_FAILED").write_text(json.dumps(dict(meta, error=str(ex), last_role=last, trace=traceback.format_exc()[-1500:]),
                                                   ensure_ascii=False, indent=1, default=str), encoding="utf-8")
            return "FAILED"
        finally:
            TG.deactivate()
        spec_doc = dict(target=kind, specs=r["specs"], inner=r["inner"], pstar=r["pstar"], inner_folds=r["inner_folds"],
                        htfs_frozen=h, htfs_inner=hs, htfs_chosen_target=hchosen)
    else:
        EN.set_namespace(db="nhanes", oc=oc, ph=ph, seed=seed, fold=fold, branch="main", data_fp=fp, code=csha)
        try:
            r = EN.develop(D, lay, seed * 100 + fold)
            fpred = EN.final_predict(r["specs"], D, Te, lay)
            for k, v in fpred.items():
                out[f"{k}|raw"] = v["raw"]; out[f"{k}|cal"] = v["cal"]
            meta.update(pstar=r["pstar"], chosen={k: v["chosen"] for k, v in fpred.items()})
        except EN.FoldFailure as ex:
            last = EN.FITLOG[-1]["role"] if EN.FITLOG else None
            (od / "_FAILED").write_text(json.dumps(dict(meta, error=str(ex), last_role=last, trace=traceback.format_exc()[-1500:]),
                                                   ensure_ascii=False, indent=1, default=str), encoding="utf-8")
            return "FAILED"
        spec_doc = dict(specs=r["specs"], inner=r["inner"], pstar=r["pstar"], inner_folds=r["inner_folds"])
    if EN.DRYRUN:
        (od / "_DONE").write_text("dryrun", encoding="utf-8")
        return "OK"
    out.to_parquet(od / "preds_test.parquet", index=False)
    (od / "specs.json").write_text(json.dumps(spec_doc, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.DataFrame(r["rows"]).to_parquet(od / "inner_scores.parquet", index=False)
    FL = pd.DataFrame(EN.FITLOG); FL.to_parquet(od / "fitlog.parquet", index=False)
    pd.DataFrame(EN.CALLOG).to_parquet(od / "callog.parquet", index=False)
    meta.update(sec_wall=round(time.time() - t0, 1), n_fits=len(FL), peak_rss_mb=round(proc.memory_info().rss / 2**20, 1),
                requests=dict(EN.REQLOG))
    meta["artifacts"] = {f.name: fsha(f) for f in sorted(od.glob("*.parquet"))}
    (od / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (od / "_DONE").write_text(str(meta["sec_wall"]), encoding="utf-8")
    return "OK"


# ── 单元清单 ────────────────────────────────────────────────────
def included_tasks():
    return [tuple(t) for t in json.loads((R45T / "INCLUDED_TASKS.json").read_text(encoding="utf-8"))["纳入任务"]]


def units(kind):
    import pgss_r44_nhanes_run as NR, pgss_r45_target as TG
    if kind == "main":
        tasks = [(oc, ph) for ph in ["女性", "肥胖", "年龄≥65", "eGFR降低", "贫血", "低白蛋白"] for oc in NR.OUTCOMES]
    elif kind == "pop":
        tasks = included_tasks()
    else:
        tasks = list(TG.B_TASKS)
    return [(kind, oc, ph, s, f) for (oc, ph) in tasks for s in NR.SEEDS for f in range(NR.K_OUTER)]


def _entry(args):
    try:
        return args, run_unit(*args)
    except Exception as e:
        od = DIRS[args[0]] / tag_of(*args); od.mkdir(parents=True, exist_ok=True)
        (od / "_FAILED").write_text(f"非预期异常 {type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}", encoding="utf-8")
        return args, "ERROR"


def run(phase, jobs):
    from concurrent.futures import ProcessPoolExecutor
    kinds = ["main"] if phase == "main" else ["pop", "B"]
    allu = [u for k in kinds for u in units(k)]
    if phase != "main":       # 目标臂需要本分支同一单元的 main specs；main 不可执行的单元不派发（原规则下同样不可执行）
        allu = [u for u in allu if (DIRS["main"] / tag_of("main", *u[1:]) / "_DONE").exists()]
    jl = [a for a in allu if not (DIRS[a[0]] / tag_of(*a) / "_DONE").exists()]
    ok = fail = 0; t0 = time.time(); st = BR / f"_STATUS_{phase}.json"
    with ProcessPoolExecutor(max_workers=jobs, max_tasks_per_child=1) as ex:
        for args, res in ex.map(_entry, jl):
            ok += res == "OK"; fail += res in ("FAILED", "ERROR")
            st.write_text(json.dumps(dict(总数=len(allu), 本次待跑=len(jl), 完成=ok, 失败或不可执行=fail,
                                          已用分钟=round((time.time() - t0) / 60, 1), 更新=time.strftime("%m-%d %H:%M:%S")),
                                     ensure_ascii=False, indent=1), encoding="utf-8")
            print(args, res, flush=True)
    return ok, fail


# ── 干跑审计（main；目标臂可执行集合与 main 相同，见 pgss_r45_target 文档）────────────
def _audit_entry(args):
    import pgss_xgb_engine as EN
    EN.DRYRUN = True
    try:
        res = run_unit(*args, runs=BR / "pathaudit_runs")
    except Exception as e:
        res = f"ERROR {type(e).__name__}: {e}"
    od = BR / "pathaudit_runs" / tag_of(*args)
    info = ""
    if (od / "_FAILED").exists():
        try:
            j = json.loads((od / "_FAILED").read_text(encoding="utf-8")); info = f"{j.get('error')} @ {j.get('last_role')}"
        except Exception:
            info = (od / "_FAILED").read_text(encoding="utf-8")[:300]
    return dict(结局=args[1], 表型=args[2], seed=args[3], 外层折=args[4], 结果=res, 信息=info)


def original_failures():
    """原 R44 main 的不可执行单元（_FAILED）。"""
    out = []
    for u in units("main"):
        d = R44D / "runs" / tag_of(*u)
        if (d / "_FAILED").exists():
            out.append(u[1:])
    return sorted(out)


def audit(jobs):
    from concurrent.futures import ProcessPoolExecutor
    pa = BR / "pathaudit_runs"
    if pa.exists():
        shutil.rmtree(pa)
    with ProcessPoolExecutor(max_workers=jobs, max_tasks_per_child=1) as ex:
        rows = list(ex.map(_audit_entry, units("main")))
    A = pd.DataFrame(rows)
    A.to_csv(BR / "R52_PATHAUDIT.csv", index=False, encoding="utf-8-sig")
    fails = sorted((r.结局, r.表型, int(r.seed), int(r.外层折)) for r in A.itertuples() if r.结果 != "OK")
    orig = original_failures()
    s = dict(可执行=int((A.结果 == "OK").sum()), 总数=len(A), 不可执行单元=[list(x) for x in fails], 原R44不可执行单元=[list(x) for x in orig],
             与原版逐单元相同=fails == orig, code_sha=code_sha())
    (BR / "R52_PATHAUDIT.json").write_text(json.dumps(s, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(s, ensure_ascii=False, indent=1, default=str))
    return s


# ── 零修正等价：用原数据逐位复现现行冻结单元 ─────────────────────────
EQUIV = [("main", "O_MORT10", "低白蛋白", 20261003, 0), ("main", "O_CVD10", "女性", 20261004, 2),
         ("main", "O_MORT10", "eGFR降低", 20261005, 1), ("pop", "O_CVD10", "女性", 20261004, 2),
         ("B", "O_MORT10", "低白蛋白", 20261003, 0)]
EQ = BR / "equiv_runs"


def compare(new_dir, old_dir):
    import pgss_r46_rerun as RR
    return RR.compare(new_dir, old_dir)


def _equiv_entry(a):
    try:
        return run_unit(*a, runs=EQ / "zero", zero=True)
    except Exception as e:
        return f"ERROR {type(e).__name__}: {e} {traceback.format_exc()[-800:]}"


def _nonzero_entry(a):
    try:
        return run_unit(*a, runs=EQ / "nonzero", zero=False)
    except Exception as e:
        return f"ERROR {type(e).__name__}: {e} {traceback.format_exc()[-800:]}"


def equiv():
    from concurrent.futures import ProcessPoolExecutor
    if EQ.exists():
        shutil.rmtree(EQ)
    nz = [EQUIV[2]]                 # 变异对照：eGFR 任务用校准后数据跑同一单元，必须与冻结单元不同
    with ProcessPoolExecutor(max_workers=len(EQUIV) + len(nz), max_tasks_per_child=1) as ex:
        fz = [ex.submit(_equiv_entry, a) for a in EQUIV]; fn = [ex.submit(_nonzero_entry, a) for a in nz]
        res = [f.result() for f in fz]; resn = [f.result() for f in fn]
    rep = []
    for a, r in zip(EQUIV, res):
        nd = EQ / "zero" / tag_of(*a); od = frozen_dir(*a)
        bad = compare(nd, od) if r == "OK" else [f"运行结果 {r}"]
        rep.append(dict(单元=f"{a[0]}:{tag_of(*a)}", 冻结来源=str(od.relative_to(ROOT)), 运行=r, 不一致项=bad, 一致=not bad))
    mut = []
    md = EQ / "_mutation"; shutil.copytree(EQ / "zero" / tag_of(*EQUIV[0]), md)
    P = pd.read_parquet(md / "preds_test.parquet"); c = [x for x in P.columns if x.startswith("HTFS|")][0]
    P.loc[0, c] = P.loc[0, c] + 1e-12; P.to_parquet(md / "preds_test.parquet", index=False)
    mut.append(dict(变异="HTFS 测试预测 +1e-12", 检出=bool(compare(md, frozen_dir(*EQUIV[0])))))
    s = json.loads((md / "specs.json").read_text(encoding="utf-8")); s["pstar"] = "S0|none" if s["pstar"] != "S0|none" else "SZ|bal"
    (md / "specs.json").write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")
    mut.append(dict(变异="P* 改名", 检出=bool(compare(md, frozen_dir(*EQUIV[0])))))
    shutil.rmtree(md)
    for a, r in zip(nz, resn):
        mut.append(dict(变异=f"校准后数据（非零修正）运行同一单元 {tag_of(*a)}",
                        检出=bool(r == "OK" and compare(EQ / "nonzero" / tag_of(*a), frozen_dir(*a)))))
    summ = dict(全部一致=all(r["一致"] for r in rep), 单元=rep, 变异对照=mut, 变异全检出=all(m["检出"] for m in mut),
                code_sha=code_sha(), 时间=time.strftime("%Y-%m-%d %H:%M:%S"))
    (BR / "R52_EQUIV.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1, default=str))
    return summ


# ── 正式运行后的单元核对：测试集合、标签、分轨、数据指纹、患者不泄漏 ─────────
def foldcheck():
    newfp = json.loads((DATA / "COHORT.json").read_text(encoding="utf-8"))["指纹"]
    mig = set(pd.read_parquet(DATA / "P.parquet").index[pd.read_parquet(DATA / "P.parquet")["eGFR降低"].to_numpy()
                                                         != pd.read_parquet(ORIG_DATA / "P.parquet")["eGFR降低"].to_numpy()])
    rows, per_seed = [], {}
    for kind in ("main", "pop", "B"):
        for u in units(kind):
            nd = DIRS[kind] / tag_of(*u); od = frozen_dir(*u)
            r = dict(类=kind, 单元=tag_of(*u), 新完成=(nd / "_DONE").exists(), 原完成=(od / "_DONE").exists())
            if r["新完成"] and r["原完成"]:
                Pn = pd.read_parquet(nd / "preds_test.parquet", columns=["unit_id", "y", "z"]).sort_values("unit_id")
                Po = pd.read_parquet(od / "preds_test.parquet", columns=["unit_id", "y", "z"]).sort_values("unit_id")
                same = np.array_equal(Pn.unit_id.to_numpy(), Po.unit_id.to_numpy())
                zd = set(Pn.unit_id[Pn.z.to_numpy() != Po.z.to_numpy()]) if same else {"?"}
                m = json.loads((nd / "meta.json").read_text(encoding="utf-8"))
                r.update(测试集合相同=bool(same), 标签相同=bool(same and np.array_equal(Pn.y.to_numpy(), Po.y.to_numpy())),
                         分轨改变人数=len(zd), 分轨改变只在迁移者=bool(zd <= mig and (u[2] == "eGFR降低" or not zd)),
                         数据指纹为分支=({k: m["data_fp"].get(k) for k in newfp} == newfp and m["data_fp"].get("branch") == BRANCH))
                # ↑ 冻结后修订（R52_AMENDMENT.json）：run_unit 把 NR.data_fp() 的四个指纹平铺存进 data_fp，原写法读不存在的「指纹」键、恒为 False
                if kind == "main":
                    per_seed.setdefault((u[1], u[2], u[3]), []).append(Pn.unit_id.to_numpy())
            rows.append(r)
    F = pd.DataFrame(rows)
    leak = []
    for (oc, ph, s), parts in per_seed.items():
        allu = np.concatenate(parts)
        leak.append(dict(结局=oc, 表型=ph, seed=s, 折数=len(parts), 测试人次=len(allu), 唯一=len(np.unique(allu)), 无重复=len(allu) == len(np.unique(allu))))
    Lk = pd.DataFrame(leak)
    both = F.新完成 & F.原完成
    ok = bool((F.新完成 == F.原完成).all() and F.loc[both, ["测试集合相同", "标签相同", "分轨改变只在迁移者", "数据指纹为分支"]].all().all()
              and Lk.无重复.all())
    F.to_csv(BR / "R52_FOLDCHECK.csv", index=False, encoding="utf-8-sig"); Lk.to_csv(BR / "R52_LEAKCHECK.csv", index=False, encoding="utf-8-sig")
    rec = dict(全部通过=ok, 单元=len(F), 新完成=int(F.新完成.sum()), 原完成=int(F.原完成.sum()), 可执行性一致=bool((F.新完成 == F.原完成).all()),
               分轨改变人数合计_main=int(F.loc[both & (F.类 == "main"), "分轨改变人数"].sum()), 迁移者=len(mig), 时间=time.strftime("%Y-%m-%d %H:%M:%S"))
    (BR / "R52_FOLDCHECK.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    return ok


# ── 冻结登记（正式运行前；driver 开跑前核对）───────────────────────────
def freeze():
    files = {f: fsha(HERE / f) for f in CODE_FILES + [x for x in FROZEN_FILES if x not in CODE_FILES] + ["pgss_r52_infer.py"]}
    data = {f"data/{t}.parquet": fsha(DATA / f"{t}.parquet") for t in ("X", "P", "OK", "meta")}
    eq = json.loads((BR / "R52_EQUIV.json").read_text(encoding="utf-8")); pa = json.loads((BR / "R52_PATHAUDIT.json").read_text(encoding="utf-8"))
    zc = json.loads((BR / "R52_INFER_ZERO_CHECK.json").read_text(encoding="utf-8"))
    rec = dict(分支=BRANCH, 代码SHA=code_sha(), 文件SHA=files, 数据文件SHA=data, 冻结代码指纹=FROZEN_CODE_SHA,
               EQUIV=eq["全部一致"] and eq["变异全检出"], EQUIV代码SHA为当前=eq["code_sha"] == code_sha(),
               PATHAUDIT=pa["与原版逐单元相同"], PATHAUDIT代码SHA为当前=pa["code_sha"] == code_sha(),
               零修正推断与R46v2逐格相同=zc["全部通过"], 时间=time.strftime("%Y-%m-%d %H:%M:%S"))
    if not all(rec[k] for k in ("EQUIV", "EQUIV代码SHA为当前", "PATHAUDIT", "PATHAUDIT代码SHA为当前", "零修正推断与R46v2逐格相同")):
        raise SystemExit(f"冻结前置条件不满足：{rec}")
    (BR / "R52_FREEZE.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rec, ensure_ascii=False, indent=1))


def verify_freeze():
    fz = json.loads((BR / "R52_FREEZE.json").read_text(encoding="utf-8"))
    bad = [f for f, v in fz["文件SHA"].items() if fsha(HERE / f) != v]
    bad += [f for f, v in fz["数据文件SHA"].items() if fsha(BR / f) != v]
    return bad


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build()
    elif cmd == "equiv":
        sys.exit(0 if equiv()["全部一致"] else 1)
    elif cmd == "audit":
        audit(int(sys.argv[2]))
    elif cmd == "run":
        print(run(sys.argv[2], int(sys.argv[3])))
    elif cmd == "freeze":
        freeze()
    elif cmd == "foldcheck":
        sys.exit(0 if foldcheck() else 1)
    elif cmd == "verify_freeze":
        b = verify_freeze(); print(b); sys.exit(1 if b else 0)
    elif cmd == "one":
        print(run_unit(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6])))
