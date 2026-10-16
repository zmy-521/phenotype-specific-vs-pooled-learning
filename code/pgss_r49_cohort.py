# -*- coding: utf-8 -*-
r"""R49 任务一（GPT 第 53 轮）：MIMIC-IV / eICU-CRD 终版纳排流程。只读，不修改 data v5，不重建研究定义，不连接数据库。

来源链（逐条核实，不假设）：
  · 队列与结局的唯一定义 = pgss_r35_cohort.py（R35 交付代码包内同名文件，逐字节核对）；
  · data v5 的 META（eligible、OUT_HOSP_MORT、OUT_LOS72、sex）与 R35 队列 parquet 逐条比较必须完全相同——相同则 R35 的流程即 data v5 的流程；
  · 逐步人数取 R35 运行记录 R35_COHORT_OUTCOMES.xlsx（与 R35 交付的 R35_CORRECTED_BENCHMARK.xlsx 表 10、11 逐值核对）；
    凡能从冻结数据重算的一步都重算并比较；不能重算的标「仅运行记录」；条件分解恢复不了的写「不可恢复」，不推算；
  · 表型任务：data v5 的 META × PHENO × PHENOOK 计数，与 R41 A1、R42 任务完整度、R47 B1 逐项对账；
  · 表型定义直接取 pgss_r35_qc.phenotypes 源码（v5 构建脚本调用的同一函数）。
不使用 R34 的任何人数（R34 的结局口径是 ICU 内死亡、ICU 住院 >7 天，与终版不同）。
"""
import sys, json, pathlib, hashlib, zipfile, inspect, io
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd

ROOT = HERE.parent
HAND = pathlib.Path(r"D:/Research/3_对话与交接/GPT交接_PGSS")          # 只用于与交付副本逐字节/逐值比对
V5 = ROOT / "19_R38" / "data_v5"
R35D = ROOT / "16_R35_FIX"
DBS = {"mimic": dict(name="MIMIC-IV", key="stay_id", r35="R35_MIMIC_COHORT", pid="subject_id", hosp_stay="hadm_id"),
       "eicu": dict(name="eICU-CRD", key="sid", r35="R35_EICU_COHORT", pid="uniquepid", hosp_stay="phs")}
PH = ["女性", "年龄≥65", "贫血", "eGFR降低", "低白蛋白", "肥胖"]
PH_EN = {"女性": "Female", "年龄≥65": "Age ≥65", "贫血": "Anemia", "eGFR降低": "Reduced eGFR", "低白蛋白": "Low albumin", "肥胖": "Obesity"}
OC = {"OUT_HOSP_MORT": "In-hospital death after the landmark", "OUT_LOS72": "ICU stay > 72 h"}
_C = {}


def sha16(b):
    return hashlib.sha256(b).hexdigest()[:16]


def v5(db, t):
    k = ("v5", db, t)
    if k not in _C:
        _C[k] = pd.read_parquet(V5 / f"R38_{db}_{t}.parquet")
    return _C[k]


def r35(db):
    k = ("r35", db)
    if k not in _C:
        _C[k] = pd.read_parquet(R35D / f"{DBS[db]['r35']}.parquet").set_index(DBS[db]["key"])
    return _C[k]


# ── A：来源与版本核对 ────────────────────────────────────────────
def a_sources():
    rows = []
    man = json.loads((V5 / "DATA_VERSION.json").read_text(encoding="utf-8"))["v5_SHA"]
    for db in DBS:
        for t in ("META", "PHENO", "PHENOOK"):
            b = (V5 / f"R38_{db}_{t}.parquet").read_bytes()
            rows.append(dict(对象=f"data v5 {db}_{t}", 本地SHA16=sha16(b), 对照="DATA_VERSION.json 登记的 v5_SHA", 对照SHA16=man[f"{db}_{t}"],
                             一致=sha16(b) == man[f"{db}_{t}"]))
    z = zipfile.ZipFile(HAND / "35" / "R35_CODE_核心.zip")
    a, b = (HERE / "pgss_r35_cohort.py").read_bytes(), z.read("pgss_r35_cohort.py")
    rows.append(dict(对象="pgss_r35_cohort.py（队列与结局定义）", 本地SHA16=sha16(a), 对照="R35 交付 35/R35_CODE_核心.zip 内同名文件", 对照SHA16=sha16(b), 一致=a == b))
    for f in ("pgss_r35_qc.py", "pgss_r383_build_v5.py"):
        rows.append(dict(对象=f"{f}（{'表型定义' if 'qc' in f else 'data v5 构建'}）", 本地SHA16=sha16((HERE / f).read_bytes()),
                         对照="R47_ENV.json 登记的代码 SHA", 对照SHA16=json.loads((ROOT / "29_R47_写作交接" / "R47_ENV.json").read_text(encoding="utf-8"))["code_sha"].get(f, ""),
                         一致=sha16((HERE / f).read_bytes()) == json.loads((ROOT / "29_R47_写作交接" / "R47_ENV.json").read_text(encoding="utf-8"))["code_sha"].get(f, "")))
    return pd.DataFrame(rows)


def a_v5_equals_r35():
    """data v5 META 与 R35 队列 parquet 逐条比较（eligible 对应 R35 的 eligible & mort_source_ok，与 pgss_r35_qc 组装 META 的写法一致）。"""
    rows = []
    for db, d in DBS.items():
        M, C = v5(db, "META"), r35(db)
        same_idx = set(M.index) == set(C.index) and M.index.is_unique and C.index.is_unique
        C = C.loc[M.index]
        for col, a, b in (("eligible", M.eligible, C.eligible & C.mort_source_ok), ("OUT_HOSP_MORT", M.OUT_HOSP_MORT, C.OUT_HOSP_MORT),
                          ("OUT_LOS72", M.OUT_LOS72, C.OUT_LOS72), ("sex", M.sex.fillna("缺失"), C.sex.fillna("缺失"))):
            rows.append(dict(数据库=d["name"], 字段=col, 分析单位=len(M), 单位集合相同=bool(same_idx), 不同的单位=int((a.to_numpy() != b.to_numpy()).sum())))
    out = pd.DataFrame(rows); out["一致"] = out.单位集合相同 & (out.不同的单位 == 0)
    return out


def r35_record():
    """R35 运行记录（本地）及其与 R35 交付工作簿的逐值核对。"""
    loc = pd.read_excel(R35D / "R35_COHORT_OUTCOMES.xlsx", sheet_name=None)
    dlv = pd.read_excel(HAND / "35" / "R35_CORRECTED_BENCHMARK.xlsx", sheet_name=["10_队列与结局", "11_未知与冲突登记"])
    def same(a, b):
        try:
            pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_dtype=False, check_exact=True); return True
        except AssertionError:
            return False
    chk = pd.DataFrame([dict(记录="1_队列与结局", 交付副本="35/R35_CORRECTED_BENCHMARK.xlsx ∷ 10_队列与结局", 行数=len(loc["1_队列与结局"]),
                             逐值相同=same(loc["1_队列与结局"], dlv["10_队列与结局"])),
                        dict(记录="2_未知与冲突登记", 交付副本="35/R35_CORRECTED_BENCHMARK.xlsx ∷ 11_未知与冲突登记", 行数=len(loc["2_未知与冲突登记"]),
                             逐值相同=same(loc["2_未知与冲突登记"][["项", "N", "数据库"]], dlv["11_未知与冲突登记"][["项", "N", "数据库"]]))])
    return loc["1_队列与结局"], loc["2_未知与冲突登记"], chk


# ── C1：纳排流程（论文与补充材料用）───────────────────────────────
STEP = {
    "mimic": [("分析单位：每次住院的首段 ICU 住院（icustays 按 hadm_id 取最早 intime）", "Analysis units: first ICU stay of each hospital admission"),
              ("入 ICU 后 24 h（landmark）时仍在该 ICU（outtime > intime + 24 h）", "Still in the ICU at the 24-h landmark"),
              ("＋ 年龄 ≥ 18 岁且年龄已知", "Aged ≥18 years with known age"),
              ("＋ landmark 前无已知院内死亡、死亡时点可靠、无死亡—出院时间冲突（最终合格队列）",
               "No known in-hospital death at or before the landmark; reliable outcome source (final eligible cohort)")],
    "eicu": [("分析单位：每次住院的首个 ICU 单元住院（unitvisitnumber = 1）", "Analysis units: first ICU unit stay of each hospital stay"),
             ("入 ICU 后 24 h（landmark）时仍在该单元（unitdischargeoffset > 1440 min）", "Still in the ICU at the 24-h landmark"),
             ("＋ 年龄 ≥ 18 岁且年龄已知（\u2018> 89\u2019 记为 90）", "Aged ≥18 years with known age"),
             ("＋ landmark 前无已知院内死亡、住院出院状态非缺失（最终合格队列）",
              "No known in-hospital death at or before the landmark; hospital discharge status recorded (final eligible cohort)")],
}


def c1_flow():
    rec, reg, _ = r35_record()
    rows = []
    for db, d in DBS.items():
        M, C = v5(db, "META"), r35(db).loc[v5(db, "META").index]
        r = rec[rec.数据库 == d["name"]].reset_index(drop=True)
        n_rec = [int(r.N.iloc[i]) for i in range(4)]
        e = M.eligible
        recomputed = [len(M), None, None, int(e.sum())]
        pts = [M.group_id.nunique(), None, None, int(M.loc[e, "group_id"].nunique())]
        hosp_unique = int(C[d["hosp_stay"]].nunique())
        for i, (zh, en) in enumerate(STEP[db]):
            rows.append(dict(数据库=d["name"], 步骤=i + 1, 步骤说明=zh, Step=en, 住院单位数=n_rec[i],
                             排除=None if i == 0 else n_rec[i - 1] - n_rec[i],
                             患者数=pts[i] if pts[i] is not None else np.nan,
                             从冻结数据重算=recomputed[i] if recomputed[i] is not None else np.nan,
                             重算与运行记录一致=(recomputed[i] == n_rec[i]) if recomputed[i] is not None else np.nan,
                             来源="R35 运行记录 + data v5 META 重算" if recomputed[i] is not None else "仅 R35 运行记录（冻结数据未保存该步的标志，无法重算）"))
        # 结局可评价人数与事件
        for oc, en in OC.items():
            ev = int((e & M[oc]).sum()); rr = r[r.项.str.startswith(oc)].iloc[0]
            rows.append(dict(数据库=d["name"], 步骤=np.nan, 步骤说明=f"结局可评价：{oc}", Step=f"Outcome evaluable: {en}", 住院单位数=int(e.sum()),
                             排除=0, 患者数=int(M.loc[e, "group_id"].nunique()), 从冻结数据重算=int(e.sum()),
                             重算与运行记录一致=(int(e.sum()) == int(rr.N)) and (ev == int(rr.事件)), 来源="data v5 META（结局无缺失）；与 R35 运行记录核对",
                             事件=ev, 事件率=ev / int(e.sum())))
        rows[-1]["住院单位与住院一一对应"] = rows[-2]["住院单位与住院一一对应"] = (hosp_unique == len(C))
    return pd.DataFrame(rows)


def c1b_exclusion_reasons():
    """各步排除原因：只写记录直接给出或由记录逻辑推出的；推不出的写不可恢复。"""
    rec, reg, _ = r35_record()
    out = []
    for db, d in DBS.items():
        r = rec[rec.数据库 == d["name"]].reset_index(drop=True); g = reg[reg.数据库 == d["name"]].set_index("项").N
        n = [int(r.N.iloc[i]) for i in range(4)]
        out.append(dict(数据库=d["name"], 步骤="1→2", 排除=n[0] - n[1], 原因=("landmark 时不在该 ICU：outtime ≤ intime + 24 h，或出 ICU 时间缺失（含 24 h 内出 ICU 或死亡）" if db == "mimic" else "landmark 时不在该单元：unitdischargeoffset ≤ 1440 min 或缺失（含 24 h 内出 ICU 或死亡）"), 分解=""
                        , 依据="R35 运行记录；该步无更细分解"))
        if db == "mimic":
            out.append(dict(数据库=d["name"], 步骤="2→3", 排除=n[1] - n[2], 原因="年龄 < 18 或年龄不明", 分解=f"全部分析单位中年龄不明 {int(g['年龄不明'])}",
                            依据="R35 运行记录"))
            z_conf = int(g["死亡时点晚于出院时间超1天（时间冲突）"]); z_src = int(g["hef=1 且替代后仍无死亡时点（结局来源不可靠，排除）"])
            out.append(dict(数据库=d["name"], 步骤="3→4", 排除=n[2] - n[3], 原因="landmark 前已发生院内死亡（死亡时点 ≤ landmark）",
                            分解=(f"全部 {n[2] - n[3]} 例：登记表中另两项排除原因（时间冲突、来源不可靠）在全部分析单位中均为 0"
                                if z_conf == 0 and z_src == 0 else "不可恢复"),
                            依据="由 R35 未知与冲突登记推出（逻辑推出，不是估计）"))
        else:
            out.append(dict(数据库=d["name"], 步骤="2→3", 排除=n[1] - n[2], 原因="年龄 < 18 或年龄不明",
                            分解="不可恢复（登记表只有全部分析单位中的年龄不明数，没有该步内的分解）", 依据="R35 运行记录"))
            out.append(dict(数据库=d["name"], 步骤="3→4", 排除=n[2] - n[3], 原因="landmark 前已院内死亡，或住院出院状态缺失",
                            分解="不可恢复（两项原因在全部分析单位中均非 0，冻结数据没有保存该步的逐条原因；不推算）", 依据="R35 运行记录"))
    return pd.DataFrame(out)


def c1c_registry():
    rec, reg, _ = r35_record()
    g = reg.copy(); g.insert(0, "说明", "全部分析单位中的计数（不以前序步骤为条件），R35 运行记录原样")
    return g[["说明", "数据库", "项", "N"]]


# ── C2：六个表型的最终任务 ─────────────────────────────────────────
def c2_tasks():
    r41 = pd.read_excel(ROOT / "22_R41_Stage0" / "R41_Stage0_证据表.xlsx", sheet_name="A1_任务与轨").set_index(["库", "结局", "表型"])
    r42 = pd.read_excel(ROOT / "23_R42_正式主实验" / "inference" / "R42_INFERENCE.xlsx", sheet_name="1_任务完整度").set_index(["库", "结局", "表型"])
    r47 = pd.read_excel(ROOT / "29_R47_写作交接" / "R47_冻结数字包.xlsx", sheet_name="B1_数据与任务概况")
    r47 = r47[r47.库.isin(DBS)].set_index(["库", "结局", "表型"])
    rows = []
    for db, d in DBS.items():
        M, P, OK = v5(db, "META"), v5(db, "PHENO"), v5(db, "PHENOOK")
        e = M.eligible
        for oc in OC:
            for ph in PH:
                t = e & OK[ph]; a = t & P[ph]; b = t & ~P[ph]; y = M[oc]
                k = (db, oc, ph)
                r = dict(数据库=d["name"], 结局=oc, 表型=ph, Phenotype=PH_EN[ph], 合格队列=int(e.sum()), 表型不可判定=int((e & ~OK[ph]).sum()),
                         任务n=int(t.sum()), 任务患者数=int(M.loc[t, "group_id"].nunique()), 事件=int((t & y).sum()),
                         A轨n=int(a.sum()), A轨事件=int((a & y).sum()), B轨n=int(b.sum()), B轨事件=int((b & y).sum()))
                r["事件率"] = r["事件"] / r["任务n"]; r["A轨占比"] = r["A轨n"] / r["任务n"]
                x = r41.loc[k]
                r["R41分组单位数"] = int(x.任务内划分分量数)
                r["与R41一致"] = (r["任务n"], r["事件"], r["A轨n"], r["A轨事件"], r["B轨n"], r["B轨事件"]) == \
                    (int(x.n), int(x.事件), int(x.全体A_n), int(x.全体A_事件), int(x.全体B_n), int(x.全体B_事件))
                r["与R42评价单位一致"] = r["任务n"] == int(r42.loc[k].n_units)
                y47 = r47.loc[k]
                r["与R47一致"] = (r["任务n"], r["事件"], r["A轨n"], r["A轨事件"], r["B轨n"], r["B轨事件"]) == \
                    (int(y47.n), int(y47.事件), int(y47.A轨n), int(y47.A轨事件), int(y47.B轨n), int(y47.B轨事件))
                rows.append(r)
    return pd.DataFrame(rows)


# ── C3：表型定义与 landmark 规则（直接取源码）──────────────────────────
def c3_definitions():
    import ast                                                     # 不导入 pgss_r35_qc（它在导入时要求数据库口令）；从源码语法树取函数文本
    text = (HERE / "pgss_r35_qc.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "phenotypes")
    src = ast.get_source_segment(text, fn)
    lines = {ph: " ".join(l.strip() for l in src.splitlines() if f'P["{ph}"]' in l or f'OK["{ph}"]' in l) for ph in PH}
    zh = {"女性": "性别为女（F）；性别缺失或非 F/M 不可判定", "年龄≥65": "年龄 ≥ 65 岁", "贫血": "血红蛋白：女 < 12 g/dL、男 < 13 g/dL；性别未知不可判定",
          "eGFR降低": "CKD-EPI 2021 免种族公式 eGFR < 60 mL/min/1.73 m²（landmark 时的血肌酐、年龄、性别；性别未知返回缺失）",
          "低白蛋白": "白蛋白 < 3.5 g/dL", "肥胖": "BMI ≥ 30 kg/m²（身高体重重算；BMI 限 10–80）"}
    rows = [dict(类别="表型", 项=ph, 定义=zh[ph], 源码=lines[ph], 代码位置="pgss_r35_qc.phenotypes（pgss_r383_build_v5 调用）") for ph in PH]
    man = json.loads((V5 / "DATA_VERSION.json").read_text(encoding="utf-8"))
    rows += [dict(类别="landmark", 项="入组时点", 定义="入 ICU 后 24 h；须在该时点仍在 ICU，且 landmark 前无已知院内死亡", 源码="MIMIC: st.outtime > st.lm（lm = intime + 24 h）；eICU: unitdischargeoffset > 1440", 代码位置="pgss_r35_cohort.py"),
             dict(类别="landmark", 项="预测变量时间窗（MIMIC，data v5）", 定义=man["v5窗口规则"]["体征体温化验"] + "；" + man["v5窗口规则"]["取值"], 源码="",
                  代码位置="pgss_r383_build_v5.py；DATA_VERSION.json v5窗口规则"),
             dict(类别="landmark", 项="预测变量（eICU）", 定义="data v5 未改动 eICU 输入（DATA_VERSION.json「未改动项」含 eICU 全部输入）", 源码="",
                  代码位置="DATA_VERSION.json"),
             dict(类别="结局", 项="OUT_HOSP_MORT", 定义="landmark 之后、同次住院内死亡。MIMIC：hospital_expire_flag = 1 且死亡时点 > landmark（死亡时点取 deathtime，缺失时取 dischtime）；"
                  "eICU：hospitaldischargestatus = Expired 且 hospitaldischargeoffset > 1440", 源码="", 代码位置="pgss_r35_cohort.build_mimic / build_eicu"),
             dict(类别="结局", 项="OUT_LOS72", 定义="ICU 停留 > 72 h（自入 ICU 起算）。MIMIC：icustays.los > 3 天；eICU：unitdischargeoffset > 4320 min", 源码="",
                  代码位置="pgss_r35_cohort.build_mimic / build_eicu")]
    return pd.DataFrame(rows)


def c4_units():
    rows = []
    for db, d in DBS.items():
        M, C = v5(db, "META"), r35(db).loc[v5(db, "META").index]
        e = M.eligible
        rows.append(dict(数据库=d["name"], 分析单位="ICU 住院（每次住院一段）", 住院键=d["hosp_stay"], 患者键=f"{d['pid']}（data v5 META.group_id）",
                         分析单位数=len(M), 住院数=int(C[d["hosp_stay"]].nunique()), 患者数=int(M.group_id.nunique()),
                         合格单位数=int(e.sum()), 合格住院数=int(C.loc[e, d["hosp_stay"]].nunique()), 合格患者数=int(M.loc[e, "group_id"].nunique()),
                         合格医院数=int(M.loc[e, "hospital_id"].nunique()) if M.hospital_id.notna().any() else np.nan,
                         推断的重抽单位="患者" if db == "mimic" else "患者—医院连通分量（全局注册，R41；不在本表重算）",
                         数据来源="19_R38/data_v5/R38_{0}_META.parquet；16_R35_FIX/{1}.parquet".format(db, d["r35"])))
    return pd.DataFrame(rows)


def build():
    S = {}
    S["A1_来源与版本"] = a_sources()
    S["A2_v5与R35队列逐条比较"] = a_v5_equals_r35()
    S["A3_R35运行记录与交付副本"] = r35_record()[2]
    S["C1_纳排流程"] = c1_flow()
    S["C1b_各步排除原因"] = c1b_exclusion_reasons()
    S["C1c_R35未知与冲突登记"] = c1c_registry()
    S["C2_表型任务"] = c2_tasks()
    S["C3_定义与landmark"] = c3_definitions()
    S["C4_计数口径"] = c4_units()
    return S


if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 60)
    S = build()
    for k, v in S.items():
        print("==", k, v.shape); print(v.to_string()[:3000])
