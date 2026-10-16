# -*- coding: utf-8 -*-
r"""R49 任务二（GPT 第 53 轮）：NWICU / NHANES 辅助指标的跨 seed 描述性汇总。只读已交付的逐 seed 结果表；不重新拟合、不重新校准、
不平均预测概率、不把 seed 当作独立样本（只报均值与最小–最大范围，不报跨 seed 的标准误或区间）。

输入（全部为已交付文件，逐字节核对）：
  · 有效版本：NWICU v2、NHANES v2（R46 代码包内的推断工作簿；住院死亡 / O_CVD10 为 R46 重跑，其余结局与 v1 逐值相同——本模块逐任务核对）；
  · v1：R45 NWICU 推断、R44 NHANES 推断（只用于判定哪些任务被 R46 重跑）；
  · 格式参照与验证：R43 B1/B2/B3/B4（ICU）——用本模块同一汇总函数从 R43 B1/B2 重算 B3/B4，必须与 R43 逐值相同；
  · 独立交叉核对：R46 X1「各模型自身校准全表」（另一份代码从预测直接算的跨 seed 均值、最小、最大）。
NHANES 的结果知情后目标一致性追加臂单列（D5），不与原始分析混在一起；pop = 研究样本比例加权（按记录等权），不是 NHANES 调查权重。
"""
import sys, pathlib, hashlib, zipfile, io
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd

ROOT = HERE.parent
HAND = pathlib.Path(r"D:/Research/3_对话与交接/GPT交接_PGSS")
R46D = ROOT / "28_R46_最终核验与T2"
Z46 = HAND / "46" / "R46_CODE_TESTS.zip"
INP = {
    "NWICU_v2": (R46D / "nwicu_v2" / "inference" / "R45_NWICU_INFERENCE.xlsx", ("zip", Z46, "r46/nwicu_v2/inference/R45_NWICU_INFERENCE.xlsx")),
    "NHANES_v2": (R46D / "nhanes_v2" / "r44v2" / "inference" / "R44_INFERENCE.xlsx", ("zip", Z46, "r46/nhanes_v2/r44v2/inference/R44_INFERENCE.xlsx")),
    "NWICU_v1": (ROOT / "27_R45_外部复现与机制核验" / "nwicu" / "inference" / "R45_NWICU_INFERENCE.xlsx",
                 ("zip", HAND / "45" / "R45_CODE_TESTS.zip", "r45/nwicu/inference/R45_NWICU_INFERENCE.xlsx")),
    "NHANES_v1": (ROOT / "25_R44_NHANES跨领域复现" / "inference" / "R44_INFERENCE.xlsx", ("zip", HAND / "44" / "R44_CODE_TESTS.zip", "r44/inference/R44_INFERENCE.xlsx")),
    "R43证据表": (ROOT / "24_R43_结果补齐与T1" / "R43_结果补齐与T1_证据表.xlsx", ("file", HAND / "43" / "R43_结果补齐与T1_证据表.xlsx")),
    "R46证据表": (R46D / "R46_最终核验与T2_证据表.xlsx", ("file", HAND / "46" / "R46_最终核验与T2_证据表.xlsx")),
}
MET = ["AUROC", "AUPRC", "Brier", "logloss", "校准截距", "校准斜率"]
ARMS = ["P*", "HTFS", "Hfull", "S0*", "SZ*"]
SCOPE_ORDER = ["全体", "A", "B", "两轨等权"]
X1_LAB = {"mimic": "mimic", "eicu": "eicu", "nwicu": "nwicu（R46 v2）", "nhanes": "nhanes（R46 v2）"}
_C = {}


def sha16(b):
    return hashlib.sha256(b).hexdigest()[:16]


def delivered(key):
    ref = INP[key][1]
    if ref[0] == "file":
        return ref[1].read_bytes()
    z = zipfile.ZipFile(ref[1])
    return z.read(ref[2]) if ref[2] in z.namelist() else None


def rd(key, sheet):
    k = (key, sheet)
    if k not in _C:
        _C[k] = pd.read_excel(INP[key][0], sheet_name=sheet)
    return _C[k].copy()


def a_inputs():
    rows = []
    for k, (p, ref) in INP.items():
        b, rb = p.read_bytes(), delivered(k)
        where = str(ref[1].relative_to(HAND)) + ("" if ref[0] == "file" else f" ∷ {ref[2]}")
        rows.append(dict(输入=k, 本地路径=str(p.relative_to(ROOT)), 本地SHA16=sha16(b), 交付副本=where if rb is not None else where + "（交付包内无此成员）",
                         交付SHA16=sha16(rb) if rb is not None else "", 逐字节相同=(rb is not None and b == rb)))
    return pd.DataFrame(rows)


# ── 汇总函数（同一函数用于 R43 验证与 NWICU/NHANES）───────────────
def r43_b3(P):
    """R43 B3 的定义（pgss_r43_report.py 原样）：按 候选×参照×范围，跨任务×seed 取差值中位数与「差 > 0」的个数。"""
    pr = []
    for (c, r, sc), h in P.groupby(["候选", "参照", "范围"]):
        row = dict(候选=c, 参照=r, 范围=sc, 任务乘seed=len(h))
        for k in ("Brier", "logloss", "AUROC", "AUPRC", "校准截距", "校准斜率"):      # R43 的列顺序
            col = f"{k}_候选减参照"
            if col in h and h[col].notna().any():
                row[f"{k}差中位"] = h[col].median(); row[f"{k}差为正"] = int((h[col] > 0).sum())
        pr.append(row)
    return pd.DataFrame(pr)


def r43_b4(A, arms):
    """R43 B4 的定义（pgss_r43_report.py 原样）：A、B 两轨的逐任务×seed 校准截距与斜率的分布。"""
    cal = (A[A.范围.isin(["A", "B"]) & A.臂.isin(arms)].groupby("臂")[["校准截距", "校准斜率"]].describe(percentiles=[.05, .5, .95]).reset_index())
    cal.columns = ["_".join([str(x) for x in c if x]) for c in cal.columns]
    return cal


def per_task(A, keys, cols):
    """逐任务跨 3 个 seed：均值、最小、最大（不报跨 seed 的标准误或区间）。"""
    g = A.groupby(keys, sort=False, observed=True)
    out = g.size().rename("seed数").reset_index()
    for c in cols:
        s = g[c]
        out[f"{c}_均值"] = s.mean().to_numpy(); out[f"{c}_最小"] = s.min().to_numpy(); out[f"{c}_最大"] = s.max().to_numpy()
    return out


def aux(db):
    if db in ("mimic", "eicu"):
        a = rd("R43证据表", "B1_辅助指标逐seed"); a = a[(a.库 == db) & (a.评价对象 == "全体") & a.臂.isin(ARMS)]
        return a
    return rd({"nwicu": "NWICU_v2", "nhanes": "NHANES_v2"}[db], "6_辅助指标逐seed")


def pairs(db):
    if db in ("mimic", "eicu"):
        p = rd("R43证据表", "B2_配对差逐seed"); return p[(p.库 == db) & p.候选.isin(ARMS) & p.参照.isin(ARMS)]
    return rd({"nwicu": "NWICU_v2", "nhanes": "NHANES_v2"}[db], "7_配对差逐seed")


DBL = {"mimic": "MIMIC-IV", "eicu": "eICU-CRD", "nwicu": "NWICU", "nhanes": "NHANES"}
SRC = {"mimic": "R43 B1/B2（评价对象 = 全体）", "eicu": "R43 B1/B2（评价对象 = 全体）", "nwicu": "NWICU v2（R46）", "nhanes": "NHANES v2（R46）"}


def d0_versions():
    """v2 与 v1 逐任务比较：哪些任务被 R46 重跑（逐 seed、全部臂与范围、全部指标）。"""
    rows = []
    for db, v2, v1 in (("nwicu", "NWICU_v2", "NWICU_v1"), ("nhanes", "NHANES_v2", "NHANES_v1")):
        for sheet in ("6_辅助指标逐seed", "7_配对差逐seed"):
            a, b = rd(v2, sheet), rd(v1, sheet)
            key = ["结局", "表型", "seed"] + (["臂", "范围", "评价对象"] if sheet.startswith("6") else ["候选", "参照", "范围"])
            num = [c for c in a.columns if c not in key + ["库"] and a[c].dtype.kind in "fi"]
            m = a.merge(b, on=key, how="outer", suffixes=("_v2", "_v1"), indicator=True)
            for (oc, ph), h in m.groupby(["结局", "表型"]):
                both = (h._merge == "both").all()
                same = both and all(np.array_equal(h[f"{c}_v2"].to_numpy(float), h[f"{c}_v1"].to_numpy(float), equal_nan=True) for c in num)
                rows.append(dict(数据库=DBL[db], 表=sheet, 结局=oc, 表型=ph, 行数_v2=int((h._merge != "right_only").sum()),
                                 行数_v1=int((h._merge != "left_only").sum()), 与v1逐值相同=bool(same),
                                 版本判定="与 v1 相同（未重跑，沿用既有有效结果）" if same else "R46 重跑（v2 为有效版本）"))
    return pd.DataFrame(rows)


def d1_models():
    out = []
    for db in ("mimic", "eicu", "nwicu", "nhanes"):
        a = aux(db).copy()
        a["范围"] = pd.Categorical(a.范围, SCOPE_ORDER); a["臂"] = pd.Categorical(a.臂, ARMS)
        a = a.sort_values(["结局", "表型", "臂", "范围", "seed"])
        t = per_task(a, ["结局", "表型", "臂", "范围"], MET + ["平均预测"])
        nn = a.groupby(["结局", "表型", "臂", "范围"], sort=False, observed=True).agg(n=("n", "first"), n恒定=("n", lambda s: s.nunique(dropna=False) == 1),
                                                                                  事件=("事件", "first"))
        t = t.merge(nn.reset_index(), on=["结局", "表型", "臂", "范围"], how="left")
        t.insert(0, "数据库", DBL[db]); t.insert(1, "来源", SRC[db])
        out.append(t)
    d = pd.concat(out, ignore_index=True)
    d["臂"] = d.臂.astype(str); d["范围"] = d.范围.astype(str)
    return d[d.seed数 > 0].reset_index(drop=True)


def d2_pairs():
    out = []
    cols = [f"{k}_候选减参照" for k in MET] + ["Brier_相对改善", "logloss_相对改善"]
    for db in ("mimic", "eicu", "nwicu", "nhanes"):
        p = pairs(db).copy()
        p["范围"] = pd.Categorical(p.范围, SCOPE_ORDER)
        p = p.sort_values(["结局", "表型", "候选", "参照", "范围", "seed"])
        t = per_task(p, ["结局", "表型", "候选", "参照", "范围"], [c for c in cols if c in p.columns])
        t.insert(0, "数据库", DBL[db]); t.insert(1, "来源", SRC[db])
        out.append(t)
    d = pd.concat(out, ignore_index=True); d["范围"] = d.范围.astype(str)
    return d[d.seed数 > 0].reset_index(drop=True)


def d3_b3():
    icu = r43_b3(rd("R43证据表", "B2_配对差逐seed")); icu.insert(0, "数据库", "MIMIC-IV + eICU-CRD（= R43 B3）")
    nw = r43_b3(rd("NWICU_v2", "7_配对差逐seed")); nw.insert(0, "数据库", "NWICU（v2）")
    nh = r43_b3(rd("NHANES_v2", "7_配对差逐seed")); nh.insert(0, "数据库", "NHANES（v2）")
    d = pd.concat([icu, nw, nh], ignore_index=True)
    d.insert(1, "说明", "任务×seed 合并后的描述：差 = 候选 − 参照；Brier、logloss 差 > 0 表示候选更差；AUROC、AUPRC 差 > 0 表示候选更好；seed 不是独立样本")
    return d


def d4_b4():
    icu = r43_b4(rd("R43证据表", "B1_辅助指标逐seed"), ["P*", "HTFS", "Hfull", "S0*", "SZ*", "CERT", "P*_F"]); icu.insert(0, "数据库", "MIMIC-IV + eICU-CRD（= R43 B4）")
    nw = r43_b4(rd("NWICU_v2", "6_辅助指标逐seed"), ARMS); nw.insert(0, "数据库", "NWICU（v2）")
    nh = r43_b4(rd("NHANES_v2", "6_辅助指标逐seed"), ARMS); nh.insert(0, "数据库", "NHANES（v2）")
    d = pd.concat([icu, nw, nh], ignore_index=True)
    d.insert(1, "说明", "A、B 两轨的逐任务×seed 取值分布（与 R43 B4 同一定义）")
    return d


def d5_target():
    x = rd("R46证据表", "X1_各模型自身校准_全表")
    t = x[x.库 == "nhanes（目标一致性追加臂）"].copy()
    t.insert(0, "性质", "结果知情后登记的目标一致性追加对照（L3），与原始 NHANES 分析分开；pop = 研究样本比例加权，不是 NHANES 调查权重")
    t.insert(1, "来源", "R46 证据表 X1（各臂跨 seed 的校准均值、最小、最大，原样）")
    return t.reset_index(drop=True)


# ── 核对 ─────────────────────────────────────────────────────────
def same_frame(a, b, tol=0.0):
    a, b = a.reset_index(drop=True), b.reset_index(drop=True)
    if list(a.columns) != list(b.columns) or a.shape != b.shape:
        return False, np.inf
    worst = 0.0
    for c in a.columns:
        x, y = a[c], b[c]
        if x.dtype.kind in "fi" and y.dtype.kind in "fi":
            xv, yv = x.to_numpy(float), y.to_numpy(float)
            if not np.array_equal(np.isnan(xv), np.isnan(yv)):
                return False, np.inf
            m = ~np.isnan(xv)
            worst = max(worst, float(np.max(np.abs(xv[m] - yv[m]))) if m.any() else 0.0)
        elif not (x.astype(str).to_numpy() == y.astype(str).to_numpy()).all():
            return False, np.inf
    return worst <= tol, worst


def a_validate_r43():
    b3_ok, b3_d = same_frame(r43_b3(rd("R43证据表", "B2_配对差逐seed")), rd("R43证据表", "B3_配对差汇总"), tol=1e-12)
    b4_ok, b4_d = same_frame(tol=1e-12, a=r43_b4(rd("R43证据表", "B1_辅助指标逐seed"), ["P*", "HTFS", "Hfull", "S0*", "SZ*", "CERT", "P*_F"]),
                             b=rd("R43证据表", "B4_校准诊断汇总"))
    return pd.DataFrame([dict(检查="本模块汇总函数从 R43 B2 重算 B3", 一致=b3_ok, 容差="≤ 1e-12（中位数的浮点舍入；计数列整数相同）", 最大绝对差=b3_d),
                         dict(检查="本模块汇总函数从 R43 B1 重算 B4", 一致=b4_ok, 容差="≤ 1e-12（describe 的浮点求和顺序）", 最大绝对差=b4_d)])


def a_cross_x1(d1):
    """D1 的跨 seed 均值/最小/最大，与 R46 X1（另一份代码从预测直接计算）逐项比较：校准截距、校准斜率、Brier，范围 全体/A/B。"""
    x = rd("R46证据表", "X1_各模型自身校准_全表")
    rows = []
    for db, lab in X1_LAB.items():
        key = ["结局", "表型", "臂", "范围"]
        xa = x[x.库 == lab].rename(columns=lambda c: c if c in key else f"X1_{c}")
        d = d1[(d1.数据库 == DBL[db]) & d1.范围.isin(["全体", "A", "B"])]
        m = d.merge(xa, on=key, how="inner")
        pairs_ = [("校准截距_均值", "X1_校准截距"), ("校准截距_最小", "X1_校准截距_最小"), ("校准截距_最大", "X1_校准截距_最大"), ("校准斜率_均值", "X1_校准斜率"),
                  ("校准斜率_最小", "X1_校准斜率_最小"), ("校准斜率_最大", "X1_校准斜率_最大"), ("Brier_均值", "X1_Brier")]
        worst = max(float(np.nanmax(np.abs(m[a].to_numpy(float) - m[b].to_numpy(float)))) for a, b in pairs_) if len(m) else np.nan
        rows.append(dict(数据库=DBL[db], R46_X1行=len(xa), 匹配行=len(m), 本表可比行=len(d), 最大绝对差=worst, 通过=bool(len(m) > 0 and worst < 1e-12)))
    return pd.DataFrame(rows)


def build():
    S = {}
    S["A4_任务二输入与交付副本"] = a_inputs()
    S["D0_版本判定"] = d0_versions()
    S["D1_各模型跨seed汇总"] = d1_models()
    S["D2_配对差跨seed汇总"] = d2_pairs()
    S["D3_配对差汇总_R43格式"] = d3_b3()
    S["D4_校准诊断汇总_R43格式"] = d4_b4()
    S["D5_NHANES目标一致性追加臂"] = d5_target()
    S["A5_R43格式验证"] = a_validate_r43()
    S["A6_与R46_X1交叉核对"] = a_cross_x1(S["D1_各模型跨seed汇总"])
    return S


if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 50)
    S = build()
    for k, v in S.items():
        print("==", k, v.shape)
        if k.startswith("A") or k == "D0_版本判定":
            print(v.to_string()[:4000])
    print(S["D1_各模型跨seed汇总"].groupby(["数据库"]).size())
    print(S["D3_配对差汇总_R43格式"].groupby("数据库").size())
