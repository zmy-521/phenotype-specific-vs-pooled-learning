# -*- coding: utf-8 -*-
r"""R46 只读更正与补表（GPT 第 50 轮 §3.4、§8）。不训练、不重抽。

1. NWICU 体征可用性：R45 的「窗内有体征」实为「窗内心率非缺失」；这里两种口径并列——窗内心率非缺失、窗内任一规定体征非缺失
   （心率、收缩压、舒张压、平均动脉压、呼吸频率、SpO2，取自已构建特征表 X），按首个 ICU 类型、锚定年份组与全体；
   住院死亡率按 R46 v2 可评价人次（不改入组与拟合）。
2. 各模型自身的校准：截距（CITL）、斜率、平均预测、观察事件率、Brier（每 seed 完整 OOF；3 seed 均值与范围），
   来源：MIMIC/eICU = R43 只读补齐「3_辅助指标逐seed」；NHANES = R46 v2 的 R44 推断；NWICU = R46 v2 推断；
   NHANES 目标一致性追加臂（P*_pop、HTFS_pop、P*_B、HTFS_B）= 从 R45/R46 目标单元按同一函数 pgss_r43_readonly.metrics 计算。
3. 预定判读范围更正：只有 O_MORT10×低白蛋白 的 C1、D1 有方案预定的解释性措辞；其余任务只报三态。
输出：28_R46_最终核验与T2/readonly/R46_READONLY.xlsx、.json
"""
import sys, json, pathlib
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pgss_threads  # noqa: F401
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd

ROOT = HERE.parent
R46 = ROOT / "28_R46_最终核验与T2"
OUT = R46 / "readonly"
SRC = pathlib.Path(r"E:/ICU_外部数据/nwicu-northwestern-icu/0.1.0/data")
VITALS = ["心率", "收缩压", "舒张压", "平均动脉压", "呼吸频率", "SpO2"]
ARMS = ["P*", "HTFS", "Hfull", "S0*", "SZ*"]
PRESPEC = {("C1", "O_MORT10", "低白蛋白"), ("D1", "O_MORT10", "低白蛋白")}


def vitals():
    D = ROOT / "27_R45_外部复现与机制核验" / "nwicu" / "data"
    X = pd.read_parquet(D / "X.parquet"); meta = pd.read_parquet(D / "meta.parquet")
    S = pd.read_parquet(R46 / "nwicu_v2" / "_local_only" / "MORT_STATUS.parquet")
    st = pd.read_csv(SRC / "nw_icu/icustays.csv.gz", usecols=["stay_id", "subject_id", "first_careunit"]).set_index("stay_id")
    p = pd.read_csv(SRC / "nw_hosp/patients.csv.gz", usecols=["subject_id", "anchor_year_group"]).set_index("subject_id")
    E = pd.DataFrame(index=X.index)
    E["first_careunit"] = st.first_careunit.reindex(E.index)
    E["anchor_year_group"] = p.anchor_year_group.reindex(st.subject_id.reindex(E.index)).to_numpy()
    E["窗内心率非缺失"] = X["心率"].notna()
    E["窗内任一规定体征非缺失"] = X[VITALS].notna().any(axis=1)
    E["可评价"] = S.住院死亡可评价.reindex(E.index).astype(bool)
    E["y"] = S.OUT_HOSP_MORT_v2.reindex(E.index)
    rows = []
    for by in ("first_careunit", "anchor_year_group", None):
        groups = E.groupby(by) if by else [("全部", E)]
        for k, h in groups:
            r = dict(分组方式=by or "全部", 组=k, 人次=len(h))
            for col in ("窗内心率非缺失", "窗内任一规定体征非缺失"):
                r[f"{col}_比例"] = float(h[col].mean())
                ev = h[h.可评价]
                r[f"住院死亡率_{col}"] = float(ev[ev[col]].y.mean()) if ev[col].any() else np.nan
                r[f"住院死亡率_非{col}"] = float(ev[~ev[col]].y.mean()) if (~ev[col]).any() else np.nan
            rows.append(r)
    T = pd.DataFrame(rows)
    per_var = pd.DataFrame([dict(体征=v, 窗内非缺失比例=float(X[v].notna().mean())) for v in VITALS])
    return T, per_var, dict(人次=len(E), 心率非缺失=float(E.窗内心率非缺失.mean()), 任一体征非缺失=float(E.窗内任一规定体征非缺失.mean()),
                            两口径不一致人次=int((E.窗内心率非缺失 != E.窗内任一规定体征非缺失).sum()))


def calib_from_aux(df, db_label):
    d = df[df.臂.isin(ARMS) & df.范围.isin(["A", "B", "全体"])]
    if "评价对象" in d:
        d = d[d.评价对象 == "全体"]
    g = d.groupby(["结局", "表型", "臂", "范围"])
    out = g.agg(seed数=("seed", "nunique"), n=("n", "mean"), 事件=("事件", "mean"), 平均预测=("平均预测", "mean"), 观察事件率=("观察事件率", "mean"),
                校准截距=("校准截距", "mean"), 校准截距_最小=("校准截距", "min"), 校准截距_最大=("校准截距", "max"),
                校准斜率=("校准斜率", "mean"), 校准斜率_最小=("校准斜率", "min"), 校准斜率_最大=("校准斜率", "max"),
                Brier=("Brier", "mean")).reset_index()
    out.insert(0, "库", db_label)
    return out


def target_calib():
    """NHANES 目标一致性追加臂的校准（同一 metrics 函数）；受影响任务读 R46 v2 单元。"""
    import pgss_r46_inference as RI, pgss_r44_nhanes_run as NR, pgss_r45_target as TG, pgss_r45_target_inference as TI
    import pgss_r44_nhanes_inference as NI, pgss_r43_readonly as RO
    RI.patch_arrays(NR, "nh44", False); RI.route(NR, "nh44", False); RI.route(TG, "nhpop", False)
    inc = json.loads((TG.OUT / "INCLUDED_TASKS.json").read_text(encoding="utf-8"))["纳入任务"]
    rows = []
    for oc, ph in inc:
        per44, _, _ = NI.collect(oc, ph)
        pq, mq, _ = TI.target_cols("pop", oc, ph)
        pb = TI.target_cols("B", oc, ph)[0] if (oc, ph) in TG.B_TASKS else {}
        for s in NR.SEEDS:
            d = per44[s][["y", "z"]].join(pq[s], how="left")
            if s in pb:
                d = d.join(pb[s], how="left")
            y = d.y.to_numpy(float); z = d.z.to_numpy(bool)
            for a in [c for c in d.columns if c not in ("y", "z")]:
                for sc, m in (("A", z), ("B", ~z), ("全体", np.ones_like(z))):
                    rows.append(dict(结局=oc, 表型=ph, seed=s, 臂=a, 范围=sc, **RO.metrics(y[m], d[a].to_numpy(float)[m])))
    return _agg_target(pd.DataFrame(rows))


def _agg_target(df):
    g = df.groupby(["结局", "表型", "臂", "范围"])
    out = g.agg(seed数=("seed", "nunique"), n=("n", "mean"), 事件=("事件", "mean"), 平均预测=("平均预测", "mean"), 观察事件率=("观察事件率", "mean"),
                校准截距=("校准截距", "mean"), 校准截距_最小=("校准截距", "min"), 校准截距_最大=("校准截距", "max"),
                校准斜率=("校准斜率", "mean"), 校准斜率_最小=("校准斜率", "min"), 校准斜率_最大=("校准斜率", "max"),
                Brier=("Brier", "mean")).reset_index()
    out.insert(0, "库", "nhanes（目标一致性追加臂）")
    return out


def calib():
    ro = pd.read_excel(ROOT / "24_R43_结果补齐与T1" / "readonly" / "R43_READONLY.xlsx", sheet_name="3_辅助指标逐seed")
    parts = [calib_from_aux(ro[ro.库 == db], db) for db in ("mimic", "eicu")]
    nh = pd.read_excel(R46 / "nhanes_v2" / "r44v2" / "inference" / "R44_INFERENCE.xlsx", sheet_name="6_辅助指标逐seed")
    nw = pd.read_excel(R46 / "nwicu_v2" / "inference" / "R45_NWICU_INFERENCE.xlsx", sheet_name="6_辅助指标逐seed")
    parts += [calib_from_aux(nh, "nhanes（R46 v2）"), calib_from_aux(nw, "nwicu（R46 v2）"), target_calib()]
    C = pd.concat(parts, ignore_index=True)
    C["平均预测减观察"] = C.平均预测 - C.观察事件率
    return C


def prespec():
    V = pd.read_excel(R46 / "nhanes_v2" / "target_v2" / "R45_TARGET_INFERENCE.xlsx", sheet_name="4_预定判读")
    V["方案预定的解释性判读"] = V.apply(lambda r: (r.对照, r.结局, r.表型) in PRESPEC, axis=1)
    V["报告用措辞"] = np.where(V.方案预定的解释性判读, V.预定措辞, "（未预定解释性判读，只报三态）")
    return V


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    VT, VV, vs = vitals()
    C = calib()
    V = prespec()
    summ = dict(体征=vs, 校准行数=len(C), 校准库=C.库.unique().tolist(), 预定判读行=len(V), 其中方案预定=int(V.方案预定的解释性判读.sum()))
    with pd.ExcelWriter(OUT / "R46_READONLY.xlsx", engine="openpyxl") as xw:
        VT.to_excel(xw, sheet_name="1_NWICU体征可用性两口径", index=False)
        VV.to_excel(xw, sheet_name="2_NWICU各体征窗内非缺失", index=False)
        C.to_excel(xw, sheet_name="3_各模型自身校准", index=False)
        V.to_excel(xw, sheet_name="4_预定判读范围更正", index=False)
    (OUT / "R46_READONLY.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
