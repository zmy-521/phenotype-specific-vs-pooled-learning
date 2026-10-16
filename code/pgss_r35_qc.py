# -*- coding: utf-8 -*-
"""R35：组装宽表 + 6 个主表型 + 逐变量 QC（IMPACT_LIST #19/#20/#22）。

关键修正
  #19 **恢复性别与年龄为预测变量**。R34 以"同义泄漏"为由排除性别/BMI/eGFR，这个理由是错的——
      预测变量定义了表型并不自动构成结局泄漏。本轮特征 = 性别(F指示) + 年龄 + 全部过 QC 的化验/体征。
      **仍不纳入**：BMI（缺失高）、衍生 eGFR（由已在特征里的肌酐/年龄/性别推出，避免重复）。
  #20 **性别三态**：F / M / 未知。未知者**不套男性标准**——
      贫血（WHO 性别特异切点）与 eGFR（性别特异公式）对其**不可判定**，单独登记，
      不计入该表型的分母。
  #22 取值在 SQL 内已先判有效再取末值，本模块不再做"取完再置缺失"。

用法：python pgss_r35_qc.py
"""
import pathlib, sys, warnings
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
HERE = pathlib.Path(__file__).parent; sys.path.insert(0, str(HERE))
F = HERE.parent / "16_R35_FIX"; RAW = F / "raw"
from pgss_r35_extract import LAB, CHART, E_VITAL

CONT_VARS = list(LAB) + list(CHART) + ["体温", "GCS总分", "年龄"]
MISS_MAX = 0.40                         # 两库都 <40% 缺失才进特征池


def build_mimic():
    co = pd.read_parquet(F / "R35_MIMIC_COHORT.parquet").set_index("stay_id")
    lab = pd.read_parquet(RAW / "mimic_lab.parquet")
    ch = pd.read_parquet(RAW / "mimic_chart.parquet")
    tp = pd.read_parquet(RAW / "mimic_temp.parquet").set_index("stay_id").tc.rename("体温")
    bd = pd.read_parquet(RAW / "mimic_body.parquet").set_index("stay_id")
    L = lab.pivot_table(index="stay_id", columns="var", values="valuenum", aggfunc="first")
    C = ch.pivot_table(index="stay_id", columns="var", values="valuenum", aggfunc="first")
    X = pd.concat([L, C, tp], axis=1).reindex(co.index)
    X["年龄"] = co.age
    X["GCS总分"] = np.nan                 # MIMIC GCS 本轮未重抽（见 §D：只重算受影响部分）
    X["身高cm"] = bd.height_cm.reindex(co.index)
    X["体重kg"] = bd.weight_kg.reindex(co.index)
    meta = pd.DataFrame(dict(sex=co.sex, eligible=co.eligible & co.mort_source_ok,
                            group_id=co.subject_id.astype(str), hospital_id=np.nan,
                            OUT_HOSP_MORT=co.OUT_HOSP_MORT, OUT_LOS72=co.OUT_LOS72))
    return X, meta


def build_eicu():
    co = pd.read_parquet(F / "R35_EICU_COHORT.parquet").set_index("sid")
    lab = pd.read_parquet(RAW / "eicu_lab.parquet")
    vi = pd.read_parquet(RAW / "eicu_vital.parquet")
    tg = pd.read_parquet(RAW / "eicu_temp_gcs.parquet").set_index("sid")
    L = lab.pivot_table(index="sid", columns="var", values="labresult", aggfunc="first")
    V = vi.pivot_table(index="sid", columns="var", values="val", aggfunc="first")
    X = pd.concat([L, V], axis=1).reindex(co.index)
    X["体温"] = tg.tc.reindex(co.index)
    X["GCS总分"] = tg.gcs.reindex(co.index)
    X["年龄"] = co.age
    X["身高cm"] = pd.to_numeric(co.height_raw, errors="coerce").where(
        lambda s: (s >= 100) & (s <= 230))
    X["体重kg"] = pd.to_numeric(co.weight_raw, errors="coerce").where(
        lambda s: (s >= 25) & (s <= 300))
    meta = pd.DataFrame(dict(sex=co.sex, eligible=co.eligible & co.mort_source_ok,
                            group_id=co.uniquepid.astype(str),
                            hospital_id=co.hospitalid.astype(str),
                            OUT_HOSP_MORT=co.OUT_HOSP_MORT, OUT_LOS72=co.OUT_LOS72))
    return X, meta


def ckd_epi_2021(scr, age, sex):
    """CKD-EPI 2021 race-free。**性别未知者返回 NaN**（#20：不套男性公式）"""
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
    OK["贫血"] = hgb.notna() & known      # 性别未知 ⇒ 不可判定（#20）
    P["eGFR降低"] = eg < 60;              OK["eGFR降低"] = eg.notna()
    P["低白蛋白"] = alb < 3.5;             OK["低白蛋白"] = alb.notna()
    P["肥胖"] = bmi >= 30;                OK["肥胖"] = bmi.notna()
    X = X.copy(); X["BMI"] = bmi; X["eGFR_landmark"] = eg
    return X, P.fillna(False).astype(bool), OK.fillna(False).astype(bool)


def main():
    res = {}
    for name, fn in (("MIMIC-IV", build_mimic), ("eICU-CRD", build_eicu)):
        print(f"组装 {name} …", flush=True)
        X, meta = fn()
        X, P, OK = phenotypes(X, meta)
        e = meta.eligible.fillna(False).to_numpy(bool)
        qc = []
        for v in CONT_VARS + ["身高cm", "体重kg", "BMI"]:
            if v not in X.columns: continue
            s = pd.to_numeric(X[v], errors="coerce")[e]
            qc.append(dict(数据库=name, 变量=v, 合格队列N=int(e.sum()),
                           非缺失=int(s.notna().sum()), 缺失率=round(float(s.isna().mean()), 4),
                           P1=round(float(s.quantile(.01)), 3) if s.notna().any() else None,
                           中位=round(float(s.median()), 3) if s.notna().any() else None,
                           P99=round(float(s.quantile(.99)), 3) if s.notna().any() else None,
                           聚合方式="[ICU入室, landmark) 内**最后一个有效**测量（范围过滤在 SQL 内完成）",
                           时间资格=("MIMIC 用 storetime（结果可获得时间）"
                                 if name.startswith("MIMIC") else
                                 "eICU 用 labresultoffset / observationoffset（该库无独立采样-可得二分字段）")))
        tag = "mimic" if name.startswith("MIMIC") else "eicu"
        X.to_parquet(F / f"R35_{tag}_X.parquet")
        P.to_parquet(F / f"R35_{tag}_PHENO.parquet")
        OK.to_parquet(F / f"R35_{tag}_PHENOOK.parquet")
        meta.to_parquet(F / f"R35_{tag}_META.parquet")
        res[name] = dict(qc=pd.DataFrame(qc), P=P, OK=OK, e=e, meta=meta, X=X)
        print(f"  {X.shape[0]:,} 单位 × {X.shape[1]} 列；合格 {int(e.sum()):,}", flush=True)
    Q = pd.concat([res[k]["qc"] for k in res], ignore_index=True)
    piv = Q.pivot_table(index="变量", columns="数据库", values="缺失率")
    piv["两库都可用"] = (piv.get("MIMIC-IV") < MISS_MAX) & (piv.get("eICU-CRD") < MISS_MAX)
    ph = []
    for k, r in res.items():
        for c in r["P"].columns:
            m = r["OK"][c].to_numpy(bool) & r["e"]
            ph.append(dict(数据库=k, 表型=c, 可判定N=int(m.sum()),
                           阳性N=int((r["P"][c].to_numpy(bool) & m).sum()),
                           阳性率=round(float((r["P"][c].to_numpy(bool) & m).sum() / max(m.sum(), 1)), 4),
                           不可判定N=int(r["e"].sum() - m.sum())))
    PH = pd.DataFrame(ph)
    feats = [v for v in piv.index[piv["两库都可用"] == True]
             if v not in ("BMI", "eGFR_landmark", "身高cm", "体重kg")]
    fl = pd.DataFrame(dict(特征=feats + ["性别(F指示)"]))
    with pd.ExcelWriter(F / "R35_DATA_QC.xlsx", engine="openpyxl") as xw:
        Q.to_excel(xw, sheet_name="1_逐变量QC", index=False)
        piv.reset_index().to_excel(xw, sheet_name="2_两库缺失率对照", index=False)
        PH.to_excel(xw, sheet_name="3_表型分布", index=False)
        fl.to_excel(xw, sheet_name="4_最终特征池", index=False)
    print(f"\n两库都可用的连续变量 {int(piv['两库都可用'].sum())} / {len(piv)}；"
          f"最终特征池 {len(feats)+1} 个（含性别）")
    print("\n=== 表型分布 ===\n" + PH.to_string(index=False))
    print(f"\n已写 {F/'R35_DATA_QC.xlsx'}")


if __name__ == "__main__":
    main()
